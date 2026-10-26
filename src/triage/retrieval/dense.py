"""Dense Vector Retrieval using Sentence-Transformers (all-MiniLM-L6-v2).

Computes dense embeddings for issue title and problem text,
builds a cosine similarity index, and performs nearest-neighbour retrieval.
TEMPORAL LEAKAGE RULE:
Restricted strictly to tickets created BEFORE the query ticket's created_at timestamp.
Evaluates Recall@5 on gold duplicate pairs.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_FEATURES_PATH = os.path.join("data", "interim", "features.parquet")
DEFAULT_INDEX_DIR = os.path.join("data", "interim", "embeddings")
DEFAULT_EMBEDDINGS_FILE = os.path.join(DEFAULT_INDEX_DIR, "minilm_embeddings.npy")
DEFAULT_METADATA_FILE = os.path.join(DEFAULT_INDEX_DIR, "index_metadata.parquet")
MODEL_NAME = "all-MiniLM-L6-v2"


class DenseTicketRetriever:
    """Dense vector retriever using sentence embeddings with chronological temporal filtering."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.metadata_df: Optional[pd.DataFrame] = None

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build_or_load_index(
        self,
        features_path: str = DEFAULT_FEATURES_PATH,
        index_dir: str = DEFAULT_INDEX_DIR,
        force_rebuild: bool = False,
    ) -> None:
        """Build or load the embedding matrix and ticket metadata."""
        os.makedirs(index_dir, exist_ok=True)
        emb_file = os.path.join(index_dir, "minilm_embeddings.npy")
        meta_file = os.path.join(index_dir, "index_metadata.parquet")

        if not force_rebuild and os.path.exists(emb_file) and os.path.exists(meta_file):
            print(f"Loading dense embeddings from {emb_file}...", flush=True)
            self.embeddings = np.load(emb_file)
            self.metadata_df = pd.read_parquet(meta_file)
            print(f"Loaded {len(self.metadata_df)} indexed tickets.", flush=True)
            return

        print(f"Building dense embeddings using {self.model_name}...", flush=True)
        df = pd.read_parquet(features_path)
        df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
        df.sort_values("created_at_dt", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Corpus text: Title + Clean Problem Text
        titles = df["title"].fillna("").astype(str)
        problems = df["clean_problem_text"].fillna("").astype(str)
        corpus = [f"{t}\n{p}"[:1000] for t, p in zip(titles, problems)]

        model = self._get_model()
        t0 = time.time()
        # Normalize embeddings for fast cosine similarity via dot product
        embeddings = model.encode(
            corpus, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        self.embeddings = np.array(embeddings, dtype=np.float32)

        meta_cols = [
            "ticket_number",
            "created_at",
            "created_at_dt",
            "title",
            "state",
            "integration_clean",
        ]
        self.metadata_df = df[meta_cols].copy()

        np.save(emb_file, self.embeddings)
        self.metadata_df.to_parquet(meta_file, index=False)
        print(
            f"Index built and saved in {round(time.time() - t0, 1)}s. Total vectors: {len(self.embeddings)}",
            flush=True,
        )

    def retrieve_similar(
        self,
        query_text: str,
        query_created_at: Optional[pd.Timestamp] = None,
        query_ticket_number: Optional[int] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar prior tickets.
        STRICT TEMPORAL RULE: Only tickets with created_at < query_created_at are eligible.
        """
        if self.embeddings is None or self.metadata_df is None:
            raise ValueError("Index not initialized. Call build_or_load_index() first.")

        model = self._get_model()
        query_emb = model.encode([query_text], normalize_embeddings=True)[0]

        # Filter mask: prior tickets only
        mask = np.ones(len(self.metadata_df), dtype=bool)
        if query_created_at is not None:
            query_ts = pd.to_datetime(query_created_at)
            mask = mask & (self.metadata_df["created_at_dt"] < query_ts).values

        if query_ticket_number is not None:
            mask = mask & (self.metadata_df["ticket_number"] != query_ticket_number).values

        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return []

        # Cosine similarity via dot product
        candidate_embeddings = self.embeddings[valid_indices]
        sims = np.dot(candidate_embeddings, query_emb)

        # Top K
        k = min(top_k, len(sims))
        top_sub_indices = np.argsort(sims)[::-1][:k]
        top_real_indices = valid_indices[top_sub_indices]

        results = []
        for real_idx, sub_idx in zip(top_real_indices, top_sub_indices):
            row = self.metadata_df.iloc[real_idx]
            results.append(
                {
                    "ticket_number": int(row["ticket_number"]),
                    "title": str(row["title"]),
                    "state": str(row["state"]),
                    "integration": str(row["integration_clean"]),
                    "similarity_score": round(float(sims[sub_idx]), 4),
                }
            )

        return results

    def evaluate_recall_at_k(self, k: int = 5) -> Dict[str, Any]:
        """
        Evaluate Recall@K on tickets closed as duplicate that reference the original issue.
        """
        if self.embeddings is None or self.metadata_df is None:
            raise ValueError("Index not loaded.")

        comments_path = os.path.join("data", "interim", "comments.parquet")
        duplicate_pairs: List[Tuple[int, int]] = []

        if os.path.exists(comments_path):
            df_comms = pd.read_parquet(comments_path)
            with_dups = df_comms[df_comms["duplicate_of_issue"].notna()]
            for _, r in with_dups.iterrows():
                duplicate_pairs.append((int(r["issue_number"]), int(r["duplicate_of_issue"])))

        # Also check state_reason == duplicate
        dup_tickets = self.metadata_df[
            self.metadata_df["ticket_number"].isin([p[0] for p in duplicate_pairs])
        ]
        if dup_tickets.empty:
            # Fallback to tickets closed with duplicate reference
            return {
                "evaluated_pairs": 0,
                "recall_at_5": 0.0,
                "note": "No ground truth duplicate pairs found",
            }

        hits = 0
        evaluated = 0

        # Create quick mapping of ticket_number -> real_idx
        ticket_to_idx = {int(t): i for i, t in enumerate(self.metadata_df["ticket_number"])}

        for dup_num, orig_num in duplicate_pairs:
            if dup_num not in ticket_to_idx or orig_num not in ticket_to_idx:
                continue

            dup_idx = ticket_to_idx[dup_num]
            orig_row = self.metadata_df.iloc[ticket_to_idx[orig_num]]
            dup_row = self.metadata_df.iloc[dup_idx]

            # Original must be created before duplicate
            if orig_row["created_at_dt"] >= dup_row["created_at_dt"]:
                continue

            # Query duplicate issue text against past tickets
            dup_emb = self.embeddings[dup_idx]
            query_ts = dup_row["created_at_dt"]

            mask = (self.metadata_df["created_at_dt"] < query_ts).values & (
                self.metadata_df["ticket_number"] != dup_num
            ).values
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                continue

            sims = np.dot(self.embeddings[valid_indices], dup_emb)
            top_sub = np.argsort(sims)[::-1][:k]
            retrieved_numbers = self.metadata_df.iloc[valid_indices[top_sub]][
                "ticket_number"
            ].values

            evaluated += 1
            if orig_num in retrieved_numbers:
                hits += 1

        recall = round(float(hits / evaluated), 4) if evaluated > 0 else 0.0
        return {
            "evaluated_pairs": evaluated,
            "hits_at_k": hits,
            "recall_at_5": recall,
            "k": k,
        }
