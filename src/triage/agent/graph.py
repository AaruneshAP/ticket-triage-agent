"""LangGraph Agent Pipeline for GitHub Ticket Triage.

Orchestrates the intelligent triage workflow via StateGraph:
1. parse_ticket: Extracts structured form headings and diagnostic indicators.
2. completeness_check: Evaluates missing logs, version fields, and regression signals.
3. predict: Invokes ML models (Survival AFT with Low Confidence flag, Not-Planned with SHAP, Label Recommender).
4. retrieve_similar: Finds top similar prior tickets via dense cosine retrieval with temporal guard.
5. draft_reply: Synthesizes polite, non-promising response requesting missing info and citing similar issues.
6. output: Formats final schema for FastAPI and Streamlit UI.
"""

import logging
import os
from typing import Any, Dict, List, TypedDict

import joblib
import pandas as pd
from langgraph.graph import END, StateGraph

from triage.agent.completeness import check_ticket_completeness
from triage.agent.llm import call_llm_draft
from triage.agent.tracer import trace_agent_run, trace_node
from triage.features.build_features import build_feature_dataframe
from triage.parse.form_parser import parse_issue
from triage.retrieval.dense import DenseTicketRetriever

logger = logging.getLogger(__name__)

MODELS_DIR = "models"


class TriageState(TypedDict, total=False):
    # Inputs
    raw_issue: Dict[str, Any]

    # Node 1: Parsed data
    parsed_issue: Dict[str, Any]
    features_df: Any

    # Node 2: Completeness
    missing_info: List[str]

    # Node 3: Predictions
    predicted_days_to_close: Dict[str, Any]
    fixed_horizon_probabilities: Dict[str, Any]
    p_not_planned: float
    p_not_planned_confidence: str
    top_shap_features: List[Dict[str, Any]]
    suggested_label: str
    label_confidence: float

    # Node 4: Retrieved peer issues
    similar_tickets: List[Dict[str, Any]]

    # Node 5: Draft reply
    draft_reply: str

    # Node 6: Final Output
    output: Dict[str, Any]


class TriageAgentGraph:
    """Compiled LangGraph agent workflow for ticket triage."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.survival_model = None
        self.fixed_horizon_14d = None
        self.fixed_horizon_30d = None
        self.not_planned_model = None
        self.labeler_model = None
        self.retriever = None
        self._load_dependencies()
        self.app = self._build_graph()

    def _load_dependencies(self) -> None:
        """Load trained models and vector retrieval index."""
        aft_path = os.path.join(self.models_dir, "survival_aft.joblib")
        fh_14_path = os.path.join(self.models_dir, "fixed_horizon_14d.joblib")
        fh_30_path = os.path.join(self.models_dir, "fixed_horizon_30d.joblib")
        np_path = os.path.join(self.models_dir, "not_planned.joblib")
        lbl_path = os.path.join(self.models_dir, "labeler.joblib")

        if os.path.exists(aft_path):
            self.survival_model = joblib.load(aft_path)
        if os.path.exists(fh_14_path):
            self.fixed_horizon_14d = joblib.load(fh_14_path)
        if os.path.exists(fh_30_path):
            self.fixed_horizon_30d = joblib.load(fh_30_path)
        if os.path.exists(np_path):
            self.not_planned_model = joblib.load(np_path)
        if os.path.exists(lbl_path):
            self.labeler_model = joblib.load(lbl_path)

        # Initialize retriever
        self.retriever = DenseTicketRetriever()
        try:
            self.retriever.build_or_load_index()
        except Exception as e:
            logger.warning(f"Could not load retriever index: {e}")

    def node_parse_ticket(self, state: TriageState) -> Dict[str, Any]:
        """Node 1: Parse raw issue body and extract form fields."""
        with trace_node("parse_ticket"):
            raw = state["raw_issue"]
            body = raw.get("body", "") or ""
            title = raw.get("title", "") or ""
            parsed = parse_issue(body, title)

            # Extract integration from labels if not already explicitly present
            integration = raw.get("integration") or ""
            labels = raw.get("labels", [])
            if not integration and isinstance(labels, list):
                for lbl in labels:
                    name = lbl.get("name", "") if isinstance(lbl, dict) else str(lbl)
                    if name.lower().startswith("integration:"):
                        integration = name.split(":", 1)[1].strip()
                        break

            combined = {
                **raw,
                **parsed,
                "integration": integration if integration else None,
                "form_problem": parsed.get("problem"),
                "form_version": parsed.get("version"),
                "form_last_working_version": parsed.get("last_working_version"),
                "form_install_type": parsed.get("install_type"),
                "form_integration": parsed.get("integration_in_form"),
            }
            single_df = pd.DataFrame([combined])
            features_df = build_feature_dataframe(single_df)

            return {"parsed_issue": combined, "features_df": features_df}

    def node_completeness_check(self, state: TriageState) -> Dict[str, Any]:
        """Node 2: Evaluate missing fields and diagnostic logs."""
        with trace_node("completeness_check"):
            missing = check_ticket_completeness(state["parsed_issue"])
            return {"missing_info": missing}

    def node_predict(self, state: TriageState) -> Dict[str, Any]:
        """Node 3: Predict time to close, not-planned risk, and label recommendation."""
        with trace_node("predict_models"):
            df = state["features_df"]
            parsed = state["parsed_issue"]

            # 1. Resolution Latency (Suppressed: Negative result with no out-of-sample predictive signal)
            fixed_horizon = {
                "p_closed_within_14d": None,
                "p_closed_within_30d": None,
                "roc_auc_14d": 0.4621,
                "roc_auc_30d": 0.5329,
                "confidence": "none",
                "confidence_reason": (
                    "No reliable signal for resolution time: out-of-sample ROC-AUC is 0.4621 (14d) and "
                    "0.5329 (30d), with bootstrap CIs crossing 0.50. Actionable closure probability display is suppressed."
                ),
                "status": "no reliable signal for resolution time",
            }

            predicted_days = {
                "p50": None,
                "p90": None,
                "confidence": "low",
                "confidence_reason": (
                    "No reliable signal for resolution time: out-of-sample survival C-index is 0.514 "
                    "(95% CI [0.492, 0.536], indistinguishable from chance 0.50); the cause was not investigated."
                ),
                "status": "no reliable signal for resolution time",
            }

            # 2. Not-planned risk
            if self.not_planned_model is not None:
                p_np_arr = self.not_planned_model.predict_proba(df)
                p_not_planned = round(float(p_np_arr[0]), 3)
                shap_explanations = self.not_planned_model.explain_ticket(df, top_k=4)
            else:
                p_not_planned = 0.163
                shap_explanations = []

            if p_not_planned >= 0.40:
                np_conf = "high"
            elif p_not_planned <= 0.10:
                np_conf = "moderate"
            else:
                np_conf = "low"

            # 3. Label recommendation
            existing_integ = parsed.get("integration") or df["integration_clean"].iloc[0]
            if existing_integ and existing_integ != "unlabeled":
                suggested_label = f"integration: {existing_integ}"
                label_conf = 1.0
            elif self.labeler_model is not None:
                clean_text = df["clean_text_for_label"].iloc[0]
                labels, confs, _ = self.labeler_model.predict_with_confidence(
                    pd.Series([clean_text])
                )
                suggested_label = f"integration: {labels[0]}" if labels[0] != "other" else "other"
                label_conf = round(float(confs[0]), 3)
            else:
                suggested_label = "other"
                label_conf = 0.5

            return {
                "predicted_days_to_close": predicted_days,
                "fixed_horizon_probabilities": fixed_horizon,
                "p_not_planned": p_not_planned,
                "p_not_planned_confidence": np_conf,
                "top_shap_features": shap_explanations,
                "suggested_label": suggested_label,
                "label_confidence": label_conf,
            }

    def node_retrieve_similar(self, state: TriageState) -> Dict[str, Any]:
        """Node 4: Dense cosine retrieval with temporal restriction."""
        with trace_node("retrieve_similar"):
            parsed = state["parsed_issue"]
            title = parsed.get("title", "")
            problem = parsed.get("form_problem", "") or parsed.get("body", "")
            query_text = f"{title}\n{problem}"[:1000]

            query_created = parsed.get("created_at")
            query_ticket = parsed.get("number") or parsed.get("ticket_number")

            similar: List[Dict[str, Any]] = []
            if self.retriever is not None and self.retriever.embeddings is not None:
                try:
                    similar = self.retriever.retrieve_similar(
                        query_text=query_text,
                        query_created_at=pd.to_datetime(query_created) if query_created else None,
                        query_ticket_number=int(query_ticket) if query_ticket else None,
                        top_k=5,
                    )
                except Exception as e:
                    logger.warning(f"Retrieval error: {e}")

            return {"similar_tickets": similar}

    def node_draft_reply(self, state: TriageState) -> Dict[str, Any]:
        """Node 5: Generate triage response."""
        with trace_node("generate_draft_reply"):
            parsed = state["parsed_issue"]
            ticket_num = int(parsed.get("number") or parsed.get("ticket_number") or 0)
            title = str(parsed.get("title", ""))
            problem = str(parsed.get("form_problem", "") or parsed.get("body", ""))

            draft = call_llm_draft(
                ticket_number=ticket_num,
                title=title,
                problem_text=problem,
                missing_info=state["missing_info"],
                similar_tickets=state["similar_tickets"],
                suggested_label=state["suggested_label"],
                form_fields=parsed,
            )
            return {"draft_reply": draft}

    def node_output(self, state: TriageState) -> Dict[str, Any]:
        """Node 6: Format final response dictionary."""
        with trace_node("output_formatting"):
            parsed = state["parsed_issue"]
            ticket_num = int(parsed.get("number") or parsed.get("ticket_number") or 0)

            output = {
                "ticket_number": ticket_num,
                "predicted_days_to_close": state["predicted_days_to_close"],
                "fixed_horizon_probabilities": state["fixed_horizon_probabilities"],
                "p_not_planned": state["p_not_planned"],
                "p_not_planned_confidence": state["p_not_planned_confidence"],
                "top_shap_features": state["top_shap_features"],
                "similar_tickets": state["similar_tickets"],
                "missing_info": state["missing_info"],
                "draft_reply": state["draft_reply"],
                "suggested_label": state["suggested_label"],
                "label_confidence": state["label_confidence"],
            }
            return {"output": output}

    def _build_graph(self):
        """Assemble the StateGraph pipeline."""
        workflow = StateGraph(TriageState)

        workflow.add_node("parse_ticket", self.node_parse_ticket)
        workflow.add_node("completeness_check", self.node_completeness_check)
        workflow.add_node("predict", self.node_predict)
        workflow.add_node("retrieve_similar", self.node_retrieve_similar)
        workflow.add_node("draft_reply", self.node_draft_reply)
        workflow.add_node("output", self.node_output)

        workflow.set_entry_point("parse_ticket")
        workflow.add_edge("parse_ticket", "completeness_check")
        workflow.add_edge("completeness_check", "predict")
        workflow.add_edge("predict", "retrieve_similar")
        workflow.add_edge("retrieve_similar", "draft_reply")
        workflow.add_edge("draft_reply", "output")
        workflow.add_edge("output", END)

        return workflow.compile()

    def run(self, raw_issue: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the graph pipeline for an issue with Langfuse tracing."""
        ticket_num = int(raw_issue.get("number") or raw_issue.get("ticket_number") or 0)
        title = str(raw_issue.get("title", ""))
        with trace_agent_run(ticket_number=ticket_num, title=title) as trace_info:
            initial_state: TriageState = {"raw_issue": raw_issue}
            final_state = self.app.invoke(initial_state)
            out = final_state["output"]
            if trace_info.get("trace_id"):
                out["trace_id"] = trace_info["trace_id"]
                out["trace_url"] = trace_info["trace_url"]
            return out
