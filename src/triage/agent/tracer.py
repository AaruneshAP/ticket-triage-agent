"""Langfuse Observability & Tracing for LangGraph Agent and LLM Pipeline.

Features:
- Traces complete LangGraph triage runs with root observation.
- Traces individual LangGraph nodes (parse_ticket, completeness_check, predict_models, retrieve_similar, generate_draft_reply).
- Traces LLM calls with latency, token usage, provider, model name, and cache hit/miss status.
- Resilient: Gracefully no-ops if Langfuse credentials are not configured or network fails.
"""

import contextlib
import logging
import os
from typing import Any, Dict, Generator, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_langfuse_client = None
_client_initialized = False


def get_langfuse() -> Optional[Any]:
    """Get or initialize singleton Langfuse client."""
    global _langfuse_client, _client_initialized
    if _client_initialized:
        return _langfuse_client

    _client_initialized = True
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    host = os.getenv("LANGFUSE_HOST") or os.getenv(
        "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
    )

    if not public_key or not secret_key:
        logger.debug("Langfuse credentials not found. Tracing disabled.")
        return None

    # Ensure LANGFUSE_HOST is set in environment for OpenTelemetry exporter
    os.environ["LANGFUSE_HOST"] = host

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        logger.info(f"Langfuse client initialized successfully against {host}.")
        return _langfuse_client
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse client: {e}")
        return None


class _NoOpObservation:
    """Fallback no-op observation object when Langfuse is disabled or fails."""

    def update(self, *args, **kwargs) -> None:
        pass


@contextlib.contextmanager
def trace_agent_run(ticket_number: int, title: str) -> Generator[Dict[str, Any], None, None]:
    """Root observation context for a full LangGraph triage execution."""
    lf = get_langfuse()
    trace_info: Dict[str, Any] = {
        "trace_id": None,
        "trace_url": None,
    }

    if lf is None:
        yield trace_info
        return

    try:
        root_cm = lf.start_as_current_observation(name="triage_agent_run")
        root_span = root_cm.__enter__()
        root_span.update(
            input={"ticket_number": ticket_number, "title": title},
            metadata={"ticket_number": ticket_number},
        )
        trace_info["trace_id"] = lf.get_current_trace_id()
        trace_info["trace_url"] = lf.get_trace_url()
    except Exception as e:
        logger.debug(f"Langfuse start error in trace_agent_run: {e}")
        yield trace_info
        return

    try:
        yield trace_info
        try:
            root_span.update(output={"status": "completed"})
        except Exception:
            pass
    finally:
        try:
            root_cm.__exit__(None, None, None)
            lf.flush()
        except Exception as e:
            logger.debug(f"Langfuse exit error in trace_agent_run: {e}")


@contextlib.contextmanager
def trace_node(
    node_name: str, input_summary: Optional[Dict[str, Any]] = None
) -> Generator[Any, None, None]:
    """Span context for an individual LangGraph workflow node."""
    lf = get_langfuse()
    if lf is None:
        yield _NoOpObservation()
        return

    try:
        node_cm = lf.start_as_current_observation(name=node_name)
        node_span = node_cm.__enter__()
        if input_summary:
            node_span.update(input=input_summary)
    except Exception as e:
        logger.debug(f"Langfuse start error in node span {node_name}: {e}")
        yield _NoOpObservation()
        return

    try:
        yield node_span
    finally:
        try:
            node_cm.__exit__(None, None, None)
        except Exception as e:
            logger.debug(f"Langfuse exit error in node span {node_name}: {e}")


def record_llm_generation(
    name: str,
    prompt: str,
    system_prompt: str,
    output_text: str,
    provider: str,
    model: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    cache_hit: bool,
    error: Optional[str] = None,
) -> None:
    """Record an LLM call generation span in Langfuse."""
    lf = get_langfuse()
    if lf is None:
        return

    try:
        with lf.start_as_current_observation(name=name, as_type="generation") as gen_span:
            gen_span.update(
                input={"system": system_prompt[:300], "prompt": prompt[:600]},
                output=output_text[:1000] if output_text else (f"ERROR: {error}" if error else ""),
                model=model,
                level="ERROR" if error else "DEFAULT",
                status_message=error,
            )
            lf.update_current_generation(
                name=name,
                model=model,
                usage_details={
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                },
                metadata={
                    "provider": provider,
                    "cache_hit": cache_hit,
                    "latency_ms": round(latency_ms, 2),
                },
            )
    except Exception as e:
        logger.debug(f"Langfuse error recording generation {name}: {e}")
