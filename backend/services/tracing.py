from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from opentelemetry import context as otel_context
    from opentelemetry import propagate, trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import SpanKind
except Exception:  # pragma: no cover
    otel_context = None  # type: ignore[assignment]
    propagate = None  # type: ignore[assignment]
    trace = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]

    class SpanKind:  # type: ignore[no-redef]
        INTERNAL = None
        SERVER = None
        CLIENT = None
        CONSUMER = None


_tracing_configured = False


def _truthy(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def tracing_enabled() -> bool:
    if _truthy("OTEL_SDK_DISABLED", "false"):
        return False
    return _truthy("TOKEMIZER_TRACING_ENABLED", "false")


def configure_tracing(service_name: str = "tokemizer-backend") -> None:
    global _tracing_configured
    if _tracing_configured or not tracing_enabled() or trace is None:
        return

    try:
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": service_name,
                    "service.namespace": "tokemizer",
                }
            )
        )

        exporter_name = os.environ.get("OTEL_TRACES_EXPORTER", "").strip().lower()
        processor = None

        if exporter_name in {"otlp", "otlp_http", "otlp_proto_http"}:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                processor = BatchSpanProcessor(OTLPSpanExporter())
            except Exception:
                logger.warning("OTLP exporter not available; tracing will remain local")
        elif exporter_name == "console":
            try:
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter

                processor = BatchSpanProcessor(ConsoleSpanExporter())
            except Exception:
                logger.warning("Console exporter unavailable; tracing will remain local")

        if processor is not None:
            provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        _tracing_configured = True
        logger.info("Tracing configured (exporter=%s)", exporter_name or "none")
    except Exception:
        logger.exception("Failed to configure tracing")


def _tracer():
    if trace is None:
        return None
    return trace.get_tracer("tokemizer.backend")


def extract_context_from_headers(headers: Mapping[str, str]):
    if propagate is None:
        return None
    carrier = {str(k): str(v) for k, v in headers.items()}
    return propagate.extract(carrier)


def extract_context_from_carrier(carrier: Mapping[str, str]):
    if propagate is None:
        return None
    return propagate.extract(dict(carrier))


def inject_context_to_headers(headers: Dict[str, str]) -> Dict[str, str]:
    if propagate is None:
        return headers
    propagate.inject(headers)
    return headers


def inject_context_to_carrier() -> Dict[str, str]:
    carrier: Dict[str, str] = {}
    if propagate is not None:
        propagate.inject(carrier)
    return carrier


def attach_context(ctx):
    if otel_context is None or ctx is None:
        return None
    return otel_context.attach(ctx)


def detach_context(token) -> None:
    if otel_context is None or token is None:
        return
    otel_context.detach(token)


@contextmanager
def start_span(name: str, *, kind: Any = None, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    tracer = _tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name, kind=kind, attributes=attributes or {}) as span:
        yield span


def current_trace_ids() -> Dict[str, str]:
    if trace is None:
        return {}
    span = trace.get_current_span()
    if span is None:
        return {}
    ctx = span.get_span_context()
    if ctx is None or not getattr(ctx, "is_valid", False):
        return {}
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
    }
