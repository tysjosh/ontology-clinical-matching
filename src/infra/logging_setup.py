"""Structured JSON logging.

`configure_logging` is the single entry point for wiring up logging for a
pipeline run (`design.md` Section 3.1 — "structured JSON logging; every log
record includes `experiment_id`, `stage`, and `seed`."). Every stage of the
pipeline (`design.md` Section 2.1) calls it at the start of the stage with the
current `experiment_id`, `stage` name, and `seed`; every subsequent log record
emitted through the returned logger carries those three fields automatically,
so log lines can be joined back to a specific run and stage without the
caller having to remember to pass them on every call site.

Log records are emitted as single-line JSON objects (one JSON document per
line, "JSON Lines"), which keeps them both human-readable and trivially
parseable by downstream tooling (log aggregation, the manifest writer, CI
failure triage) without a custom parser.

This module intentionally does not know about `ResolvedConfig` or the
experiment registry (`src/infra/registry.py`) — it only accepts the three
plain values it needs (`experiment_id`, `stage`, `seed`), so it has no import
dependency on the rest of `src/infra` and can be configured before the
registry or manifest writer exist for a run.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import IO, Any

#: Default name of the logger configured/returned by `configure_logging`.
DEFAULT_LOGGER_NAME = "causal_witness_pwm"

#: Sentinel marker set on handlers created by `configure_logging`, so a
#: repeat call can find and remove its own previous handler(s) instead of
#: either duplicating output or accidentally clearing handlers installed by
#: something else.
_HANDLER_MARKER = "_causal_witness_pwm_structured_handler"


class _ContextFilter(logging.Filter):
    """Attaches the run-level context fields to every record it sees."""

    def __init__(self, experiment_id: str, stage: str, seed: int) -> None:
        super().__init__()
        self._experiment_id = experiment_id
        self._stage = stage
        self._seed = seed

    def filter(self, record: logging.LogRecord) -> bool:
        record.experiment_id = self._experiment_id
        record.stage = self._stage
        record.seed = self._seed
        return True


class JsonFormatter(logging.Formatter):
    """Renders a `LogRecord` as a single-line JSON object.

    Every record includes `timestamp`, `level`, `logger`, `message`,
    `experiment_id`, `stage`, and `seed`. The three context fields default to
    `None` if a record somehow reaches the formatter without having passed
    through a `_ContextFilter` (e.g. a third-party library logging through a
    handler attached to this formatter), rather than raising, so a missing
    context can never itself crash logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "experiment_id": getattr(record, "experiment_id", None),
            "stage": getattr(record, "stage", None),
            "seed": getattr(record, "seed", None),
        }
        if record.exc_info is not None:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging(
    experiment_id: str,
    stage: str,
    seed: int,
    *,
    logger_name: str = DEFAULT_LOGGER_NAME,
    level: int = logging.INFO,
    stream: IO[str] | None = None,
) -> logging.Logger:
    """Configure and return a logger that emits structured JSON log records.

    Every record subsequently logged through the returned logger (or any of
    its children, since the filter and handler are attached to the named
    logger itself) is rendered as a JSON object containing `experiment_id`,
    `stage`, and `seed` alongside the standard `timestamp`/`level`/`logger`/
    `message` fields.

    Idempotent and safe to call repeatedly with a changed `stage` (e.g. once
    per pipeline stage, per `design.md` Section 2.1): a previous call's
    handler/filter pair is removed before the new one is attached, so
    records are never duplicated and a later stage's log lines never carry
    an earlier stage's name.

    Args:
        experiment_id: the resolved-config hash identifying this run
            (`design.md` Section 2.2).
        stage: the name of the current pipeline stage (e.g. `"cohort_construction"`,
            `"smoke"`).
        seed: the seed in effect for this stage (the master seed, or a
            derived sub-seed, depending on the caller).
        logger_name: the name of the logger to configure. Defaults to the
            package-wide logger name so callers across modules share one
            configuration by default.
        level: the minimum severity to emit.
        stream: the stream to write JSON lines to. Defaults to `sys.stdout`.

    Returns:
        The configured `logging.Logger`.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Remove any handler/filter pair this function previously attached, so
    # repeated calls (e.g. once per stage) don't accumulate duplicate output.
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(_ContextFilter(experiment_id=experiment_id, stage=stage, seed=seed))
    setattr(handler, _HANDLER_MARKER, True)
    logger.addHandler(handler)

    return logger
