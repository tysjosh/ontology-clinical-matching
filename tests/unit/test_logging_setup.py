"""Unit tests for `src/infra/logging_setup.py`.

**Validates: Requirements 1.1**
"""

from __future__ import annotations

import json
import logging
from io import StringIO

from infra.logging_setup import configure_logging


def _read_records(stream: StringIO) -> list[dict]:
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def test_emitted_record_is_valid_json_with_required_fields():
    stream = StringIO()
    logger = configure_logging(
        experiment_id="abc123",
        stage="cohort_construction",
        seed=42,
        logger_name="test_logger_basic",
        stream=stream,
    )

    logger.info("hello world")

    records = _read_records(stream)
    assert len(records) == 1
    record = records[0]

    assert record["message"] == "hello world"
    assert record["experiment_id"] == "abc123"
    assert record["stage"] == "cohort_construction"
    assert record["seed"] == 42
    assert record["level"] == "INFO"
    assert "timestamp" in record


def test_every_record_includes_context_fields_across_multiple_log_calls():
    stream = StringIO()
    logger = configure_logging(
        experiment_id="exp-1",
        stage="stage-1",
        seed=7,
        logger_name="test_logger_multiple",
        stream=stream,
    )

    logger.info("first")
    logger.warning("second")
    logger.error("third")

    records = _read_records(stream)
    assert len(records) == 3
    for record in records:
        assert record["experiment_id"] == "exp-1"
        assert record["stage"] == "stage-1"
        assert record["seed"] == 7


def test_repeated_configure_logging_updates_stage_without_duplicating_handlers():
    stream = StringIO()
    logger_name = "test_logger_reconfigure"

    configure_logging(
        experiment_id="exp-1", stage="stage-1", seed=1, logger_name=logger_name, stream=stream
    )
    configure_logging(
        experiment_id="exp-1", stage="stage-2", seed=1, logger_name=logger_name, stream=stream
    )

    logging.getLogger(logger_name).info("only once")

    records = _read_records(stream)
    assert len(records) == 1
    assert records[0]["stage"] == "stage-2"


def test_each_output_line_is_independently_parseable_json():
    stream = StringIO()
    logger = configure_logging(
        experiment_id="exp-json",
        stage="stage-json",
        seed=99,
        logger_name="test_logger_json_lines",
        stream=stream,
    )

    logger.info("line one")
    logger.info("line two")

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
