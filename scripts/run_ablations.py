#!/usr/bin/env python
"""Run the ten frozen primary ablations through the shared execution harness
(Requirement 12.6/12.9; task 12.1).

`decisions.md` Section 17 freezes the ten primary ablations
(`src.evaluation.ablation_matrix.ABLATIONS`, task 12). Requirement 12.9
requires that "all primary ablations SHALL reuse identical splits, seeds,
and evaluation procedures."

Documented simplification -- what this script actually demonstrates today
----------------------------------------------------------------------------

There is currently no single unified "build model from
`AblationConfig.config_diff`" entry point in this codebase: actually
disabling/enabling an intervention channel, a protocol encoder, a witness,
etc. per ablation is deep model-architecture wiring (`models.intervention_aware`,
`witnesses.aggregation`, ...) that is out of scope for this task.

What this script establishes instead is the **shared execution harness**
that a future task can extend once that wiring exists: for each of the ten
named `AblationConfig`s, it runs the exact same frozen pipeline
(`cohorts.pipeline.run_observationalization_pipeline`, task 6.4) with the
exact same `master_seed` and the exact same raw cohort input -- so every
ablation gets identical splits and identical per-split seeds, per
Requirement 12.9's "identical splits, seeds" -- and then fits the exact
same placeholder "evaluation procedure" (`models.non_neural_baseline.CoxBaseline`,
fit on the development split, evaluated via
`evaluation.metrics.concordance_index` on that same development split) as
a stand-in for the real per-ablation model evaluation.

The per-ablation `config_diff` (the actual architectural variation each
ablation is supposed to express) is recorded in each result file for
provenance, but it does not yet change which model is fit or how it is
evaluated -- every ablation's `concordance_index` here is computed from the
identical `CoxBaseline` fit on the identical population, so the ten
resulting numbers are expected to be equal (or, if `run_observationalization_pipeline`
had any non-deterministic step keyed by ablation name, merely close). This
is intentional and documented, not a bug: this task's job is to prove the
harness runs all ten ablations end to end through the same pipeline stages
without error and collects one result file per ablation, not to implement
the model-swap logic itself.

Fitting/evaluation population (documented simplification)
-------------------------------------------------------------

`CoxBaseline` is fit on the pipeline's development split's full patient
population (`result.split_cohorts.dev.patients`) -- not the
observationalization-*retained* subset of dev -- and `concordance_index`
is computed on that same population. Using the full dev split (rather than
the retained-only subset) avoids a degenerate/too-small fit population on
a small synthetic cohort; using the *same* population for both fitting and
evaluation (rather than a held-out split) is a placeholder "evaluation
procedure" appropriate to this harness-proving task, not a claim of
out-of-sample predictive validity.

Experiment identifier (documented simplification)
------------------------------------------------------

`_compute_harness_experiment_id(master_seed)` below is a simple
deterministic identifier derived from a fixed label and `master_seed`
only -- it is **not** `infra.registry.compute_experiment_id`'s full
`sha256(canonical_json(resolved_config))` over a real `ResolvedConfig`.
Wiring the ablation harness into the full experiment-registry round trip
(`infra.registry.ExperimentRegistry`) is out of scope here; this
identifier exists only so every ablation run's results land under one
shared, reproducible `results/<experiment_id>/ablations/<name>/` directory
for a given `master_seed`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cohorts.cohort_construction import PatientRecord
from cohorts.pipeline import run_observationalization_pipeline
from data.loaders.actg175 import Actg175DataNotFoundError, load_actg175
from data.loaders.actg175_adapter import raw_tables_to_pipeline_inputs
from data.timelines import PatientTimeline as DataPatientTimeline
from evaluation.ablation_matrix import ABLATIONS
from evaluation.metrics import concordance_index
from models.non_neural_baseline import CoxBaseline
from protocols.endpoints import EndpointResult
from protocols.schema import TargetTrialSpec

#: Repository root, resolved relative to this file's location (matches
#: `scripts/run_stage.py`'s convention).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Default results root (`results/<experiment_id>/ablations/<name>/result.json`).
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"

#: Fixed label folded into the harness's simplified experiment identifier
#: (see module docstring, "Experiment identifier"). Distinguishes this
#: harness's IDs from any real `infra.registry.compute_experiment_id` value.
_HARNESS_LABEL = "ablation_harness_v1"

#: The real, packaged raw ACTG175 v3.2 data directory (`docs/data_access.md`
#: Section 2's approved storage location: `data/raw/`, gitignored). Used
#: only when `--real-data-path` is supplied to `main()`.
DEFAULT_REAL_DATA_PATH = REPO_ROOT / "data" / "raw" / "actg175_v3_2"


def _compute_harness_experiment_id(master_seed: int) -> str:
    """A simple, deterministic identifier for one harness run.

    `sha256(f"{_HARNESS_LABEL}:{master_seed}")[:16]` -- deterministic given
    `master_seed` alone, distinct from `infra.registry.compute_experiment_id`
    (see module docstring's "Experiment identifier" section for why a full
    `ResolvedConfig` round trip is not used here).
    """
    digest = hashlib.sha256(f"{_HARNESS_LABEL}:{master_seed}".encode()).hexdigest()
    return digest[:16]


def _elapsed_weeks_since_assignment(patient: PatientRecord) -> float:
    """Weeks from treatment assignment (time zero) to `endpoint_result.event_time`.

    `evaluation.metrics.concordance_index` requires numeric `event_times`;
    `PatientRecord.endpoint_result.event_time` is a `datetime`. This mirrors
    `causal.baselines._elapsed_weeks`'s elapsed-time computation locally
    (a small, self-contained helper) rather than importing that module's
    private function.
    """
    treatment_events = patient.timeline.treatment_events
    if not treatment_events:
        raise ValueError(
            f"patient_id={patient.patient_id!r} has no treatment_events; "
            "cannot compute elapsed time to the endpoint."
        )
    assignment_time = min(event.observed_at for event in treatment_events)
    elapsed_days = (patient.endpoint_result.event_time - assignment_time).days
    return elapsed_days / 7.0


def _fit_and_evaluate(patients: Sequence[PatientRecord]) -> float:
    """Fit `CoxBaseline` on `patients` and compute concordance on the same
    population (see module docstring's "Fitting/evaluation population").

    Returns:
        The concordance index, or `float("nan")` if there are no comparable
        pairs (e.g. too few patients or no observed events -- possible on a
        very small synthetic cohort; `concordance_index` itself returns
        `nan` in that case rather than raising).
    """
    baseline = CoxBaseline().fit(patients)
    fitted = baseline.fitted

    risk_scores = [fitted.risk_scores[patient.patient_id] for patient in patients]
    event_times = [_elapsed_weeks_since_assignment(patient) for patient in patients]
    event_indicators = [patient.endpoint_result.event_indicator for patient in patients]

    return concordance_index(risk_scores, event_times, event_indicators)


def run_all_ablations(
    raw_timelines: Sequence[DataPatientTimeline] | Mapping[str, DataPatientTimeline],
    treatment_arms: Mapping[str, int],
    spec: TargetTrialSpec,
    master_seed: int,
    results_root: Path,
    precomputed_endpoint_results: Mapping[str, EndpointResult] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all ten frozen ablations through the shared execution harness.

    For each `AblationConfig` in `evaluation.ablation_matrix.ABLATIONS`
    (in list order):

    1. runs `cohorts.pipeline.run_observationalization_pipeline` with the
       IDENTICAL `raw_timelines`, `treatment_arms`, `spec`, and
       `master_seed` for every ablation -- so every ablation gets identical
       splits and identical per-split seeds (Requirement 12.9);
    2. fits `models.non_neural_baseline.CoxBaseline` on the resulting dev
       split's full patient population and computes
       `evaluation.metrics.concordance_index` on that same population (see
       module docstring's "Fitting/evaluation population" -- a placeholder
       evaluation procedure, not the real per-ablation model);
    3. writes `<results_root>/<experiment_id>/ablations/<ablation.name>/result.json`
       containing `ablation_name`, `config_diff`, `concordance_index`, and
       `master_seed`;
    4. collects every ablation's result dict into the returned mapping,
       keyed by `ablation.name`.

    Args:
        raw_timelines: candidate patients' `data.timelines.PatientTimeline`s,
            forwarded unchanged to `run_observationalization_pipeline` for
            every ablation.
        treatment_arms: `{patient_id: arm}` raw treatment-arm assignments,
            forwarded unchanged for every ablation.
        spec: the target-trial specification, forwarded unchanged for every
            ablation.
        master_seed: the single top-level seed forwarded unchanged for
            every ablation, so every ablation's splits and per-split seeds
            are identical (Requirement 12.9).
        results_root: the root directory under which
            `<experiment_id>/ablations/<name>/result.json` is written for
            each ablation.
        precomputed_endpoint_results: forwarded unchanged to
            `run_observationalization_pipeline`'s
            `precomputed_endpoint_results` parameter for every ablation.
            `None` (the default) preserves the original synthetic-fixture
            recompute-from-timeline-buckets behavior; a real-data caller
            (`data.loaders.actg175_adapter.raw_tables_to_pipeline_inputs`'s
            third return value) passes its `{patient_id: EndpointResult}`
            mapping here, since the real ACTG175 data provides only a
            composite outcome with no per-component breakdown.

    Returns:
        `{ablation.name: result_dict}` with exactly one entry per
        `AblationConfig` in `ABLATIONS`.
    """
    experiment_id = _compute_harness_experiment_id(master_seed)
    ablations_root = Path(results_root) / experiment_id / "ablations"

    results: dict[str, dict[str, Any]] = {}
    for ablation in ABLATIONS:
        pipeline_result = run_observationalization_pipeline(
            raw_timelines,
            treatment_arms,
            spec,
            master_seed,
            precomputed_endpoint_results=precomputed_endpoint_results,
        )

        dev_patients = pipeline_result.split_cohorts.dev.patients
        concordance = _fit_and_evaluate(dev_patients)

        result: dict[str, Any] = {
            "ablation_name": ablation.name,
            "config_diff": ablation.config_diff,
            "concordance_index": concordance,
            "master_seed": master_seed,
        }

        ablation_dir = ablations_root / ablation.name
        ablation_dir.mkdir(parents=True, exist_ok=True)
        result_path = ablation_dir / "result.json"
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)

        results[ablation.name] = result

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ten frozen primary ablations (evaluation.ablation_matrix.ABLATIONS) "
            "through the shared execution harness (Requirement 12.6/12.9), against the "
            "real, packaged ACTG175 v3.2 raw data by default. The integration test "
            "(tests/integration/test_run_ablations.py) exercises run_all_ablations() "
            "directly against a synthetic cohort instead, for fast, data-independent "
            "testing."
        )
    )
    parser.add_argument(
        "--target-trial-config",
        type=Path,
        required=True,
        help="Path to a target-trial-spec YAML config (see configs/target_trials/main.yaml).",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        required=True,
        help="The single top-level seed every ablation's splits/seeds are derived from.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Root directory results are written under (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--real-data-path",
        type=Path,
        default=DEFAULT_REAL_DATA_PATH,
        help=(
            "Directory containing the real raw ACTG175 v3.2 table CSVs "
            f"(default: {DEFAULT_REAL_DATA_PATH}). If this path does not exist, "
            "the command refuses with a clear error rather than falling back to "
            "a synthetic cohort."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: load the real ACTG175 data, adapt it into pipeline
    inputs, and run all ten frozen ablations against it.

    Loads real raw tables via `data.loaders.actg175.load_actg175`
    (checksum-verified against the frozen dataset contract) and converts
    them via `data.loaders.actg175_adapter.raw_tables_to_pipeline_inputs`
    into `(timelines, treatment_arms, endpoint_results)` -- excluding arms
    2/3 before any event is built, and passing `endpoint_results` as
    `run_all_ablations`'s `precomputed_endpoint_results` argument, since
    the real data has no per-component composite-endpoint breakdown (see
    `actg175_adapter`'s module docstring).

    Returns 0 on success. On any exception -- including the raw data being
    absent at `--real-data-path` -- prints the exception message to stderr
    and returns 1 without raising. The substantive logic lives in
    `run_all_ablations`, which is exercised directly (against a synthetic
    cohort, for fast tests) by `tests/integration/test_run_ablations.py`.
    """
    import yaml

    from protocols.schema import TargetTrialSpec

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        with args.target_trial_config.open("r", encoding="utf-8") as handle:
            raw_spec = yaml.safe_load(handle)
        spec = TargetTrialSpec.model_validate(raw_spec)

        tables = load_actg175(args.real_data_path)
        timelines, treatment_arms, endpoint_results = raw_tables_to_pipeline_inputs(tables)
        print(
            f"loaded real ACTG175 data: {len(timelines)} arm-0/1 patients "
            f"(checksum {tables.checksum_sha256[:12]}...)"
        )

        results = run_all_ablations(
            timelines,
            treatment_arms,
            spec,
            args.master_seed,
            args.results_root,
            precomputed_endpoint_results=endpoint_results,
        )
    except Actg175DataNotFoundError as exc:
        print(
            f"run_ablations.py failed: real ACTG175 data not found: {exc}. "
            "See docs/data_access.md Section 2 for the approved storage location "
            "(data/raw/actg175_v3_2/) and Section 5 for the data-use agreement.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary, see docstring
        print(f"run_ablations.py failed: {exc}", file=sys.stderr)
        return 1

    experiment_id = _compute_harness_experiment_id(args.master_seed)
    print(f"ran {len(results)} ablations; results under "
          f"{args.results_root / experiment_id / 'ablations'}")
    print("run_ablations.py succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
