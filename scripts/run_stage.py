#!/usr/bin/env python
"""CLI entry point for running individual pipeline stages.

`design.md` Section 2.1 lists the full stage sequence (Stage 0 through
Stage 19). This script is the one-command entry point Requirement 1.4
describes ("a smoke experiment SHALL run end-to-end from one documented
command") and is structured so later tasks can add more `--stage` choices
without reshaping the CLI: each stage is implemented as its own
`_run_<stage>(args) -> int` function, dispatched from `main()` by name.

Currently implemented stages:

- `smoke`: exercises config loading, seeding, structured logging,
  experiment registration, and manifest writing end-to-end on a tiny
  synthetic fixture (the existing `tests/fixtures/config/valid_*.yaml`
  fixtures already used by the config-loading unit tests) — never real
  ACTG175 data (Requirement 1.2-1.4).
- `full_pipeline`: exercises cohort construction through the predictive
  baseline on a tiny synthetic cohort (Requirement 1.2, 1.3; see below).
- `real_data_pipeline`: the same cohort-construction-through-predictive-
  baseline sequence as `full_pipeline`, but against the real, packaged
  ACTG175 v3.2 raw tables at `data/raw/actg175_v3_2/` (via
  `data.loaders.actg175.load_actg175` and
  `data.loaders.actg175_adapter.raw_tables_to_pipeline_inputs`), using the
  real `unadjusted_cox_reference_estimator` for the sealed internal
  randomized reference rather than the placeholder. Requires the real raw
  data to be present at that path (see `docs/data_access.md`); refuses with
  a clear error if it is absent, rather than silently falling back to a
  synthetic cohort.

The frozen full pipeline order, and where each stage actually lives
------------------------------------------------------------------------

Task 18 (Requirement 1.2, 1.3, 15.5; Correctness Properties 1-8) asks for
the full frozen pipeline to be wired "end-to-end". The frozen order itself
(`decisions.md` Section 9 and `design.md` Section 2.1) is deliberately
**not** force-unified into a single script here — each later stage
(ablations, robustness, go/no-go, locked-evaluation, reporting) already
has its own dedicated, tested entry point, and re-implementing or
re-wrapping those would duplicate working code rather than integrate it.
Instead, the frozen order is documented here, stage by stage, so a reader
can trace the whole pipeline from this docstring alone even though the
stages live in different files:

1. Construct cohort, seal internal randomized reference, split
   dev/val/test, fit preprocessing, observationalize each split, run the
   leakage audit, and fit the non-neural predictive baseline: this
   script's `--stage full_pipeline`, which calls
   `cohorts.pipeline.run_observationalization_pipeline()` (itself the
   single place that chains cohort construction through
   observationalization, `decisions.md` Section 9 steps 1-8),
   `data.leakage_audit.LeakageAuditor().audit()`, and
   `models.non_neural_baseline.CoxBaseline().fit()`.
2. Ablation matrix (all ten frozen ablations): `scripts/run_ablations.py`.
3. Robustness diagnostics: `src/evaluation/robustness.py`'s robustness
   functions (positivity/overlap checks, sensitivity analyses).
4. Go/no-go checklist: `scripts/check_go_no_go.py`.
5. Locked external evaluation (unsealing the internal randomized
   reference) — only reachable with a valid, minted
   `cohorts.sealed_reference.LockedEvaluationToken`:
   `scripts/run_paper.py --stage locked-evaluation` (always refused
   without a real token; see that script's module docstring).
6. Reporting (tables, figures, model card) regenerated from saved
   predictions/aggregates: `scripts/run_paper.py --stage reproduction`
   (the default stage).

So the full pipeline is: this script's `full_pipeline` stage, then
`run_ablations.py`, then `evaluation.robustness`, then
`check_go_no_go.py`, then (only with a token) the locked-evaluation stage,
then `run_paper.py` for reporting.

Usage:

    python scripts/run_stage.py --stage smoke
    python scripts/run_stage.py --stage smoke --results-root /tmp/results
    python scripts/run_stage.py --stage full_pipeline
    python scripts/run_stage.py --stage real_data_pipeline
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

from cohorts.pipeline import run_observationalization_pipeline
from cohorts.sealed_reference import unadjusted_cox_reference_estimator
from data.event_model import ClinicalEvent, MissingnessState
from data.leakage_audit import LeakageAuditor
from data.loaders.actg175 import Actg175DataNotFoundError, load_actg175
from data.loaders.actg175_adapter import raw_tables_to_pipeline_inputs
from data.timelines import build_patient_timeline
from infra.config import ConfigError, load_config
from infra.logging_setup import configure_logging
from infra.manifest import ManifestPrivacyError, ManifestWriter
from infra.registry import DuplicateExperimentError, ExperimentRegistry, compute_experiment_id
from infra.seeding import derive_seed
from models.non_neural_baseline import CoxBaseline
from protocols.schema import TargetTrialSpec

#: Repository root, resolved relative to this file's location so the script
#: behaves the same regardless of the caller's current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Tiny synthetic fixtures used by the smoke stage. These are the same
#: fixtures already exercised by `tests/unit/test_config.py` — deliberately
#: reused rather than duplicated, so the smoke stage and the config unit
#: tests never drift against two different "synthetic ACTG175-shaped"
#: fixtures. Nothing here is real ACTG175 data: the dataset checksum is a
#: placeholder value, not a real frozen checksum (`decisions.md` Section 2).
_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "config"
DEFAULT_SMOKE_DATASET_CONFIG = _FIXTURES_DIR / "valid_dataset.yaml"
DEFAULT_SMOKE_TARGET_TRIAL_CONFIG = _FIXTURES_DIR / "valid_target_trial.yaml"
DEFAULT_SMOKE_EXPERIMENT_CONFIG = _FIXTURES_DIR / "valid_experiment.yaml"

#: Default results root (`design.md` Section 2.3: `results/<experiment_id>/`),
#: overridable via `--results-root` so tests can point it at a temp directory
#: instead of polluting the repository's `results/` tree.
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


def _run_smoke(args: argparse.Namespace) -> int:
    """Run the smoke stage end-to-end on a tiny synthetic fixture.

    Steps (Requirement 1.2-1.4):

    1. Load config via `load_config()`.
    2. Derive a stage-specific seed via `derive_seed()`.
    3. Configure structured logging via `configure_logging()`.
    4. Register the experiment via `ExperimentRegistry.register()`.
    5. Build and write a manifest via `ManifestWriter` under
       `<results_root>/<experiment_id>/manifest.json`.

    Returns 0 on success. On failure, prints a clear message to stderr and
    returns a non-zero exit code without raising — callers (including the
    e2e smoke test) can rely on the process exit code alone.
    """
    try:
        resolved = load_config(
            args.dataset_config, args.target_trial_config, args.experiment_config
        )
    except ConfigError as exc:
        print(f"smoke stage failed: could not load config: {exc}", file=sys.stderr)
        return 1

    experiment_id = compute_experiment_id(resolved)
    seed = derive_seed(resolved.experiment.seed, "smoke")
    logger = configure_logging(experiment_id=experiment_id, stage="smoke", seed=seed)
    logger.info("smoke stage starting")

    results_root: Path = args.results_root
    registry = ExperimentRegistry(
        registry_path=results_root / "registry.jsonl",
        results_root=results_root,
    )
    try:
        registry.register(resolved, git_commit=None, dirty=False, results_root=results_root)
    except DuplicateExperimentError as exc:
        print(f"smoke stage failed: {exc}", file=sys.stderr)
        return 1

    writer = ManifestWriter(
        experiment_id=experiment_id,
        stage="smoke",
        allowed_path_prefixes=(str(results_root),),
    )
    writer.add_input_hash("dataset_contract", resolved.dataset.checksum_sha256)
    writer.set_cohort_definition(
        {"population": resolved.dataset.population, "synthetic_fixture": True}
    )
    writer.set_feature_list(list(resolved.target_trial.adjustment_set))
    writer.set_treatment_definition(
        {
            "intervention": resolved.target_trial.intervention.get("concept_id"),
            "comparator": resolved.target_trial.comparator.get("concept_id"),
        }
    )
    writer.set_outcome_definition({"type": resolved.target_trial.outcome.get("type")})
    writer.set_model_config({"stage": "smoke"})
    writer.set_metrics({"smoke_check": "ok"})

    manifest_path = results_root / experiment_id / "manifest.json"
    try:
        writer.write(manifest_path)
    except ManifestPrivacyError as exc:
        print(
            f"smoke stage failed: manifest privacy scan blocked the write: {exc}",
            file=sys.stderr,
        )
        return 1

    logger.info("smoke stage completed")
    print(f"smoke stage succeeded: experiment_id={experiment_id} manifest={manifest_path}")
    return 0


#: Target-trial spec used by the `full_pipeline` stage (the same frozen
#: spec every other pipeline module uses -- never a synthetic/placeholder
#: spec, only the events feeding it are synthetic).
DEFAULT_FULL_PIPELINE_TRIAL_CONFIG = REPO_ROOT / "configs" / "target_trials" / "main.yaml"

#: The real, packaged raw ACTG175 v3.2 data directory (`docs/data_access.md`
#: Section 2's approved storage location: `data/raw/`, gitignored). Used
#: only by the `real_data_pipeline` stage.
DEFAULT_REAL_DATA_PATH = REPO_ROOT / "data" / "raw" / "actg175_v3_2"

#: Timestamps for the tiny synthetic cohort built by `_build_tiny_synthetic_inputs`.
_FULL_PIPELINE_ASSIGNMENT_TIME = datetime(2020, 1, 15, 9, 0, 0)
_FULL_PIPELINE_WEEK_20 = datetime(2020, 6, 4, 9, 0, 0)
_FULL_PIPELINE_END_OF_FOLLOWUP = datetime(2021, 1, 15, 9, 0, 0)


def _full_pipeline_event(patient_id: str, **overrides) -> ClinicalEvent:
    defaults = dict(
        patient_id=patient_id,
        encounter_id=None,
        concept_id="cd4_count",
        value=350.0,
        unit="cells/mm3",
        modality="lab",
        observed_at=_FULL_PIPELINE_WEEK_20,
        valid_from=None,
        valid_to=None,
        source="actg175",
        missingness=MissingnessState.OBSERVED,
        provenance="loader:actg175",
    )
    defaults.update(overrides)
    return ClinicalEvent(**defaults)


def _build_tiny_synthetic_inputs(n_per_arm: int = 10):
    """A tiny (`n_per_arm` patients per arm) synthetic cohort for the
    `full_pipeline` stage, adapted from the same hand-built-event pattern
    already used by `tests/integration/test_run_ablations.py`'s
    `_build_small_synthetic_inputs` (kept deliberately small here so the
    CLI stage stays fast). Never real ACTG175 data.
    """
    timelines = []
    treatment_arms: dict[str, int] = {}
    counter = 0
    for treatment_arm in (0, 1):
        for i in range(n_per_arm):
            counter += 1
            patient_id = f"P{counter:04d}"
            baseline_cd4 = 200.0 + (i * 6) % 300
            events = [
                _full_pipeline_event(
                    patient_id,
                    concept_id="hiv_positive",
                    value=True,
                    unit=None,
                    modality="baseline",
                    observed_at=_FULL_PIPELINE_ASSIGNMENT_TIME,
                ),
                _full_pipeline_event(
                    patient_id,
                    concept_id="baseline_cd4",
                    value=baseline_cd4,
                    unit="cells/mm3",
                    modality="baseline",
                    observed_at=_FULL_PIPELINE_ASSIGNMENT_TIME,
                ),
                _full_pipeline_event(
                    patient_id,
                    concept_id="age",
                    value=18.0 + (i % 50),
                    unit="years",
                    modality="baseline",
                    observed_at=_FULL_PIPELINE_ASSIGNMENT_TIME,
                ),
                _full_pipeline_event(
                    patient_id,
                    concept_id="karnofsky_score",
                    value=70.0 + (i % 30),
                    unit="score",
                    modality="baseline",
                    observed_at=_FULL_PIPELINE_ASSIGNMENT_TIME,
                ),
                _full_pipeline_event(
                    patient_id,
                    concept_id="symptomatic_status",
                    value=(i % 3 == 0),
                    unit=None,
                    modality="baseline",
                    observed_at=_FULL_PIPELINE_ASSIGNMENT_TIME,
                ),
                _full_pipeline_event(
                    patient_id,
                    concept_id="prior_opportunistic_infection",
                    value=(i % 5 == 0),
                    unit=None,
                    modality="baseline",
                    observed_at=_FULL_PIPELINE_ASSIGNMENT_TIME,
                ),
                _full_pipeline_event(
                    patient_id,
                    concept_id="prior_art_exposure_days",
                    value=float((i % 4) * 30),
                    unit="days",
                    modality="baseline",
                    observed_at=_FULL_PIPELINE_ASSIGNMENT_TIME,
                ),
                _full_pipeline_event(
                    patient_id,
                    concept_id=(
                        "zidovudine" if treatment_arm == 0 else "zidovudine+didanosine"
                    ),
                    value=1.0,
                    unit=None,
                    modality="treatment",
                    observed_at=_FULL_PIPELINE_ASSIGNMENT_TIME,
                ),
                _full_pipeline_event(
                    patient_id,
                    concept_id="administrative_censoring",
                    value=1.0,
                    unit=None,
                    modality="censoring",
                    observed_at=_FULL_PIPELINE_END_OF_FOLLOWUP,
                ),
                _full_pipeline_event(
                    patient_id,
                    concept_id="cd4_count",
                    value=baseline_cd4 - 10.0,
                    unit="cells/mm3",
                    modality="lab",
                    observed_at=_FULL_PIPELINE_WEEK_20,
                ),
            ]
            timelines.append(build_patient_timeline(events))
            treatment_arms[patient_id] = treatment_arm
    return timelines, treatment_arms


def _run_full_pipeline(args: argparse.Namespace) -> int:
    """Run cohort construction through the predictive baseline on a tiny
    synthetic cohort (Requirement 1.2, 1.3; Correctness Properties 1-8).

    Stages exercised, in order (see module docstring for the full frozen
    order and where its later stages live):

    1. Build a tiny synthetic cohort (never real ACTG175 data).
    2. `cohorts.pipeline.run_observationalization_pipeline()` -- construct
       the cohort, seal the internal randomized reference, split
       dev/val/test, fit preprocessing, and observationalize each split.
    3. `data.leakage_audit.LeakageAuditor().audit()` on the observationalized
       development split's retained patients.
    4. `models.non_neural_baseline.CoxBaseline().fit()` on the same
       development split, as the predictive-baseline stage.

    Returns 0 on success. On any exception, prints the exception message to
    stderr and returns 1 without raising.
    """
    try:
        with args.trial_config.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        spec = TargetTrialSpec.model_validate(raw)

        timelines, treatment_arms = _build_tiny_synthetic_inputs()
        master_seed = derive_seed(args.master_seed, "full_pipeline")

        result = run_observationalization_pipeline(
            timelines, treatment_arms, spec, master_seed
        )
        dev_patients = [
            patient
            for patient in result.split_cohorts.dev
            if patient.patient_id in set(result.dev.retained_patient_ids)
        ]
        print(f"cohort constructed: {len(dev_patients)} patients (dev, retained)")

        dev_timelines = [timeline for timeline in timelines if timeline.patient_id in
                          {p.patient_id for p in dev_patients}]
        violations = LeakageAuditor().audit(dev_timelines, list(spec.adjustment_set))
        print(f"leakage audit: {len(violations)} violations")

        CoxBaseline().fit(dev_patients)
        print("cox baseline fit: converged")
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary, see docstring
        print(f"full_pipeline stage failed: {exc}", file=sys.stderr)
        return 1

    print("full_pipeline stage succeeded")
    return 0


def _run_real_data_pipeline(args: argparse.Namespace) -> int:
    """Run cohort construction through the predictive baseline on the real,
    packaged ACTG175 v3.2 raw data (Requirement 1.2, 1.3; Correctness
    Properties 1-8).

    Identical stage sequence to `_run_full_pipeline`, but:

    1. loads the real raw tables via `data.loaders.actg175.load_actg175`
       (checksum-verified against the frozen dataset contract) instead of
       building a synthetic cohort;
    2. converts them via
       `data.loaders.actg175_adapter.raw_tables_to_pipeline_inputs` into
       `(timelines, treatment_arms, endpoint_results)`, excluding arms 2/3
       before any event is built (see that module's docstring);
    3. passes `endpoint_results` as `run_observationalization_pipeline`'s
       `precomputed_endpoint_results` argument, since the real data has no
       per-component composite-endpoint breakdown (see
       `actg175_adapter`'s docstring) -- never via
       `protocols.endpoints.CompositeEndpoint.compute`;
    4. uses `cohorts.sealed_reference.unadjusted_cox_reference_estimator`
       (a real Cox-model estimate) for the sealed internal randomized
       reference, instead of `placeholder_estimator`.

    Returns 0 on success. On any exception -- including the raw data being
    absent at `--real-data-path` -- prints the exception message to stderr
    and returns 1 without raising.
    """
    try:
        with args.trial_config.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        spec = TargetTrialSpec.model_validate(raw)

        tables = load_actg175(args.real_data_path)
        timelines, treatment_arms, endpoint_results = raw_tables_to_pipeline_inputs(tables)
        print(
            f"loaded real ACTG175 data: {len(timelines)} arm-0/1 patients "
            f"(checksum {tables.checksum_sha256[:12]}...)"
        )

        master_seed = derive_seed(args.master_seed, "real_data_pipeline")

        result = run_observationalization_pipeline(
            timelines,
            treatment_arms,
            spec,
            master_seed,
            reference_estimator=unadjusted_cox_reference_estimator,
            precomputed_endpoint_results=endpoint_results,
        )
        dev_patients = [
            patient
            for patient in result.split_cohorts.dev
            if patient.patient_id in set(result.dev.retained_patient_ids)
        ]
        print(f"cohort constructed: {len(dev_patients)} patients (dev, retained)")

        dev_timelines = [
            timeline
            for timeline in timelines
            if timeline.patient_id in {p.patient_id for p in dev_patients}
        ]
        violations = LeakageAuditor().audit(dev_timelines, list(spec.adjustment_set))
        print(f"leakage audit: {len(violations)} violations")

        CoxBaseline().fit(dev_patients)
        print("cox baseline fit: converged")

        total_cohort_size = (
            len(result.split_cohorts.dev)
            + len(result.split_cohorts.val)
            + len(result.split_cohorts.test)
        )
        print(
            f"internal randomized reference sealed: computed over {total_cohort_size} "
            "patients (value withheld until locked external evaluation)"
        )
    except Actg175DataNotFoundError as exc:
        print(
            f"real_data_pipeline stage failed: real ACTG175 data not found: {exc}. "
            "See docs/data_access.md Section 2 for the approved storage location "
            "(data/raw/actg175_v3_2/) and Section 5 for the data-use agreement.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary, see docstring
        print(f"real_data_pipeline stage failed: {exc}", file=sys.stderr)
        return 1

    print("real_data_pipeline stage succeeded")
    return 0


#: Dispatch table from `--stage` value to its runner function. Extend this
#: (and `build_parser()`'s `choices=`) as later tasks implement more stages.
_STAGE_RUNNERS = {
    "smoke": _run_smoke,
    "full_pipeline": _run_full_pipeline,
    "real_data_pipeline": _run_real_data_pipeline,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one pipeline stage of the Causal Witness Patient World Model "
            "experiment (design.md Section 2.1)."
        )
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=sorted(_STAGE_RUNNERS),
        help="Which pipeline stage to run.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Root directory under which results/<experiment_id>/ is written "
            f"(default: {DEFAULT_RESULTS_ROOT})."
        ),
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_SMOKE_DATASET_CONFIG,
        help="Path to the dataset-contract YAML config (smoke stage only).",
    )
    parser.add_argument(
        "--target-trial-config",
        type=Path,
        default=DEFAULT_SMOKE_TARGET_TRIAL_CONFIG,
        help="Path to the target-trial-spec YAML config (smoke stage only).",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_SMOKE_EXPERIMENT_CONFIG,
        help="Path to the experiment YAML config (smoke stage only).",
    )
    parser.add_argument(
        "--trial-config",
        type=Path,
        default=DEFAULT_FULL_PIPELINE_TRIAL_CONFIG,
        help="Path to the target-trial-spec YAML config (full_pipeline stage only).",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=20240101,
        help=(
            "Top-level seed for the full_pipeline/real_data_pipeline stage's run "
            "(full_pipeline and real_data_pipeline stages only)."
        ),
    )
    parser.add_argument(
        "--real-data-path",
        type=Path,
        default=DEFAULT_REAL_DATA_PATH,
        help=(
            "Directory containing the real raw ACTG175 v3.2 table CSVs "
            f"(real_data_pipeline stage only; default: {DEFAULT_REAL_DATA_PATH})."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    runner = _STAGE_RUNNERS[args.stage]
    return runner(args)


if __name__ == "__main__":
    sys.exit(main())
