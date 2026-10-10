"""Item 1: canonical dev-retained cohort, folds, and the shipped 14-column design.

Mirrors scripts/run_causal_analysis.py (master seed 20240101, derive_seed paths).
Reads no Test_Split data and calls no unseal/require_test_split/read_sealed_artifact.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import yaml

from causal.propensity import (
    COVARIATE_FIELDS,
    _assign_folds,
    _feature_vector,
    _fit_standardization,
)
from cohorts.pipeline import run_observationalization_pipeline
from cohorts.sealed_reference import unadjusted_cox_reference_estimator
from data.loaders.actg175 import load_actg175
from data.loaders.actg175_adapter import raw_tables_to_pipeline_inputs
from infra.seeding import derive_seed
from protocols.schema import TargetTrialSpec

ROOT = Path(__file__).resolve().parents[4]
MASTER_SEED = 20240101
N_FOLDS = 5


def load_spec() -> TargetTrialSpec:
    with (ROOT / "configs/target_trials/main.yaml").open() as handle:
        return TargetTrialSpec.model_validate(yaml.safe_load(handle))


def canonical_dev_patients():
    spec = load_spec()
    tables = load_actg175(ROOT / "data/raw/actg175_v3_2")
    timelines, arms, endpoints = raw_tables_to_pipeline_inputs(tables)
    result = run_observationalization_pipeline(
        timelines,
        arms,
        spec,
        derive_seed(MASTER_SEED, "causal_analysis_pipeline"),
        reference_estimator=unadjusted_cox_reference_estimator,
        precomputed_endpoint_results=endpoints,
    )
    keep = set(result.dev.retained_patient_ids)
    patients = [p for p in result.split_cohorts.dev if p.patient_id in keep]
    return patients, spec


def folds(patients, n_folds: int = N_FOLDS):
    """[(fold, train_patients, holdout_patients)], exactly as causal.propensity._cross_fit."""
    by_id = {p.patient_id: p for p in patients}
    fold_of = _assign_folds(list(by_id), n_folds)
    out = []
    for fold in range(n_folds):
        hold = [by_id[i] for i in sorted(i for i, f in fold_of.items() if f == fold)]
        train = [by_id[i] for i in sorted(i for i, f in fold_of.items() if f != fold)]
        out.append((fold, train, hold))
    return out


def design(train, hold):
    """(X_train, y_train, X_hold, hold_ids) with the shipped train-fold-only encoding."""
    params = _fit_standardization(train)
    x_train = [_feature_vector(p, params) for p in train]
    y_train = [float(p.treatment_arm) for p in train]
    x_hold = [_feature_vector(p, params) for p in hold]
    return x_train, y_train, x_hold, [p.patient_id for p in hold]


def pseudonym(patient_id: str) -> str:
    return hashlib.sha256(patient_id.encode()).hexdigest()[:10]


if __name__ == "__main__":
    start = time.perf_counter()
    patients, _ = canonical_dev_patients()
    arms = {1: 0, 0: 0}
    for p in patients:
        arms[int(p.treatment_arm)] += 1
    print(f"n={len(patients)} arms={arms} columns={len(COVARIATE_FIELDS)}")
    print("columns:", list(COVARIATE_FIELDS))
    assert len(patients) == 383 and arms == {1: 201, 0: 182}, "canonical cohort not reproduced"
    print(f"wall={time.perf_counter() - start:.2f}s")
