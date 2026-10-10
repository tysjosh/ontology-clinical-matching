"""The frozen Section 13 adjustment set, mapped to stored concept ids and
encodings (`decisions.md` Section 13; Section 26 amendment of 2026-09-30,
item 2; neural-experiment-pipeline task 7.5).

This is the single, explicit Section-13-name -> concept-id(s) -> encoding
mapping. `causal.propensity` builds its feature vector from it, and
`causal.overlap` is meant to read it for the balance report (task 7.6), so
both sides agree on which stored key a Section 13 covariate lives under.
Lookups never rely on name equality: several Section 13 names differ from
the `static_covariates` keys the loader writes
(`data.loaders.actg175_adapter`):

============================================  ========================================
Section 13 / `adjustment_set` name            stored concept id(s), checked in order
============================================  ========================================
`age`                                         `age`
`weight`                                      `weight_kg`, then `weight`
`sex`                                         `sex`
`race_ethnicity`                              `race_ethnicity`
`hemophilia`                                  `hemophilia`
`homosexual_activity`                         `homosexual_activity`
`intravenous_drug_use_history`                `iv_drug_use_history`
`karnofsky_score`                             `karnofsky_score`
`prior_opportunistic_infection`               `prior_opportunistic_infection`
`symptomatic_status`                          `symptomatic_status`
`baseline_cd4`                                `baseline_cd4`
`baseline_cd8`                                `baseline_cd8`
`prior_antiretroviral_exposure_and_duration`  `prior_art_exposure_days`, then
                                              `prior_antiretroviral_exposure_duration`
============================================  ========================================

The second id for `weight` and for prior ART is the normalized name from
`configs/normalization/concept_unit_mapping.yaml`. The pipeline does not
currently apply that normalization to `static_covariates`, so the loader's
own id is what is found in practice; the normalized id is a documented
fallback, never a guess. The first id with a non-`None` stored value wins.

Only baseline (`demographic` / `baseline` modality) keys appear here. The
normalized `cd4_count` / `cd8_count` ids are deliberately *not* listed for
`baseline_cd4` / `baseline_cd8`: those ids also name the longitudinal
week-20/96 lab events, which Section 13 excludes from adjustment.

This module is pure data plus one lookup helper. It imports nothing from
`torch` and nothing from the rest of `causal`, so any torch-free consumer
can share it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

#: How a Section 13 covariate becomes feature columns. See
#: `causal.propensity`'s module docstring for the exact per-kind rules.
EncodingKind = Literal["continuous", "boolean", "binary_category", "exposure_and_duration"]


@dataclass(frozen=True)
class AdjustmentCovariate:
    """One Section 13 covariate and how it is read and encoded.

    Attributes:
        name: the Section 13 / `adjustment_set` name.
        concept_ids: the stored `static_covariates` keys to try, in order.
        encoding: the encoding kind (see `EncodingKind`).
        columns: the encoded feature-column names this covariate produces,
            in order. Their concatenation over `SECTION_13_ADJUSTMENT_SET`
            is `causal.propensity.COVARIATE_FIELDS`.
        reference_level: for `binary_category`, the level encoded as `0.0`.
        allowed_levels: for `binary_category`, the closed set of values that
            count as a category at all, or `None` for a free string. A value
            outside a closed set is treated as missing.
    """

    name: str
    concept_ids: tuple[str, ...]
    encoding: EncodingKind
    columns: tuple[str, ...]
    reference_level: str | None = None
    allowed_levels: tuple[str, ...] | None = None


#: The frozen Section 13 adjustment set, in Section 13's own listing order
#: (which is also `configs/target_trials/main.yaml`'s `adjustment_set`
#: order). Thirteen covariates, fourteen encoded columns.
SECTION_13_ADJUSTMENT_SET: tuple[AdjustmentCovariate, ...] = (
    AdjustmentCovariate("age", ("age",), "continuous", ("age",)),
    AdjustmentCovariate("weight", ("weight_kg", "weight"), "continuous", ("weight",)),
    AdjustmentCovariate(
        "sex",
        ("sex",),
        "binary_category",
        ("sex_female",),
        reference_level="M",
        allowed_levels=("M", "F"),
    ),
    AdjustmentCovariate(
        "race_ethnicity",
        ("race_ethnicity",),
        "binary_category",
        ("race_ethnicity_non_white",),
        reference_level="white",
    ),
    AdjustmentCovariate("hemophilia", ("hemophilia",), "boolean", ("hemophilia",)),
    AdjustmentCovariate(
        "homosexual_activity", ("homosexual_activity",), "boolean", ("homosexual_activity",)
    ),
    AdjustmentCovariate(
        "intravenous_drug_use_history",
        ("iv_drug_use_history",),
        "boolean",
        ("intravenous_drug_use_history",),
    ),
    AdjustmentCovariate(
        "karnofsky_score", ("karnofsky_score",), "continuous", ("karnofsky_score",)
    ),
    AdjustmentCovariate(
        "prior_opportunistic_infection",
        ("prior_opportunistic_infection",),
        "boolean",
        ("prior_opportunistic_infection",),
    ),
    AdjustmentCovariate(
        "symptomatic_status", ("symptomatic_status",), "boolean", ("symptomatic_status",)
    ),
    AdjustmentCovariate("baseline_cd4", ("baseline_cd4",), "continuous", ("baseline_cd4",)),
    AdjustmentCovariate("baseline_cd8", ("baseline_cd8",), "continuous", ("baseline_cd8",)),
    AdjustmentCovariate(
        "prior_antiretroviral_exposure_and_duration",
        ("prior_art_exposure_days", "prior_antiretroviral_exposure_duration"),
        "exposure_and_duration",
        ("prior_art", "prior_art_duration"),
    ),
)

#: `{Section 13 name: AdjustmentCovariate}`.
ADJUSTMENT_COVARIATES_BY_NAME: dict[str, AdjustmentCovariate] = {
    covariate.name: covariate for covariate in SECTION_13_ADJUSTMENT_SET
}

#: The thirteen Section 13 names, in order.
ADJUSTMENT_SET_NAMES: tuple[str, ...] = tuple(c.name for c in SECTION_13_ADJUSTMENT_SET)

#: Every encoded feature column, in order (14 columns).
ENCODED_COLUMNS: tuple[str, ...] = tuple(
    column for covariate in SECTION_13_ADJUSTMENT_SET for column in covariate.columns
)


def resolve_raw_value(
    static_covariates: Mapping[str, Any], covariate: AdjustmentCovariate
) -> Any | None:
    """The stored value for `covariate`, or `None` if no mapped key holds one.

    Tries `covariate.concept_ids` in order and returns the first value that
    is present and not `None`. Never falls back to `covariate.name` itself:
    a Section 13 name is only a stored key when it is listed explicitly.
    """
    for concept_id in covariate.concept_ids:
        value = static_covariates.get(concept_id)
        if value is not None:
            return value
    return None
