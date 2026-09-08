# Causal Witness Patient World Model

Research pipeline for the Causal Witness Patient World Model (ACTG175 critical path).

See `.kiro/specs/causal-witness-pwm/` for the requirements, design, and frozen decisions
that govern this implementation. `decisions.md` is the single source of truth for every
dataset, model, threshold, and statistical choice; this codebase must not reintroduce
discretion that document has already closed.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Repository layout

```text
configs/     dataset contracts, target-trial specs, experiment configs
src/         infra, data, protocols, cohorts, causal, models, uncertainty,
             witnesses, evaluation, reporting packages
scripts/     one-command entry points (smoke experiment, ablations, reports)
tests/       unit, integration, property, and e2e tests
results/     versioned per-experiment output (gitignored)
docs/        data access policy, limitations, responsible-use documentation
```
