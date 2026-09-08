# Secret handling

Resolves Requirement 2.4: secrets and credentials must be loaded through an
approved secret mechanism and excluded from version control.

## Convention

- **Secrets are never hardcoded and never committed.** No config file,
  fixture, script, or source file under version control may contain a
  credential, API key, password, or access token.
- **Secrets are loaded from environment variables at runtime.** Any code
  that needs a credential (e.g. a future cloud-storage or data-warehouse
  client) reads it via `os.environ["SOME_VAR"]` (or `os.environ.get(...)`
  with an explicit, non-secret default) at the point of use, never from a
  config file that gets checked in.
- Local development values go in a `.env` file (already covered by
  `.gitignore`) or the shell environment, and are loaded with a standard
  environment-loading mechanism (e.g. `python-dotenv`) if one is added to
  the project's dependencies. `.env.example` (committed) documents which
  variable names are expected, with placeholder values only, e.g.:

  ```text
  # .env.example — copy to .env and fill in real values locally; never commit .env
  ACTG175_STORAGE_ACCESS_KEY=
  ACTG175_STORAGE_SECRET_KEY=
  ```

- In CI, secrets are provided via the CI provider's own secret store
  (e.g. encrypted repository/environment secrets), injected as environment
  variables at job runtime — never checked into a workflow file.
- Manifests, logs, and any other run artifact must not embed secret values.
  This is enforced automatically by `src/infra/manifest.py`'s
  `PrivacyScanner` (Requirement 1.11 / 2.7), which blocks a manifest write
  if it detects a secret-shaped string or a known credential field name.

## Automated enforcement

Two scripts share the same detection heuristics
(`src/infra/privacy_heuristics.py`) used by the manifest privacy scanner, so
none of these checks can drift out of sync with each other:

- **`scripts/check_no_secrets.py`** — a pre-commit / CI check that scans
  staged files (`git diff --cached`, the default) or every tracked file
  (`--all-tracked`) for secret-shaped strings, and fails the commit/build if
  one is found. Wired into `.pre-commit-config.yaml` at the repository root.
- **`scripts/scan_exports.py`** — an export-time scan (Requirement 2.7) over
  an arbitrary directory (e.g. a paper-artifact export or a `results/`
  subtree about to be shared), catching direct identifiers, quasi-identifiers,
  secrets, and unauthorized filesystem paths, not only secrets.

Both scanners are heuristic, structural checks — not a substitute for the
manual privacy review required before artifact release (Requirement 2.9).
