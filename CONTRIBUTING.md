# Contributing

```bash
git clone https://github.com/rajanalamuri/mlfingerprint && cd mlfingerprint
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

| Command | What it runs |
|---|---|
| `make test` | unit + security tests (seconds, no network) |
| `make emulators` | starts Azurite + fake-gcs-server in Docker (S3 runs in-process via moto) |
| `make integration` | integration tests against S3, Azure and GCS emulators |
| `make security` | bandit + security tests + pip-audit |
| `make lint` | ruff check, ruff format --check, mypy |
| `make build` | sdist + wheel + `twine check` |
| `make all` | everything CI runs |

Rules of thumb: the core keeps **zero runtime dependencies**; anything optional is lazy-imported behind an extra.
Every bug fix gets a test; every security fix gets a test in `tests/security`.

README animations (hero, concept explainer, CLI demo) live in `docs/assets/*.svg` (pure SVG + CSS, light and dark). Regenerate them with
`python scripts/make_readme_assets.py`.
