# Publishing

Nothing has been published yet. The name `dataset-fingerprint` was free on PyPI on 2026-11-22 (decide the final name first; see below).

## The recommended path

1. **Pick the final name** (`dataset-fingerprint`, `dataseal`, …). Renaming after a PyPI release is painful; before it, it is a search-and-replace.
2. **Make the GitHub repo public early.** JOSS and most reviewers want real public history, issues and CI runs, not a code dump.
   - Settings → Code security: enable *Private vulnerability reporting*, *Dependabot alerts*, *Secret scanning + push protection*.
   - Branch protection on `main`: require the `ci` and `security` checks, require PRs, block force-pushes.
3. **One-time Trusted Publishing setup (no API tokens ever)**
   - test.pypi.org and pypi.org → *Your account → Publishing → Add a pending publisher*:
     owner `rajanalamuri`, repo `dataset-fingerprint`, workflow `release.yml`, environments `testpypi` / `pypi`.
   - GitHub → Settings → Environments: create `testpypi` and `pypi`; on `pypi` add **yourself as required reviewer**
     and restrict it to tags `v*`.
4. **Dry run**: Actions → *release* → *Run workflow*. It runs CI + security, builds once, attests provenance and uploads
   to **TestPyPI**. Check it: `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple dataset-fingerprint`.
5. **Release**: bump `__version__` in `src/dataset_fingerprint/__init__.py`, add `## 0.1.0` to `CHANGELOG.md`, then
   `git tag v0.1.0 && git push --tags`. The workflow refuses to run if the tag, version and changelog disagree, publishes
   to TestPyPI, waits for your approval, publishes to PyPI (with PEP 740 attestations), and creates the GitHub Release
   with notes from the changelog.
6. **After release**: add badges (CI, PyPI version, Scorecard, coverage) to the README, publish the article with the
   explainer video, and consider a conda-forge recipe once there are users.
7. **Supply-chain extra mile**: pin every GitHub Action to a commit SHA (`pip install pinact`-style tools or
   `npx pin-github-action .github/workflows/*.yml`); Dependabot keeps the pins fresh.

## What runs where

| Workflow | When | Gates |
|---|---|---|
| `ci.yml` | every push / PR | ruff, format, mypy · unit tests on Linux + macOS × Python 3.9–3.13 (Windows experimental) · coverage ≥ 80 % · integration on S3/Azure/GCS emulators (fails if any backend is skipped) · build, `twine check --strict`, clean-venv install with zero deps, POC run |
| `security.yml` | every push / PR + weekly | bandit · security regression tests · pip-audit on all extras · gitleaks · CodeQL · dependency review |
| `scorecard.yml` | push to main + weekly | OpenSSF Scorecard |
| `release.yml` | tag `v*` / manual | reuses ci + security → build → provenance → TestPyPI → (approval) → PyPI → GitHub Release |

## Making it citable (Zenodo DOI)

1. Make the repository **public** (Zenodo only archives public repos).
2. Sign in at zenodo.org with GitHub → *GitHub* tab → flip the switch for `rajanalamuri/dataset-fingerprint`.
3. Create a GitHub **Release** (the release workflow already does this on `v*` tags). Zenodo archives it and mints a
   DOI using `.zenodo.json`; it also gives a *concept DOI* that always points at the latest version.
4. Add the DOI badge to the README and the DOI to `CITATION.cff` (`doi:` field).
