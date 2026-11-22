.PHONY: lint test integration security build emulators emulators-down all
PY ?= python

lint:
	ruff check src tests examples
	ruff format --check src tests examples
	mypy src

test:
	$(PY) -m pytest -m "not integration" --cov

emulators:
	docker compose -f docker-compose.emulators.yml up -d

emulators-down:
	docker compose -f docker-compose.emulators.yml down

integration:
	STORAGE_EMULATOR_HOST=http://127.0.0.1:4443 $(PY) -m pytest -m integration -rs

security:
	bandit -c pyproject.toml -r src
	$(PY) -m pytest -m security
	pip-audit --skip-editable

build:
	rm -rf dist && $(PY) -m build && twine check --strict dist/*

all: lint test security integration build
