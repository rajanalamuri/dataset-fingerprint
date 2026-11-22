.PHONY: lint test integration security build emulators emulators-down benchmark all
PY ?= python

lint:
	ruff check src tests examples scripts benchmarks
	ruff format --check src tests examples scripts benchmarks
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
	pip freeze --exclude-editable | grep -vi "^dataset-fingerprint" > .audit-req.txt && pip-audit -r .audit-req.txt --strict --no-deps --disable-pip; rm -f .audit-req.txt

build:
	rm -rf dist && $(PY) -m build && twine check --strict dist/*

all: lint test security integration build

benchmark:
	bash benchmarks/download.sh .bench/raw
	$(PY) benchmarks/run_benchmark.py --raw .bench/raw --work .bench/work --out benchmarks/results
	$(PY) benchmarks/report.py > /dev/null
