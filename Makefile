.PHONY: install run debug clean lint lint-strict

# Prefer the project virtual environment when it exists.
PYTHON := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

run:
	$(PYTHON) main.py $(MAP)

debug:
	$(PYTHON) -m pdb main.py $(MAP)

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache \
		fly_in/__pycache__ fly_in2/__pycache__ tests/__pycache__

lint:
	$(PYTHON) -m flake8 .
	$(PYTHON) -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(PYTHON) -m flake8 .
	$(PYTHON) -m mypy . --strict
