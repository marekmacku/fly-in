.PHONY: install run debug clean lint lint-strict

install:
	python3 -m pip install -r requirements-dev.txt

run:
	python3 main.py $(MAP)

debug:
	python3 -m pdb main.py $(MAP)

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache fly_in/__pycache__ tests/__pycache__

lint:
	python3 -m flake8 .
	python3 -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	python3 -m flake8 .
	python3 -m mypy . --strict
