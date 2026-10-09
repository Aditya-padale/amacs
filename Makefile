.PHONY: lint typecheck test test-cov verify ci clean

lint:
	python -m ruff check amacs/ tests/ benchmarks/ examples/

typecheck:
	python -m mypy amacs/ benchmarks/ --ignore-missing-imports

test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --cov=amacs --cov-report=term-missing --cov-fail-under=80

verify: lint typecheck test-cov
	rm -rf dist/ build/ *.egg-info .venv_verify
	python -m build
	python -m venv .venv_verify
	.venv_verify/bin/pip install dist/*.whl
	.venv_verify/bin/amacs --version
	.venv_verify/bin/amacs benchmark
	.venv_verify/bin/python examples/basic_usage.py
	rm -rf .venv_verify dist/ build/ *.egg-info

ci: verify

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info .venv_verify
