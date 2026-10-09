.PHONY: lint typecheck test test-cov ci clean

lint:
	python -m ruff check amacs/ tests/

typecheck:
	python -m mypy amacs/ --ignore-missing-imports

test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --cov=amacs --cov-report=term-missing --cov-report=html

ci: lint typecheck test-cov

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf htmlcov/ .coverage
