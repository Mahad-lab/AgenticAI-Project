.PHONY: install run lint typecheck test clean docs

# ─── Installation ──────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-uv:
	uv sync

# ─── Run ───────────────────────────────────────────────────────────────────────
run:
	streamlit run app.py

dev:
	streamlit run app.py --server.runOnSave true

# ─── Code Quality ──────────────────────────────────────────────────────────────
lint:
	ruff check agent/ tools/ *.py
	ruff format --check agent/ tools/ *.py

lint-fix:
	ruff check --fix agent/ tools/ *.py
	ruff format agent/ tools/ *.py

typecheck:
	mypy agent/ tools/ --ignore-missing-imports

# ─── Test ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ -v --tb=short --cov=agent --cov=tools

# ─── Clean ──────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache

clean-all: clean
	rm -rf .venv/

# ─── Docs ──────────────────────────────────────────────────────────────────────
docs:
	@echo "Documentation available in docs/ directory"
	@echo "  - docs/ARCHITECTURE.md"
	@echo "  - docs/STRATEGY.md"
	@echo "  - docs/API.md"

# ─── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "Available commands:"
	@echo "  make install       Install dependencies (pip)"
	@echo "  make install-uv    Install dependencies (uv)"
	@echo "  make run           Run Streamlit app"
	@echo "  make dev           Run with auto-reload"
	@echo "  make lint          Check code style"
	@echo "  make lint-fix      Fix code style"
	@echo "  make typecheck     Run mypy type checking"
	@echo "  make test          Run tests"
	@echo "  make clean         Remove cache files"
	@echo "  make docs          View documentation"
