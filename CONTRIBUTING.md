# Contributing to PSX Multi-Agent Trading System

Thank you for considering contributing! This document outlines the guidelines for contributing to the project.

## 🚀 Development Setup

### Prerequisites
- Python 3.12+
- `uv` (recommended) or `pip`
- A free Groq API key

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/agenticai-project.git
cd agenticai-project

# Create virtual environment
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install deps
uv sync

# Set up environment
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### Run the app

```bash
streamlit run app.py
```

### Run tests

```bash
make test        # if pytest tests exist
make lint        # ruff linting
make typecheck   # mypy type checking
```

---

## 📐 Coding Standards

1. **Python 3.12+** — use modern typing (`list[str]`, `| None`, etc.)
2. **Type hints** — required for all function signatures
3. **Docstrings** — use Google-style docstrings for all public functions
4. **Imports** — standard library → third-party → local, separated by blank line
5. **Line length** — 100 characters max
6. **Naming** — `snake_case` for functions/variables, `PascalCase` for classes

### Code style

```python
from __future__ import annotations

import os
from typing import Any

import pandas as pd


def calculate_metric(data: pd.Series, period: int = 14) -> float:
    """Calculate a metric over a rolling window.

    Args:
        data: Input price series.
        period: Rolling window length.

    Returns:
        The calculated metric value.
    """
    ...
```

---

## 📁 Project Structure Guidelines

- **agent/** — AI agent logic only (prompts, agent construction, routing)
- **tools/** — Tool functions that agents call (data, analysis, charts)
- **app.py** — Streamlit UI only (no business logic)
- **config.py** — All configuration constants in one place
- **docs/** — Documentation in Markdown

---

## 🔀 Git Workflow

1. Create a branch from `dev`: `git checkout -b feature/your-feature`
2. Make changes, commit with clear messages
3. Run linting/type checking
4. Push and open a PR to `dev`
5. After review, merge to `main`

### Commit message style
```
feat: add multi-agent supervisor routing
fix: correct Fibonacci level calculation for downswings
docs: update architecture diagram
refactor: extract position sizing logic
```

---

## 🧪 Testing

- Add tests for any new tool or agent logic
- Use `pytest` for unit tests
- Mock `psxdata` calls in tests (don't hit the live API)
- Run `make test` before submitting

---

## 🐛 Reporting Issues

Open an issue with:
- A clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Python version, OS, and provider (Groq/Gemini/OpenAI)

---

## 📬 Questions?

Open a discussion or issue. We're happy to help!
