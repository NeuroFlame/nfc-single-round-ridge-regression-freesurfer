PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff

.PHONY: setup-dev lint lint-author format format-author format-check check clean-dev

setup-dev: $(RUFF)

$(RUFF): requirements-dev.txt pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements-dev.txt

lint: $(RUFF)
	$(RUFF) check .

lint-author: $(RUFF)
	$(RUFF) check app/code/computation

format: $(RUFF)
	$(RUFF) check --fix .
	$(RUFF) format .

format-author: $(RUFF)
	$(RUFF) check --fix app/code/computation
	$(RUFF) format app/code/computation

format-check: $(RUFF)
	$(RUFF) format --check .

check: lint format-check

clean-dev:
	rm -rf $(VENV) .ruff_cache
