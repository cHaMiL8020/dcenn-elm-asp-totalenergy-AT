SHELL := /bin/bash

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: help venv install preprocess train asp visualize benchmark benchmark-quick benchmark-best pipeline clean

help:
	@echo "Available targets:"
	@echo "  make venv        - Create virtual environment (.venv)"
	@echo "  make install     - Install Python dependencies"
	@echo "  make preprocess  - Run data preprocessing"
	@echo "  make train       - Train dCeNN-ELM model"
	@echo "  make asp         - Run ASP anomaly checks"
	@echo "  make visualize   - Generate result visualization plot"
	@echo "  make benchmark   - Run full dCeNN-ELM benchmark sweep"
	@echo "  make benchmark-quick - Run quick benchmark sweep"
	@echo "  make benchmark-best - Run only the best tuned benchmark config"
	@echo "  make pipeline    - Run full pipeline (preprocess -> train -> asp -> visualize)"
	@echo "  make clean       - Remove generated outputs"

venv:
	@test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	@echo "Virtual environment ready at $(VENV_DIR)"

install: venv
	$(PIP) install -r requirements.txt

preprocess:
	$(PYTHON) src/01_preprocess.py

train:
	$(PYTHON) src/02_train_dcenn_elm.py

asp:
	$(PYTHON) src/03_apply_asp.py

visualize:
	$(PYTHON) src/04_visualize_results.py

benchmark:
	$(PYTHON) src/05_benchmark_dcenn_elm.py

benchmark-quick:
	$(PYTHON) src/05_benchmark_dcenn_elm.py --quick

benchmark-best:
	$(PYTHON) src/05_benchmark_dcenn_elm.py --best-only

pipeline: preprocess train asp visualize

clean:
	rm -f data/processed_15min.csv
	rm -f data/predictions_2024.csv
	rm -f data/flagged_anomalies.csv
	rm -f notebooks/neuro_symbolic_plot.png
	@echo "Generated files removed"
