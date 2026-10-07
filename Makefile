PYTHON ?= python
PIP = $(PYTHON) -m pip

.PHONY: setup test check demo dataset-help

setup:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

check:
	$(PIP) check

demo:
	$(PYTHON) -m streamlit run app/streamlit_app.py

dataset-help:
	$(PYTHON) scripts/build_dataset.py --help
