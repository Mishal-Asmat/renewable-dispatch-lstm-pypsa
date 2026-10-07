.PHONY: help install data train evaluate network notebook test all clean

PY ?= python

help:            ## show targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:         ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

data:            ## prepare and build data/generation_data.csv file
	$(PY) src/prepare_data.py

train:           ## train LSTM ensemble -> output/forecast.csv, predictions_test.npz
	$(PY) src/lstm_model.py

evaluate:        ## benchmark vs persistence / seasonal naive / gradient boosting
	$(PY) src/evaluate_baselines.py

network:         ## PyPSA carbon-tax scenarios, sensitivity, figures
	$(PY) src/build_network.py

notebook:        ## execute the analysis notebook in place
	jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

test:            ## run the unit tests
	$(PY) -m pytest -q

all: train evaluate network notebook   ## full pipeline (data file is already committed)

clean:           ## remove caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
