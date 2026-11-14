PY = .venv/bin/python

.PHONY: setup test status fetch bench plots

setup:
	python3.13 -m venv .venv
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -v

status:
	$(PY) tools/make_overview.py

fetch:
	$(PY) bench/fetch.py

bench:
	$(PY) bench/compare.py --n 200000 --nlist 1024

plots:
	$(PY) bench/plot.py
