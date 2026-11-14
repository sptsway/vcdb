PY = .venv/bin/python
GO_BENCH = cd engine && go run ./cmd/vcbench -out ../results

.PHONY: setup test go-test py-test status fetch export bench go-bench bench-all plots

setup:
	python3.13 -m venv .venv
	$(PY) -m pip install -r requirements.txt

test: go-test py-test

go-test:
	cd engine && go vet ./... && go test -race ./...

py-test:
	$(PY) -m pytest -v

status:
	$(PY) tools/make_overview.py

fetch:
	$(PY) bench/fetch.py

# Write the vectors and ground truth as raw files for the Go benchmark.
export:
	$(PY) bench/export_data.py --n 200000

bench:
	$(PY) bench/compare.py --n 200000 --nlist 1024

go-bench: export
	$(GO_BENCH) -dir ../data/export/200k -tag 200k -nlist 1024

# Python and Faiss first, then Go, back to back on the same machine.
bench-all: bench go-bench

plots:
	$(PY) bench/plot.py
