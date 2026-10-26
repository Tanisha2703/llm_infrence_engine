.PHONY: install test test-fast serve loadtest figures up down

install:        ## install with everything needed to test, serve and evaluate
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev,eval,serve]"

test:           ## full suite (downloads the model once; needs ~6 GB RAM)
	pytest -q

test-fast:      ## skip the one test that loads the 1.5B model (needs ~9 GB RAM)
	pytest -q -m "not heavy"

serve:          ## run the server on http://127.0.0.1:8000
	python -m miniserve.server --engine paged

loadtest:       ## open-loop load test; starts its own server
	python benchmarks/loadtest.py --spawn "--engine paged" --rate 0.5 --requests 30

figures:        ## regenerate the README figures from results/
	python benchmarks/make_figures.py

up:             ## server + Prometheus + Grafana in containers (Docker or Podman)
	docker compose up --build

down:
	docker compose down
