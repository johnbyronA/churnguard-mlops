.PHONY: install lint test pipeline retrain site serve mlflow docker up down clean

install:            ## install everything (training + dev)
	pip install -r requirements.txt && pip install --no-deps -e .

lint:
	ruff check src tests

test:
	pytest --cov=churnguard --cov-report=term-missing

pipeline:           ## generate -> train -> gate -> simulate drift -> monitor
	churnguard run-all
	churnguard summary

retrain:            ## continuous-training cycle on a drifted, labelled batch
	churnguard simulate --drift --label-feedback --n-customers 8000
	churnguard monitor
	churnguard train
	churnguard promote
	churnguard summary

site:               ## rebuild the interactive demo from the current champion and check JS/Python parity
	python scripts/build_site.py
	node site/parity_test.js

serve:
	uvicorn churnguard.serving.app:app --reload --port 8000

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker:
	docker build -t churnguard-api:latest .

up:                 ## API + Prometheus + Grafana
	docker compose up --build -d

down:
	docker compose down

clean:
	rm -rf data mlruns mlflow.db logs models/challenger models/archive .pytest_cache .coverage coverage.xml
