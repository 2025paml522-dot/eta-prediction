.PHONY: setup ingest features train serve test monitor lint

setup:
	pip install -r requirements.txt

ingest:
	python src/data/ingest.py --config config/config.yaml

features:
	python src/features/build_features.py --config config/config.yaml

train:
	python src/models/train.py --config config/config.yaml

serve:
	docker compose -f docker/docker-compose.yml up --build

test:
	pytest --cov=src --cov-report=term-missing

monitor:
	python src/monitoring/detect_drift.py

lint:
	ruff check src api
