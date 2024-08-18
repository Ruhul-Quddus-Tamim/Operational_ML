# Define paths
DATA_PATH=/YOUR_DATA_PATH/emotions.csv
PYTHONPATH=$(shell pwd):$(shell pwd)/modules

# Read MLflow configuration
MLFLOW_TRACKING_URI=$(shell python3.10 -c "import yaml; config = yaml.safe_load(open('mlflow_config.yaml')); print(config['mlflow']['tracking_uri'])")
MLFLOW_ARTIFACT_ROOT=$(shell python3.10 -c "import yaml; config = yaml.safe_load(open('mlflow_config.yaml')); print(config['mlflow']['artifact_root'])")

# Set GOOGLE_APPLICATION_CREDENTIALS environment variable
export PGPASSWORD="PGPASS"
export GOOGLE_APPLICATION_CREDENTIALS=/YOU_CREDENTIALS_PATH/mlflow.json

# PostgreSQL connection information
POSTGRES_HOST=X.Y.Z.A
POSTGRES_PORT=5432
POSTGRES_DB=YOUR_DB_NAME
POSTGRES_USER=YOUR_POSTGRE_USER_NAME
POSTGRES_PASSWORD=YOU_POSTGRE_PASS

# Targets
all: mlflow_server tune train clean stop

mlflow_server:
	@echo "Starting MLflow Server..."
	@mlflow server \
		--backend-store-uri postgresql+psycopg2://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@$(POSTGRES_HOST):$(POSTGRES_PORT)/$(POSTGRES_DB) \
        --default-artifact-root $(MLFLOW_ARTIFACT_ROOT) \
        --host 0.0.0.0 \
        --port 5000 &

tune:
	@echo "Running Metaflow Tuning Flow..."
	@export PYTHONPATH=$(PYTHONPATH) && \
	python3.10 metaflow_flows/tuning_flow.py run || { echo "Tuning flow failed"; exit 1; }
	@RUN_ID=$(shell cat /Users/pt/Desktop/ML_pipeline/ML_pipeline/.metaflow/TuningFlow/latest_run) && \
	echo "Tuning Flow Run ID: ${RUN_ID}"

train:
	@echo "Running Metaflow Training Flow..."
	@export PYTHONPATH=$(PYTHONPATH) && \
	RUN_ID=$$(cat /Users/pt/Desktop/ML_pipeline/ML_pipeline/.metaflow/TuningFlow/latest_run) && \
	python3.10 metaflow_flows/ml_pipeline_flow.py run --tuning_flow_run_id $$RUN_ID

clean:
	@echo "Cleaning up..."
	@rm -f tuning_run_id.txt run_id.txt

stop:
	@echo "Stopping Cloud SQL Proxy and MLflow Server..."
	@pkill -f "mlflow server"

.PHONY: all mlflow_server tune train clean

# lsof -i :5000