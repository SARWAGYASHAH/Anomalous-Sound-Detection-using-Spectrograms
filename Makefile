PYTHON ?= python
CONFIG ?= config/default.yaml
MODEL ?=
AUDIO ?=

.PHONY: preprocess train train-dry evaluate predict serve test pipeline-colab mlflow-ui

preprocess:
	$(PYTHON) pipeline/01_preprocess.py --config $(CONFIG)

train:
	$(PYTHON) pipeline/02_train.py --config $(CONFIG)

train-dry:
	$(PYTHON) pipeline/02_train.py --config $(CONFIG) --dry-run --no-mlflow

evaluate:
	$(PYTHON) pipeline/03_evaluate.py --config $(CONFIG) $(if $(MODEL),--model-path $(MODEL),)

predict:
	$(PYTHON) pipeline/04_predict.py --config $(CONFIG) --file $(AUDIO) $(if $(MODEL),--model-path $(MODEL),)

serve:
	$(PYTHON) -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

test:
	$(PYTHON) -m pytest -q

pipeline-colab:
	$(PYTHON) run_pipeline.py --config $(CONFIG) --allow-training $(if $(AUDIO),--predict-file $(AUDIO),)

mlflow-ui:
	mlflow ui --backend-store-uri mlruns
