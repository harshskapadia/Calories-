# NutriSnap — common tasks. Override variables e.g. `make train CONFIG=configs/convnext_tiny.yaml`
PY      ?= .venv/bin/python
CONFIG  ?= configs/baseline_subset.yaml
CKPT    ?= checkpoints/best.pt

.PHONY: help setup data nutrition train train-baseline evaluate ablation test run-app export-model space deploy-space clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

setup:           ## create venv and install dependencies
	python3.11 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:            ## download Food-101 and build train/val/test splits (subset via config)
	$(PY) data/prepare_food101.py --config $(CONFIG)

nutrition:       ## rebuild data/nutrition.csv from USDA FoodData Central (FNDDS)
	$(PY) data/build_nutrition.py

train:           ## train a model from a config
	$(PY) -m src.train --config $(CONFIG)

train-baseline:  ## fast baseline on a small subset
	$(PY) -m src.train --config configs/baseline_subset.yaml

evaluate:        ## accuracy, top-5, per-class F1, confusion matrix
	$(PY) -m src.evaluate --checkpoint $(CKPT)

ablation:        ## backbone ablation (EfficientNet-B0 vs ConvNeXt-Tiny) + results table
	$(PY) -m src.ablation --configs configs/ablation_effnet_b0.yaml configs/ablation_convnext_tiny.yaml

test:            ## run unit tests
	$(PY) -m pytest

run-app:         ## launch the Gradio demo on http://localhost:7860
	$(PY) app/app.py

clean:           ## remove caches (keeps data and checkpoints)
	rm -rf .pytest_cache **/__pycache__

export-model:    ## fp16 app model from a checkpoint: make export-model CKPT=checkpoints/x/best.pt
	$(PY) scripts/export_model.py $(CKPT)

space:           ## assemble a deployable Hugging Face Space in build/space/
	$(PY) scripts/build_space.py

deploy-space:    ## upload build/space to HF Spaces (needs HF_TOKEN; SPACE=<user>/nutrisnap)
	.venv/bin/hf upload $(SPACE) build/space . --repo-type space
