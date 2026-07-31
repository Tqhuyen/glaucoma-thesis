.PHONY: setup lint test smoke train-nlp train-g96 train-g128 ddp tb

setup:
	pip install -e ".[dev,glaucoma,nlp]"

lint:
	ruff check pipeline models tests
	ruff format --check pipeline models tests

test:
	pytest tests -q

smoke:
	python -m pipeline.train --cfg configs/glaucoma_96.yaml --smoke --output-dir /tmp/smoke

train-nlp:
	python -m pipeline.train --cfg configs/base.yaml

train-g96:
	python -m pipeline.train --cfg configs/glaucoma_96.yaml

train-g128:
	python -m pipeline.train --cfg configs/glaucoma_128.yaml

ddp:
	bash scripts/launch_ddp.sh configs/glaucoma_96.yaml

tb:
	tensorboard --logdir outputs --port 6006
