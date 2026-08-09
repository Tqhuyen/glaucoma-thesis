.PHONY: setup lint test smoke sanity train-nlp train train-g ddp tb plot

setup:
	pip install -e ".[dev,glaucoma,nlp]"

lint:
	ruff check pipeline models tests
	ruff format --check pipeline models tests

test:
	pytest tests -q

smoke:
	python -m pipeline.train --cfg configs/glaucoma.yaml --smoke --output-dir /tmp/smoke

sanity:
	python -m pipeline.train --cfg configs/glaucoma.yaml --sanity --output-dir /tmp/sanity

train-nlp:
	python -m pipeline.train --cfg configs/base.yaml

train:
	python -m pipeline.train --cfg configs/glaucoma.yaml

train-g:
	python -m pipeline.train --cfg configs/glaucoma.yaml

ddp:
	bash scripts/launch_ddp.sh configs/glaucoma.yaml

tb:
	tensorboard --logdir outputs --port 6006

plot:
	python -c "from pathlib import Path; from pipeline.plotting import render_and_sync; import sys; render_and_sync(Path(sys.argv[1]), {'logging': {'plot_curves': True, 'drive_sync_dir': ''}})" $(ARGS)
