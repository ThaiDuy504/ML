all: train predict evaluate

train:
	python -m src.model.train
predict:
	python -m src.model.predict
evaluate:
	python -m src.evaluate.evaluate
