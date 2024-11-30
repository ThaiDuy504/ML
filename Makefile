all: train predict evaluate visualization

train:
	python -m src.model.train
predict:
	python -m src.model.predict
evaluate:
	python -m src.evaluate.evaluate
visualization:
	python -m src.visualization.visualize