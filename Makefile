all: train predict evaluate visualization

preprocess:
	python -m src.preprocess.preprocessImage
train:
	python -m src.model.train
predict:
	python -m src.model.predict
evaluate:
	python -m src.evaluate.evaluate
visualization:
	python -m src.visualization.visualize