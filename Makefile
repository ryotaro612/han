.PHONY: build deploy
build:
	python -m build
deploy: build
	python -m twine upload dist/*
