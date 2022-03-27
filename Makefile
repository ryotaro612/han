.PHONY: build deploy
build:
	rm dist/*
	python -m build
deploy: build
	python -m twine upload dist/*
