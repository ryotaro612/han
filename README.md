# Hierarchical Attention Networks

## Abstract

An implementation of [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf) in [PyTorch](https://pytorch.org/).

## Installation
You can install [the package](https://pypi.org/project/hierarchical-attention-networks/) from pip:

    pip install hierarchical-attention-networks


## Requirements

You can see the requirements in `setup.cfg`.

## Usage
This package provides two neural networks.
The first one is `SentenceModel` in `han.model.sentence`.
It implements a word encoder and a word attention.
The second one is `DocumentModel` in `han.model.document`.
It contains `SentenceModel`, and implements sentence attention and sentence encoder.



`han.model.sentence` expose `SentenceModel`.

## Test
You can run tests from the command line.

    pip install -e .[dev]
	python -m unittest
