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
It depends on `SentenceModel`, and implements a sentence attention and a sentence encoder.

`SentenceModel.forward` takes a list of `torch.Tensors`.
A tensor represents to a sentence and is the index of the words in the sentence.
It returns a tuple of two tensors.
The first one is the sentence embeddings, and its shape is (num of sentences, `self.sentence_dim`).
The second one represents the attention.  The shape is (the length of the longest tensor in an input, num of sentences).

`DocumentModel.forward` takes documents.
A document is a list of tensors, and a tensor represents a sentence.
It returns a quadruple.
the first item represents document embeddings.
The second and third items represend sentence attention and word attention.
The fourth items is a list of the numbers of the sentences in a document.

You can instantiate them by `SentenceModelFactory` `DocumentModelFactory`.
They can accept pretrained word embeddings.

## Example
You can fit a model that depends on `DocumentModel` on AG News by the following comands.

    import han.example.document as d
    import torchtext.vocab as v
    import torch
    d.train(
        "d_enc.pth",
        "d_model.pth",
        device=torch.device("cpu"),
        embedding_sparse=False, 
		pre_trained=v.FastText(),
    )


## Test
You can run tests from the command line.

    pip install -e .[dev]
	python -m unittest
