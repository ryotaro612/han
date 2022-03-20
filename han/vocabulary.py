"""Word embedding."""
import typing as t
import torchtext.vocab as v


def build_vocabulary(
    sentences: t.Iterator[t.Iterator[str]],
    unknown_index: int = 0,
) -> v.Vocab:
    """Build vocabulary.

    Each element of `sentences` is a list of words.

    """
    vocab: v.Vocab = v.build_vocab_from_iterator(
        (sentence for sentence in sentences)
    )
    vocab.set_default_index(unknown_index)
    return vocab
