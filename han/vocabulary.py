"""Word embedding."""
import typing as t
import torchtext.vocab as v


def build_vocabulary(
    sentences: t.Iterator[t.Iterator[str]],
    unknown_index: int = 0,
) -> v.Vocab:
    """Build vocabulary.

    Each element of `sentences` is a list of words.

    Assign `unknown_index` to a word that the retuened vocabulary
    don't know. The default value is 0, which may be same as an indice
    for padding. We use 0 because we have to pass an indice for
    unknown words to an embedding layer when we train a model while we
    can feed the words in training dataset to a `Vocabulary`.

    """
    vocab: v.Vocab = v.build_vocab_from_iterator(
        (sentence for sentence in sentences)
    )
    vocab.set_default_index(unknown_index)
    return vocab
