"""Word embedding."""
import typing as t
import torchtext.vocab as v


def build_vocabulary(
    sentences: t.Iterator[t.Iterator[str]],
    pad_symbol: str = "<pad>",
    unknown_symbol: str = "<unk>",
) -> v.Vocab:
    """Build vocabulary.

    Each element of `sentences` is a list of words.  The vocabulary
    encode unknown word to the indice of `unknown_symbol`.

    """
    vocab: v.Vocab = v.build_vocab_from_iterator(
        (sentence for sentence in sentences),
        special_first=True,
        specials=[pad_symbol, unknown_symbol],
    )
    vocab.set_default_index(1)
    return vocab
