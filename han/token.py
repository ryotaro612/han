"""Text tokenization."""
import torchtext.data as d


class Tokenizer:
    """Tokenizer."""

    def __init__(self, tokenizer: str = "basic_english"):
        """Take a tokenizer name.

        If tokenizer is basic_english,
        __call__ makes letters lower.

        References
        ----------
        https://pytorch.org/text/stable/data_utils.html#torchtext.data.utils.get_tokenizer

        """
        self.tokenizer = d.get_tokenizer(tokenizer)

    def __call__(self, text: str) -> list[str]:
        """Tokenize text."""
        return self.tokenizer(text)
