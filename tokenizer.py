"""
A minimal character-level tokenizer.

Char-level keeps the whole project self-contained (no external tokenizer
files, no vocab downloads) and makes it easy to see exactly how text
turns into integers and back. The vocab is built directly from whatever
text you train on, and saved alongside the model checkpoint so
generation later uses the exact same mapping.
"""

import tiktoken

class CharTokenizer:
    """
    We wrap tiktoken but keep the old class name to minimize changes elsewhere.
    This now uses GPT-2 BPE tokenization.
    """
    def __init__(self, text=None, vocab=None):
        self.enc = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self):
        return self.enc.n_vocab

    def encode(self, s):
        return self.enc.encode(s, allowed_special="all")

    def decode(self, ids):
        return self.enc.decode(ids)

    def merge_vocab(self, new_text):
        # Subword tokenizers have fixed vocabularies, no need to merge
        return False

    def to_dict(self):
        return {}

    def save(self, path):
        # Vocabulary is fixed and managed by tiktoken, no need to save
        pass

    @classmethod
    def load(cls, path):
        return cls()
