from typing import List
from pathlib import Path
import torch
from my_gpt.utils.settings import Settings
from my_gpt.tokenizer.pretokenizer import SimplePreTokenizer
import pickle

simple_corpus = [
    "Hello world!",
    "I am a simple string, here to help my creator to debug.",
    "Do you know that it will not be an optimized version?",
    "Why do you take so much pain for something that already works well?"
    "I love pain and I love to debug"
]

class ByteLevelBPE:
    """
    Simple implementation of the Byte-level BPE algorithm in Python 3.10.11

    Args:
        - vocab_size (int): Size of the vocabulary set at the end of the training

    """
    def __init__(
            self, 
            config: Settings = Settings(),
        ):
        self.vocab_size = config.vocab_size
        self.token_split_pattern = config.token_split_pattern
        self.special_tokens = config.special_tokens
        self.forced_tokens = config.forced_tokens
        self.dir = config.tokenizer_dir
        self.pretokenizer = SimplePreTokenizer(config)

        self.corpus = []
        self.token_id = {}
        
    @classmethod
    def from_directory(cls, config: Settings):
        tokenizer = cls(config)
        with open(config.tokenizer_dir / "bpe_tokenizer.pkl", "rb") as f:
            tokenizer.token_id = pickle.load(f)
        return tokenizer

    def add_corpus(self, corpus: str):
        self.corpus.append(corpus.encode('utf-8'))

    def make_vocab(self):
        tokens = self.pretokenizer.batch_split(self.corpus)

        # All Unicode code points from 0 to 0x10FFFF (1,114,111 characters)
        # Note: This creates a very large set and may consume significant memory
        # All valid UTF-8 byte values (0-255)
        vocab = {}

        freqs = {}
        for token in tokens:
            # Convert to bytes to handle unicode characters
            for tok in token.split():
                freqs[tok] = freqs.get(tok, 0) + 1
        freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        for token, freq in freqs:
            if len(vocab) >= self.vocab_size:
                break
            vocab[token] = freq
        while len(vocab) < self.vocab_size:
            pass


    def encode(
            self, 
            x: List[str] | List[List[str]],
            output_type: str = "to_tensor"
        ) -> torch.Tensor:
        if isinstance(x[0], List[str]):
            return self.encode_batch(x)
        # Simple and batch encoding
        pass
    
    def encode_batch(self, x: List[List[str]]) -> torch.Tensor:
        pass

    def decode(self, x: torch.Tensor) -> List[str] | List[List[str]]:
        if len(x.shape) == 3:
            return self.decode_batch(x)
        # Simple and Batch decoding
        pass

    def decode_batch(self, x: torch.Tensor) -> List[List[str]]:
        pass

    def save_tokenizer(self):
        pass