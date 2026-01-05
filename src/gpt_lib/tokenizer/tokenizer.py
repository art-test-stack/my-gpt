from gpt_lib.utils.schemas import TokenizerConfig, TrainingTokenizerConfig
from gpt_lib.tokenizer.pretokenizer import SimplePreTokenizer
# from gpt_lib.tokenizer.bpe import ByteLevelBPE
from gpt_lib.tokenizer.tok import TikTokenizer
# from rustbpe import BPETokenizer as RustByteLevelBPE

from tokenizers import Tokenizer as HFTokenizer
import torch
from typing import Callable, Iterable, List
import tiktoken

import random
from gpt_lib.utils.special_tokens import SpecialTokens

class DummyTokenizer:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab_size = config.vocab_size
        self.special_tokens = config.special_tokens
          
        self.bos_token_id = config.special_tokens.bos
        self.eos_token_id = config.special_tokens.eos
        self.pad_token_id = 0 # config.special_tokens.pad

    def batch_encode(self, texts, padding="max_length", to_torch=False, *args, **kwargs):
        encoded_batch = []
        for text in texts:
            encoded = self.encode(text, padding=padding, to_torch=to_torch, *args, **kwargs)
            encoded_batch.append(encoded)
        if to_torch:
            return torch.tensor(encoded_batch)
        return encoded_batch
    
    def encode(self, text, padding=None, to_torch=False, *args, **kwargs):
        assert isinstance(text, str), "Input text must be a string"
        
        length_encoded = random.randint(1, len(text) - 1)
        encoded = [random.randint(0, self.vocab_size - 1) for _ in range(length_encoded)]
        if padding == "max_length":
            while len(encoded) < self.config.max_context:
                encoded.append(self.pad_token_id)
        elif padding == "longest":
            pass
        encoded = encoded[-self.config.max_context:]
        if to_torch:
            return torch.tensor(encoded)
        return encoded

    def batch_decode(self, batch_tokens, *args, **kwargs):
        decoded_batch = []
        for tokens in batch_tokens:
            decoded = self.decode(tokens, *args, **kwargs)
            decoded_batch.append(decoded)
        return decoded_batch
    
    def decode(self, tokens, *args, **kwargs):
        return "".join([chr(t) for t in tokens])
     
     
def build_tokenizer(config: TokenizerConfig) -> Callable:
    return DummyTokenizer(config)

# def load_tokenizer(config: TokenizerConfig) -> Callable:
#     """ Load a tokenizer based on the provided configuration """
#     if config.source == "tiktoken":
#         tokenizer = TikTokenizer(config)
#     elif config.source == "bpe":
#         tokenizer = ByteLevelBPE.from_directory(config)
#     elif config.source == "rust_bpe":
#         tokenizer = RustByteLevelBPE.from_directory(config.dir)
#     elif config.source == "huggingface":
#         tokenizer = HFTokenizer.from_pretrained(config.name)
#     else:
#         raise ValueError(f"Unsupported tokenizer type: {config.source}")
    
#     return tokenizer


class Tokenizer:
    """ Wrapper class for different tokenizer implementations 
    ## Use cases include:
        - Loading classic TikToken tokenizer (GPT-3.5): `TikTokenizer()`
        - Loading a custom trained TikToken tokenizer from local directory: `Tokenizer.from_directory()`
        - Loading a pretrained tokenizer from HuggingFace Hub: `Tokenizer.from_pretrained()`
        - Loading a custom trained Byte-Level BPE tokenizer from local directory: `Tokenizer.from_directory()`
        - Training a new Byte-Level BPE tokenizer (python implementation) from corpus: `ByteLevelBPE()`
        - Training a new Byte-Level BPE tokenizer (Rust implementation) from corpus: `RustByteLevelBPE()`
        - Training a new HuggingFace tokenizer from corpus: `HFTokenizer()`
        - Training a new HuggingFace tokenizer from corpus and convert it in Tiktoken implementation: `HFTokenizer()`

    Args:
        tokenizer (Callable): The tokenizer instance to wrap
        config (Settings): Configuration settings for the tokenizer

    
    """
    def __init__(self, tokenizer: Callable, config: TokenizerConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.bos_token_id = self.encode_special(config.special_tokens.bos)

    @classmethod
    def from_pretrained(cls, config: TokenizerConfig):
        """ Load a pretrained tokenizer from HuggingFace Hub """
        if config.source == "tiktoken":
            tokenizer = TikTokenizer(config)
        elif config.source == "huggingface":
            tokenizer = HFTokenizer.from_pretrained(config.name)
        return cls(tokenizer, config)
    
    @classmethod
    def from_directory(cls, config: TrainingTokenizerConfig):
        if config.tokenizer_type == "bpe":
            tokenizer = ByteLevelBPE.from_directory(config)
        if config.tokenizer_type == "huggingface":
            tokenizer = HFTokenizer.from_pretrained(config.tokenizer_name)
        else:
            raise ValueError(f"Unsupported tokenizer type: {config.tokenizer_type}")
        
        return cls(tokenizer)
    
    @classmethod
    def train_from_iterator(
            cls,
            text_iterator: Iterable[str],
            config: TrainingTokenizerConfig
        ):
        tokenizer = HFTokenizer.new_from_iterator(text_iterator, vocab_size=config.vocab_size, special_tokens=config.special_tokens.list())
        return cls(tokenizer, config)
    
    def encode(
            self, 
            text: str,
            to_torch: bool = True
        ) -> List[int] | torch.Tensor:
        return self.tokenizer.encode(text, to_torch=to_torch)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)