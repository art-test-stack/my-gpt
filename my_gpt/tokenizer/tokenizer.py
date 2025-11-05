from my_gpt.utils.settings import Settings
from my_gpt.tokenizer.pretokenizer import SimplePreTokenizer
from my_gpt.tokenizer.bpe import ByteLevelBPE
from my_gpt.tokenizer.tok import TikTokenizer
from my_gpt.tokenizer.rustbpe import BPETokenizer as RustByteLevelBPE

from tokenizers import Tokenizer as HFTokenizer
import torch
from typing import Callable, Iterable, List
import tiktoken


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
    def __init__(self, tokenizer: Callable, config: Settings):
        self.tokenizer = tokenizer
        self.config = config
        self.bos_token_id = self.encode_special(config.special_tokens.bos)

    @classmethod
    def from_pretrained(cls, config: Settings):
        """ Load a pretrained tokenizer from HuggingFace Hub """
        if config.tokenizer_type == "tiktoken":
            tokenizer = TikTokenizer(config)
        elif config.tokenizer_type == "huggingface":
            tokenizer = HFTokenizer.from_pretrained(config.tokenizer_name)
        return cls(tokenizer, config)
    
    @classmethod
    def from_directory(cls, config: Settings):
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
            config: Settings
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