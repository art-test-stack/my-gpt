from my_gpt.utils.settings import Settings

from pathlib import Path
from typing import List

import tiktoken
from tiktoken import _tiktoken

tiktoken_models = list(tiktoken.model.MODEL_TO_ENCODING.keys())

class TikTokenizer:
    def __init__(
            self,
            config: Settings
        ) -> None:
        assert config.tokenizer_name in tiktoken_models, f"'{config.tokenizer_name}' is not a provided model."
        self.model = tiktoken.encoding_for_model(config.tokenizer_name)

        self.pat_str = self.model._pat_str
        self.mergeable_ranks = self.model._mergeable_ranks

        token_ids = range(
            self.model.n_vocab, self.model.n_vocab + len(config.special_tokens.list()))

        special_tokens = config.special_tokens.list() if type(config.special_tokens) == list else list(config.special_tokens)
        self.model._special_tokens = {
            token: id
            for token, id in zip(special_tokens, token_ids)
        }
        self.model._core_bpe = _tiktoken.CoreBPE(
            self.mergeable_ranks, self.model._special_tokens, self.pat_str)
        self.special_tokens = self.model._special_tokens

    def encode(
            self, 
            text: str,
            retrieve_splitted_text: bool = True, 
            verbose: bool = False
        ):
        token_ids = self.model.encode(text, allowed_special="all")
        if retrieve_splitted_text:
            return list(zip(token_ids, self.get_words(token_ids)))
        return token_ids
    

    def decode(self, token_ids: List[int]):
        return self.model.decode(token_ids) 

    def get_words(
            self,
            token_ids: List[int]
        ) -> List[str]:
        words = [ self.decode([tk]) for tk in token_ids ]
        return words