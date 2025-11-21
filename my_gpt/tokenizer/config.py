from pydantic import BaseModel
from pathlib import Path
from my_gpt.tokenizer.special_tokens import SpecialTokens

PAT_STR_GPT2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_STR_GPT4 = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BaseConfig(BaseModel):
    vocab_size: int = 32_000
    name: str = "yc_tknzr"
    pat_str: str = PAT_STR_GPT4 
    tokenizer_dir: Path = Path("./.data/tokenizers") / name
    special_tokens: SpecialTokens = SpecialTokens()

class TrainingTokenizerConfig(BaseConfig):
    max_chars: int = 10_000_000_000
    chars_per_doc: int = 10_000
    merges_per_pass: int = 512

class TokenizerConfig(BaseConfig):
    mergeable_ranks: dict = {}