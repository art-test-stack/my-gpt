from my_gpt.utils.settings import Settings
from typing import List
import regex as re

from tokenizers import NormalizedString, PreTokenizedString

class SimplePreTokenizer:
    def __init__(self, config: Settings):
        self.split_pattern = config.token_split_pattern
        self.special_tokens = config.special_tokens.list()
        self.tokenizer_type = config.tokenizer_type
        self.strip_spaces = False


    def _split(self, text: str) -> List[str]:
        if text == "":
            return []

        special_tokens = self.special_tokens
        safe_tokens = [re.escape(t) for t in special_tokens]

        # reg = (
        #     r"("
        #     + r"|".join(safe_tokens)
        #     + r"|\p{L}+"
        #     + r"|\d{1,3}(?:,\d{3})*(?:\.\d+)?"
        #     + r"|[^\p{L}\d\s]+"
        #     + r"|\s+"
        #     + r")"
        # )
        reg = f"({'|'.join(safe_tokens)}|{self.split_pattern})"

        parts = re.findall(reg, text, flags=re.UNICODE | re.VERBOSE)
        parts: List[str] = [p for p in parts if p != ""]

        tokens: List[str] = []
        buffer = ""

        for part in parts:
            if part.isspace():
                buffer = part
            elif part in special_tokens:
                tokens.append(part)
                buffer = ""
            else:
                tokens.append(buffer + part)
                buffer = ""

        if self.strip_spaces:
            tokens = [t.lstrip() for t in tokens if t.lstrip() != ""]
        return tokens

    def split(self, i: int, normalized_str: NormalizedString) -> List[NormalizedString]:
        text = normalized_str.get_str()
        tokens = self._split(text)
        return [NormalizedString(t) for t in tokens]

    def batch_split(self, texts: List[str]) -> List[str]:
        tokens = []
        for text in texts:
            tokens.extend(self._split(text))
        return tokens

    def __call__(self, text: str | List[str]) -> List[str] | List[List[str]]:
        if isinstance(text, list):
            return self.batch_split(text)
        return self.split(text)
    
    def pre_tokenize(self, pretok: PreTokenizedString) -> None:
        """In case of using the tokenizers library"""
        pretok.split(self.split)
