from typing import List
from pydantic import BaseModel


class SpecialTokens(BaseModel):
    pad: str = "<|padding|>"
    unknown: str = "<|unknown|>"
    tab: str = "<|tab|>"
    new_line: str = "<|new_line|>"
    bos: str = "<|bos|>"
    eos: str = "<|eos|>"

    start_of_assistant: str = "<|start-of-assistant|>"
    end_of_assistant: str = "<|end-of-assistant|>"

    start_of_user: str = "<|start-of-user|>"
    end_of_user: str = "<|end-of-user|>"
    
    start_of_answer: str = "<|start-of-answer|>"
    end_of_answer: str = "<|end-of-answer|>"
    
    def list(self) -> List[str]:
        """
        Returns a list of all special tokens.
        """
        return [v for v in self.__dict__.values() if v is not None]
    
    def dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}