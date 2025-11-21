from pydantic import BaseModel
from my_gpt.utils.settings import CACHE_DIR

class BaseConfig(BaseModel):
    data_dir: str  = CACHE_DIR / ".data"

class DownloadConfig(BaseConfig):
    max_retries: int = 5
    retry_delay: int = 5  # in seconds
    num_workers: int = 4
    max_shards: int = 1000

class DatasetConfig(BaseModel):
    name: str
    # tokenizer: callable | None
    split: str = "train"
