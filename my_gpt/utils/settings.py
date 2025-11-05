import torch
import os
from pathlib import Path
from pydantic import BaseModel
from typing import List
from my_gpt.tokenizer.special_tokens import SpecialTokens

# ----------- PROCESSOR -----------

CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()
if MPS_AVAILABLE:
    torch.mps.empty_cache()
    torch.mps.set_per_process_memory_fraction(0.)
DEVICE_NAME = "cuda" if CUDA_AVAILABLE else "mps" if MPS_AVAILABLE else "cpu"
DEVICE = torch.device(DEVICE_NAME)

NUM_THREADS = os.cpu_count() # 16

# ------------- DATA -------------

IS_TIKTOKEN = False # TODO: parse as arg

SPECIAL_TOKENS = SpecialTokens()
FORCED_TOKENS = ["AI", "Michel", "GPT", "MichelGPT", "michelgpt"]

DATA_FOLDER = "data"
MIN_DOCUMENT_SIZE = 0
OUTPUT_FOLDER = "output"
MODEL_FOLDER = "models"
VOCAB_SIZE = 32_000
VOCAB_FILE = DATA_FOLDER + "/vocab.json"
MAX_TOKEN_LENGTH = 32

# TOKEN_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" # GPT 4 SPLIT
TOKEN_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # GPT 2 SPLIT

# ------------- DRIVE -------------

SAVE_ON_DRIVE = True
DRIVE_FILE = ""
SAVE_ON_WANDB = True

# ------------- MODEL -------------

VOCAB_SIZE = 32_000
MAX_CONTEXT = 64

NUM_HEADS = 2
NUM_LAYERS = 2

DIM_MODEL = 128
DIM_FFN = 4 * DIM_MODEL

DIM_HEAD = DIM_MODEL // NUM_HEADS
# DIM_KEY = DIM_MODEL // NUM_HEADS
# DIM_VALUE = DIM_MODEL // NUM_HEADS

DROPOUT = .1

MASK_VALUE = -1e9
LINEAR_BIAS = False
FLASH_ATTENTION = False # TODO: Not implemented

# ------------- TRAIN -------------

BATCH_SIZE = 128
PRETRAINING_VAL_RATIO = 1e-3

MAX_LEARNING_RATE = 6e-4
MIN_LEARNING_RATE = 6e-5
WARMUP_ITERS = 2_000

WEIGHT_DECAY = .1
DECAY_ITERS = 100_000

BETA_1 = .9
BETA_2 = .95

EPSILON = 1e-8

VALIDATION_STEP = 50

class Settings(BaseModel):
    # Device to use
    device: str = DEVICE_NAME
    distributed: str = "none"  # Options: "none", "ddp", "deepspeed", "fsdp"
    rank: int = 0
    world_size: int = 1

    # Tokenizer parameters
    token_split_pattern: str = TOKEN_SPLIT_PATTERN
    special_tokens: SpecialTokens = SPECIAL_TOKENS
    forced_tokens: List[str] = FORCED_TOKENS
    vocab_file: Path = Path(VOCAB_FILE)
    tokenizer_dir: Path = Path("tokenizer")
    tokenizer_name: str = "bpe"  # Options: "bpe", "tiktoken"
    tokenizer_type: str = "simple"  # Options: "simple", "rust", "tokenizers", "tiktoken"
    
    # Data parameters
    vocab_size: int = VOCAB_SIZE
    data_folder: str = str(DATA_FOLDER)
    min_document_size: int = MIN_DOCUMENT_SIZE
    output_folder: str = Path(OUTPUT_FOLDER)
    max_token_length: int = MAX_TOKEN_LENGTH
    
    # Transformer parameters
    num_heads: int = NUM_HEADS
    num_layers: int = NUM_LAYERS
    dim_model: int = DIM_MODEL
    dim_ffn: int = DIM_FFN
    dim_head: int = DIM_HEAD
    dropout: float = DROPOUT
    flash_attention: bool = FLASH_ATTENTION
    model_folder: str = Path(MODEL_FOLDER)

    # Optimizer parameters
    optimizer: str = "AdamW"
    beta_1: float = BETA_1
    beta_2: float = BETA_2
    epsilon: float = EPSILON
    max_learning_rate: float = MAX_LEARNING_RATE
    min_learning_rate: float = MIN_LEARNING_RATE
    warmup_iters: int = WARMUP_ITERS
    weight_decay: float = WEIGHT_DECAY

    # Training parameters
    batch_size: int = BATCH_SIZE
    validation_step: int = VALIDATION_STEP
    epochs: int = 10
    save_on_drive: bool = SAVE_ON_DRIVE
    drive_file: str = DRIVE_FILE
    save_on_wandb: bool = SAVE_ON_WANDB
    pretraining_val_ratio: float = PRETRAINING_VAL_RATIO
    num_threads: int = NUM_THREADS
    mask_value: float = MASK_VALUE
    linear_bias: bool = LINEAR_BIAS
    max_context: int = MAX_CONTEXT

    # Inference parameters
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 64


    def get(self, key: str, default=None):
        return getattr(self, key) if hasattr(self, key) else default
    