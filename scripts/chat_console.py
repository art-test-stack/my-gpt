from gpt_lib.model.model import GPTModel
from gpt_lib.utils.default import (
    ModelConfig,
    TransformerOutput,
    ModelCompletionOutput,
)

from transformers import AutoModelForCausalLM
import torch
# model = AutoModelForCausalLM.from_pretrained("mistralai/Ministral-8B-Instruct-2410", torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")