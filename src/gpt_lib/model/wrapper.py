import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class HFModelWrapper:
    def __init__(self, model_name="openai-community/gpt2", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def generate(self, text, max_len=50):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_len,
                do_sample=False,  # greedy
                pad_token_id=self.tokenizer.eos_token_id
            )
        pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return pred
