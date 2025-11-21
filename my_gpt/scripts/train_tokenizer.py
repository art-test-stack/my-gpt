if not __name__ == "__main__":
    raise ImportError("This script is intended to be run as a standalone program.")

from my_gpt.tokenizer.bpe import bpe
from my_gpt.tokenizer.config import TrainingTokenizerConfig
import argparse, json
from pathlib import Path
from datasets import load_dataset
from my_gpt.tokenizer.write_corpus import write_corpus_sample
from typing import Union, Generator
import time

parser = argparse.ArgumentParser(description="Train BPE tokenizer on a given corpus or evaluate BPE tokenizer against baselines.")
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train or evaluate on (default: 10B)')
parser.add_argument("--chars_per_doc", type=int, default=10_000, help="Maximum number of characters per document to use from the corpus for training or evaluation (default: 10,000).")
parser.add_argument("--vocab_size", type=int, default=32_000, help="Vocabulary size for BPE tokenizer (default: 30,000).")
parser.add_argument("--name", type=str, default="yc1_tokenizer", help="Name of the tokenizer (default: 'yc1_tokenizer').")
parser.add_argument("--corpus_path", type=str, default="./.data/corpus.txt", help="Path to the corpus file (default: './.data/corpus.txt').")
parser.add_argument("--write_corpus", action="store_true", help="Flag to indicate training mode.")
# parser.add_argument("--nb_special_tokens", type=int, default=16, help="Number of special tokens to reserve in the tokenizer vocabulary (default: 16).")
args = parser.parse_args()

config = TrainingTokenizerConfig(
    max_chars=args.max_chars,
    chars_per_doc=args.chars_per_doc,
    vocab_size=args.vocab_size,
    name=args.name,
)

corpus_path = Path(args.corpus_path)
if args.write_corpus:
    write_corpus_sample(chars_per_doc=args.chars_per_doc, max_chars=args.max_chars, out_path=corpus_path)

def corpus_iter(path: Union[str, Path] = corpus_path) -> Generator[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line.strip()

print(f"Loaded corpus from {corpus_path}")
t0 = time.time()
merges, vocab = bpe(corpus_iter, corpus_path, config)
t1 = time.time()
print(f"Training took {t1 - t0:.2f} seconds")

with open(config.tokenizer_dir / "merges.json", "w", encoding="utf-8") as mf:
    json.dump(merges, mf, ensure_ascii=False, indent=2)

with open(config.tokenizer_dir / "vocab.json", "w", encoding="utf-8") as vf:
    json.dump(list(vocab), vf, ensure_ascii=False, indent=2)

print("Trained BPE tokenizer on provided corpus.")
print("Merges", merges)
print("Vocab size", len(vocab))
