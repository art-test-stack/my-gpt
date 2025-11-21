from pathlib import Path
from datasets import load_dataset
import random
from my_gpt.utils.settings import RANDOM_SEED
from my_gpt.data.loader import load_datasets
from my_gpt.data.normalizers import clean_codeparrot_example

from tqdm import tqdm

def tokenizer_data_loader(config):
    "Method based on local downloaded parquet files"
    cache_dir = config.get("data_dir", "./.data")

    if isinstance(config.get("tokenizer_name"), str):
        cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        raise AssertionError
    
    ds_names = config.ds_names

    for ds_name in ds_names:
        ds_path = cache_dir / ds_name
        if not ds_path.exists():
            raise AssertionError
        
        for file in ds_path.iterdir():
            yield file


def write_corpus_sample(
        sources = None, # dict ds_name: weight
        chars_per_doc=10_000,
        max_chars=1_000_000_000,
        per_dataset_normalizer=None,
        out_path: Path = Path(".data/corpus.txt"),
    ):
    if not sources:
        sources = [
            { "path": "HuggingFaceFW/fineweb-edu", "weight": 0.6 },
            { "path": "HuggingFaceTB/finemath", "weight": 0.25, "name": "finemath-4plus" },
            { "path": "codeparrot/codeparrot-clean", "weight": 0.15 },
        ]
    ds = load_datasets(sources)
    
    r = random.Random(RANDOM_SEED)
    
    char_count = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        it = { name: iter(subset) for name, subset in ds.items() }
        with tqdm(total=max_chars, desc="Writing corpus") as pbar:
            while char_count < max_chars:
                p = r.random()
                try:
                    for src in sources:
                        weight = src.get("weight", 1.0)
                        if p < weight:
                            s = next(it[src["path"]])
                            break
                        else:
                            p -= weight
                except StopIteration:
                    break
                text = s.get("text") or s.get("content") or ""

                if src.get("path") == "codeparrot/codeparrot-clean":
                    text = clean_codeparrot_example(text)

                text = text[-chars_per_doc:]
                if per_dataset_normalizer:
                    text = per_dataset_normalizer(text)
                if not text.strip():
                    continue

                fout.write(text)
                char_count += len(text)
                pbar.update(len(text))
    
    print("Wrote", char_count, "characters to", out_path)