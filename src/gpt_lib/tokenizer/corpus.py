from pathlib import Path
import random, pickle
from gpt_lib.utils.default import RANDOM_SEED, CACHE_DIR
from gpt_lib.data.loader import load_datasets
from gpt_lib.data.normalizers import clean_codeparrot_example

from tqdm import tqdm

class Corpus:
    def __init__(
            self, 
            name: str,
            total_chars: int, 
            total_docs: int,
            random_seed: int = RANDOM_SEED,
            sources: dict = {},
        ):
        self.name = name
        self.random_seed = random_seed
        self.total_chars = total_chars
        self.total_docs = total_docs
        self.sources = sources 
        self.out_path = CACHE_DIR / f"{self.name}_corpus.txt"
    
    def save(self, out_path: Path = None):
        path = out_path or (CACHE_DIR / f"{self.name}_corpus_meta.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def to_csv(self, out_path: Path = None):
        path = out_path or (CACHE_DIR / f"{self.name}_corpus_meta.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("name,random_seed,total_chars,total_docs,sources\n")
            f.write(f"{self.name},{self.random_seed},{self.total_chars},{self.total_docs},\"{self.sources}\"\n")

    @staticmethod
    def from_path(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def from_name(name: str):
        path = CACHE_DIR / f"{name}_corpus_meta.pkl"
        return Corpus.from_path(path)
    
    @staticmethod
    def write_from_sources(
            name: str,
            sources: dict,
            chars_per_doc: int = 10_000,
            max_chars: int = 1_000_000_000,
        ):
        out_path = CACHE_DIR / f"{name}_corpus.txt"
        char_count, doc_count = write_corpus_sample(
            sources=sources,
            chars_per_doc=chars_per_doc,
            max_chars=max_chars,
            out_path=out_path,
        )
        meta = Corpus(
            name=name,
            total_chars=char_count,
            total_docs=doc_count,
            sources=sources,
        )
        meta.save()
        return meta
    

def tokenizer_data_loader(config):
    "Method based on local downloaded parquet files"
    cache_dir = config.get("data_dir", CACHE_DIR )

    if isinstance(config.get("tokenizer_name"), str):
        cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        raise AssertionError
    
    ds_names = config.ds_names

    for ds_name in ds_names:
        ds_path = cache_dir / ds_name
        if not ds_path.exists():
            raise AssertionError(f"No such dataset directory: {ds_path}")
        
        for file in ds_path.iterdir():
            yield file


def write_corpus_sample(
        sources = None, # dict ds_name: weight
        chars_per_doc=10_000,
        max_chars=1_000_000_000,
        per_dataset_normalizer=None,
        out_path: Path = Path(".data/corpus.txt"),
        split: str = "train"
    ):

    if not sources:
        sources = [
            { "path": "HuggingFaceFW/fineweb-edu", "weight": 0.6 },
            { "path": "HuggingFaceTB/finemath", "weight": 0.25, "name": "finemath-4plus" },
            { "path": "codeparrot/codeparrot-clean", "weight": 0.15 },
        ]
    ds = load_datasets(sources, split=split)
    
    r = random.Random(RANDOM_SEED)
    
    char_count = 0
    doc_count = 0
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
                if not text.strip():
                    continue
                doc_count += 1
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
    return char_count, doc_count
    print("Wrote", char_count, "characters to", out_path)