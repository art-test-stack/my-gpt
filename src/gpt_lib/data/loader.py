import os
import random
import shutil
from typing import Iterator, Tuple, Iterable, Dict, Union, Callable

from datasets import load_dataset, Dataset, get_dataset_split_names
from transformers import AutoTokenizer
from torch.utils.data import DataLoader as TorchDataLoader

from gpt_lib.utils.default import Settings

def load_datasets(
        sources: Iterable[Dict[str, Union[str, float, Callable]]], # { "path": str, "subset": str (optional), "weight": float (optional), "hook": Callable (optional) } 
        cache_dir: str = "./.data",
        split: str = "train",
        streaming: bool = True,
        *args, **kwargs
    ) -> Dict[str, Iterable]:
    ds = { 
        ds["path"]: ds.get("hook", lambda x: x)(
            load_dataset(
                ds["path"], 
                name=ds.get("name", None),
                split=split,
                streaming=streaming, 
                cache_dir=cache_dir, 
                *args, **kwargs
            )
        ) for ds in sources
    }
    return ds

def weighted_sample_generator(streams, prng):
    """
    streams: list of (iterable, weight)
    yields items from one of the streams according to weights.
    Designed to keep reading from selected stream until exhausted (streams are long)
    For streaming HF datasets these are effectively infinite for training; we just sample.
    """
    # convert weights to cumulative thresholds
    total = sum(w for _, w in streams)
    cum = []
    acc = 0.0
    for _, w in streams:
        acc += w / total
        cum.append(acc)

    # create iterators
    iterators = [iter(s) for s, _ in streams]
    while True:
        p = prng.random()
        # pick which stream index
        idx = 0
        while p > cum[idx]:
            idx += 1
        try:
            yield next(iterators[idx])
        except StopIteration:
            # If a stream ends, remove it from selection.
            # For HF streaming this is unlikely; but handle gracefully.
            iterators.pop(idx)
            streams.pop(idx)
            cum = []
            total = sum(w for _, w in streams) if streams else 0
            acc = 0.0
            for _, w in streams:
                acc += w / total
                cum.append(acc)
            if not streams:
                break
    
class DataLoader:
    """
    Streaming shard-based DataLoader for large-scale text pretraining.

    At each iteration, this class:
      1. Downloads or streams a random shard of the dataset
      2. Tokenizes and caches it locally
      3. Yields a ready-to-train PyTorch DataLoader
      4. Deletes the shard after training to free space
    """

    def __init__(
            self, 
            tokenizer,
            config: Settings = Settings()
        ):
        self.name = config.get("dataset_name", "HuggingFaceFW/fineweb-edu")
        # self.name = config.get("dataset_name", "karpathy/fineweb-edu-100b-shuffle")
        self.seed = config.get("seed", 42)
        self.shard_size = config.get("shard_size", 10_000)
        self.num_shards = config.get("num_shards", 100)
        self.max_length = config.get("max_length", 512)
        self.batch_size = config.get("batch_size", 8)
        self.local_dir = config.get("local_dir", "./fineweb_shards")
        self.tokenizer_name = config.get("tokenizer_name", "gpt2")
        self.stream_mode = config.get("stream_mode", False)
        self.num_proc = config.get("num_proc", 4)

        os.makedirs(self.local_dir, exist_ok=True)

        self.tokenizer = self.tokenizer or AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.rng = random.Random(self.seed)

        try:
            self.dataset_splits = get_dataset_split_names(self.name)
        except Exception:
            self.dataset_splits = ["train"]

        print(f"✅ Initialized DataLoader for dataset: {self.name}")
        print(f"   Available splits: {self.dataset_splits}")

    def _load_random_shard(self) -> Dataset:
        """Load a random subset (or stream) from the dataset."""
        # if not self.stream_mode:
        #     ds = load_dataset(self.name, split=self.dataset_splits[0])
        #     indices = self.rng.sample(range(len(ds)), min(self.shard_size, len(ds)))
        #     return ds.select(indices)

        try:
            ds_stream = load_dataset(self.name, split=self.dataset_splits[0], streaming=True, filters=[("language_score", ">=", 0.99)])
        except Exception:
            ds_stream = load_dataset(self.name, split=self.dataset_splits[0], streaming=True)
        
        return ds_stream
        # samples = []
        # for i, ex in enumerate(ds_stream):
        #     samples.append(ex)
        #     if i >= self.shard_size:
        #         break
        # return Dataset.from_list(samples)

    def _tokenize_shard(self, ds: Dataset) -> Dataset:
        """Tokenize dataset using configured tokenizer."""

        def tokenize_fn(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        ds = ds.map(
            tokenize_fn,
            batched=True,
            # num_proc=self.num_proc,
            remove_columns=[col for col in ds.column_names if col == "text"],
        )
        return ds

    def stream(self) -> Iterator[Tuple[TorchDataLoader, int]]:
        """
        Generator yielding (dataloader, shard_id) pairs.
        Each shard is deleted after use to conserve disk space.
        """

        for shard_id in range(self.num_shards):
            print(f"\n🧩 Preparing shard {shard_id}/{self.num_shards}...")

            ds = self._load_random_shard()
            ds = self._tokenize_shard(ds)

            shard_path = os.path.join(self.local_dir, f"shard_{shard_id}")
            # ds.save_to_disk(shard_path)

            # ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
            dataloader = TorchDataLoader(ds, batch_size=self.batch_size, shuffle=True)

            print(f"✅ Shard {shard_id} ready ({len(ds)} samples). Yielding...")
            yield dataloader, shard_id

            print(f"🧹 Cleaning up shard {shard_id}...")
            shutil.rmtree(shard_path, ignore_errors=True)
            del ds, dataloader
