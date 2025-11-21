from collections import Counter
from my_gpt.tokenizer.pretokenizer import SimplePreTokenizer

from typing import Dict, Tuple, Union, Generator
from pathlib import Path
from tqdm import tqdm


def pretokenize_stream(corpus_iter: Generator[str, None, None], pretok) -> Generator[list[bytes], None, None]:
    for line in corpus_iter:
        tokens = pretok._split(line, drop_special_tokens=True)
        for t in tokens:
            yield t.encode("utf-8")


def bpe(
        corpus_iter_fn,      
        corpus_path: Union[str, Path],
        config
    ) -> Tuple[Dict[bytes, bytes], Dict[bytes, int]]:
    """
    Byte-Level BPE training implementation. Not fast.
    """

    pretknzr = SimplePreTokenizer(config)

    vocab = Counter()
    pair_counts = Counter()

    print("First streaming pass (initial vocab + pair counts)")

    for token_bytes in tqdm(pretokenize_stream(corpus_iter_fn(corpus_path), pretknzr), desc="Initial pass"):
        symbols = [bytes([b]) for b in token_bytes]

        vocab.update(symbols)
        pair_counts.update(zip(symbols, symbols[1:]))

    initial_vocab_size = len(vocab)
    nb_merges = config.vocab_size - initial_vocab_size

    print(f"Initial vocab symbols: {initial_vocab_size}")
    print(f"Will perform ~{nb_merges} merges")

    merges: Dict[bytes, bytes] = {}

    for merge_rank in tqdm(range(nb_merges), desc="BPE Merges"):
        if not pair_counts:
            break

        (A, B), _ = pair_counts.most_common(1)[0]
        merged_symbol = A + B
        merges[merged_symbol] = merged_symbol 
        vocab[merged_symbol] = 1
        new_pair_counts = Counter()

        for token_bytes in pretokenize_stream(corpus_iter_fn(corpus_path), pretknzr):
            A_len = len(A)
            B_len = len(B)

            syms = []
            i = 0
            L = len(token_bytes)

            while i < L:
                if i + A_len + B_len <= L and token_bytes[i:i+A_len] == A and token_bytes[i+A_len:i+A_len+B_len] == B:
                    syms.append(merged_symbol)
                    i += A_len + B_len
                else:
                    syms.append(bytes(token_bytes[i:i+1]))
                    i += 1

            new_pair_counts.update(zip(syms, syms[1:]))

        pair_counts = new_pair_counts

    vocab_list = sorted(vocab.keys()) 
    vocab_dict = {sym: i for i, sym in enumerate(vocab_list)}

    offset = len(vocab_dict)
    for i, special in enumerate(config.special_tokens.list()):
        vocab_dict[special.encode("utf-8")] = offset + i

    print(f"Final vocab size: {len(vocab_dict)}")
    return merges, vocab_dict