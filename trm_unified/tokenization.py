import hashlib
from typing import List, Sequence, Union

import torch
from transformers import AutoTokenizer


def _stable_token_id(token: str, vocab_size: int) -> int:
    h = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return 2 + (int(h[:8], 16) % max(1, vocab_size - 2))


class LocalHashTokenizer:
    def __init__(self, vocab_size: int = 32768):
        self.vocab_size = int(vocab_size)
        self.pad_token_id = 0
        self.unk_token_id = 1

    def _encode_one(self, text: str, max_length: int) -> List[int]:
        toks = (text or "").strip().split()
        if not toks:
            return [self.unk_token_id]
        ids = [_stable_token_id(t, self.vocab_size) for t in toks]
        return ids[:max_length] if max_length > 0 else ids

    def __call__(
        self,
        texts: Union[str, Sequence[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
    ):
        if isinstance(texts, str):
            rows = [texts]
        else:
            rows = list(texts)
        ids = [self._encode_one(x, max_length if truncation else 0) for x in rows]
        mlen = max(len(x) for x in ids) if padding else 0
        if max_length > 0 and truncation:
            mlen = min(max_length, mlen) if mlen else max_length

        out_ids = []
        out_mask = []
        for r in ids:
            rr = r[:mlen] if (truncation and mlen > 0) else r
            pad_n = max(0, mlen - len(rr)) if padding else 0
            out_ids.append(rr + [self.pad_token_id] * pad_n)
            out_mask.append([1] * len(rr) + [0] * pad_n)

        if return_tensors != "pt":
            raise ValueError("LocalHashTokenizer currently supports return_tensors='pt' only")
        return {
            "input_ids": torch.tensor(out_ids, dtype=torch.long),
            "attention_mask": torch.tensor(out_mask, dtype=torch.long),
        }


def load_tokenizer(name: str):
    n = (name or "").strip().lower()
    if n in {"local-hash", "local-simple", "local"}:
        return LocalHashTokenizer()
    try:
        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception:
        print(f"[warn] failed to load tokenizer '{name}', falling back to local-hash tokenizer")
        return LocalHashTokenizer()
