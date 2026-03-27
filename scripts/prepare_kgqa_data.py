#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm


def _snake_case_name(raw: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(raw or ""))
    return s.replace("__", "_").strip().lower()


def _candidate_hf_cache_roots(cache_dir: str = "") -> List[Path]:
    roots: List[Path] = []

    def _push(path_str: str):
        if not path_str:
            return
        p = Path(os.path.expanduser(os.path.expandvars(path_str))).resolve()
        if p not in roots:
            roots.append(p)

    if cache_dir:
        _push(cache_dir)
        _push(str(Path(cache_dir) / "datasets"))
    _push(os.environ.get("HF_DATASETS_CACHE", ""))
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        _push(str(Path(hf_home) / "datasets"))
    _push(str(Path.home() / ".cache" / "huggingface" / "datasets"))
    return roots


def _find_cached_arrow_dataset_dir(hf_name: str, cache_dir: str = "") -> Path:
    repo_name = str(hf_name or "").split("/")[-1]
    snake_repo = _snake_case_name(repo_name)
    tokens = {repo_name.lower(), snake_repo}

    candidates: List[Tuple[float, Path]] = []
    for root in _candidate_hf_cache_roots(cache_dir):
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            child_name = child.name.lower()
            if not any(tok and tok in child_name for tok in tokens):
                continue
            for info_path in child.glob("*/*/*/dataset_info.json"):
                ds_dir = info_path.parent
                if list(ds_dir.glob("*-train-*.arrow")):
                    candidates.append((ds_dir.stat().st_mtime, ds_dir))

    if not candidates:
        raise FileNotFoundError(f"no cached arrow dataset directory found for {hf_name}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _load_dataset_from_cached_arrow(hf_name: str, cache_dir: str = ""):
    from datasets import Dataset, DatasetDict, concatenate_datasets

    ds_dir = _find_cached_arrow_dataset_dir(hf_name, cache_dir=cache_dir)
    split_globs = {
        "train": "*-train-*.arrow",
        "validation": "*-validation-*.arrow",
        "test": "*-test-*.arrow",
    }
    out = {}
    for split_name, pattern in split_globs.items():
        shard_paths = sorted(ds_dir.glob(pattern))
        if not shard_paths:
            continue
        shards = [Dataset.from_file(str(p)) for p in shard_paths]
        out[split_name] = concatenate_datasets(shards) if len(shards) > 1 else shards[0]
    if not out:
        raise FileNotFoundError(f"cached arrow shards missing under {ds_dir}")
    print(f"[load] {hf_name} (cached arrow: {ds_dir})")
    return DatasetDict(out)


def _norm_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def _as_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _parse_graph_rows(graph_val) -> List[Tuple[str, str, str]]:
    # Hugging Face viewer shows graph as sequence of [head, relation, tail].
    raw = graph_val
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                raw = json.loads(s)
            except Exception:
                raw = []
        else:
            raw = []

    triples = []
    for row in _as_list(raw):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        h = _norm_text(row[0])
        r = _norm_text(row[1])
        t = _norm_text(row[2])
        if not h or not r or not t:
            continue
        triples.append((h, r, t))
    return triples


class Vocab:
    def __init__(self):
        self._idx: Dict[str, int] = {}

    def add(self, item: str):
        s = _norm_text(item)
        if not s:
            return None
        if s not in self._idx:
            self._idx[s] = len(self._idx)
        return self._idx[s]

    def get(self, item: str):
        s = _norm_text(item)
        if not s:
            return None
        return self._idx.get(s)

    def items_in_order(self) -> List[str]:
        # insertion order is deterministic in Python 3.7+
        return list(self._idx.keys())

    def __len__(self):
        return len(self._idx)


def _iter_examples(ds_dict) -> Iterable[dict]:
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds_dict:
            continue
        for ex in ds_dict[split_name]:
            yield ex


def _build_vocab(ds_dict) -> Tuple[Vocab, Vocab]:
    ent_vocab = Vocab()
    rel_vocab = Vocab()
    total = 0
    for ex in tqdm(_iter_examples(ds_dict), desc="build_vocab", unit="ex"):
        total += 1
        for h, r, t in _parse_graph_rows(ex.get("graph")):
            ent_vocab.add(h)
            rel_vocab.add(r)
            ent_vocab.add(t)
        for x in _as_list(ex.get("q_entity")):
            ent_vocab.add(_norm_text(x))
        for x in _as_list(ex.get("a_entity")):
            ent_vocab.add(_norm_text(x))
        for x in _as_list(ex.get("answer")):
            ent_vocab.add(_norm_text(x))
        for x in _as_list(ex.get("choices")):
            # choices can be strings or nested structures.
            if isinstance(x, (list, tuple)):
                for y in x:
                    ent_vocab.add(_norm_text(y))
            else:
                ent_vocab.add(_norm_text(x))
    print(f"[vocab] examples={total} entities={len(ent_vocab)} relations={len(rel_vocab)}")
    return ent_vocab, rel_vocab


def _unique_nonempty(items: Sequence[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        s = _norm_text(x)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _convert_example(ex: dict, ent_vocab: Vocab, rel_vocab: Vocab) -> dict:
    tuples = []
    for h, r, t in _parse_graph_rows(ex.get("graph")):
        h_id = ent_vocab.get(h)
        r_id = rel_vocab.get(r)
        t_id = ent_vocab.get(t)
        if h_id is None or r_id is None or t_id is None:
            continue
        tuples.append([int(h_id), int(r_id), int(t_id)])

    q_entities_text = _unique_nonempty(_as_list(ex.get("q_entity")))
    a_entities_text = _unique_nonempty(_as_list(ex.get("a_entity")))
    answer_text = _unique_nonempty(_as_list(ex.get("answer")))
    if not a_entities_text:
        a_entities_text = answer_text

    starts = []
    seen_start = set()
    for q in q_entities_text:
        qid = ent_vocab.get(q)
        if qid is None or qid in seen_start:
            continue
        seen_start.add(qid)
        starts.append(int(qid))

    answers_cid = []
    seen_ans = set()
    for a in a_entities_text:
        aid = ent_vocab.get(a)
        if aid is None or aid in seen_ans:
            continue
        seen_ans.add(aid)
        answers_cid.append(int(aid))

    answers = [{"kb_id": a} for a in a_entities_text]

    return {
        "orig_id": _norm_text(ex.get("id")),
        "question": _norm_text(ex.get("question")),
        "entities": starts,
        "answers_cid": answers_cid,
        "answers": answers,
        "subgraph": {"tuples": tuples},
        "valid_paths": [],
        "relation_paths": [],
        "q_entity_text": q_entities_text,
        "a_entity_text": a_entities_text,
    }


def _write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as w:
        for obj in rows:
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def _write_list(path: str, values: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        for v in values:
            w.write(f"{v}\n")


def _convert_dataset(
    ds_name: str,
    hf_name: str,
    out_root: str,
    cache_dir: str = "",
    cwq_vocab_only: bool = False,
):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "huggingface datasets package is required. Install with: pip install datasets"
        ) from e

    ds_dict = None
    cache_err = None
    try:
        ds_dict = _load_dataset_from_cached_arrow(hf_name, cache_dir=cache_dir)
        required_splits = {"train", "validation", "test"}
        cached_splits = set(ds_dict.keys())
        missing_splits = sorted(required_splits - cached_splits)
        if missing_splits:
            cache_err = RuntimeError(
                f"cached arrow missing required splits: {', '.join(missing_splits)}"
            )
            print(
                f"[load] {hf_name} cached arrow incomplete; "
                f"will retry remote download for missing splits: {', '.join(missing_splits)}"
            )
            ds_dict = None
    except Exception as e:
        cache_err = e

    if ds_dict is None:
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        print(f"[load] {hf_name}")
        try:
            ds_dict = load_dataset(hf_name, **kwargs)
        except Exception as e:
            cache_msg = ""
            if cache_err is not None:
                cache_msg = f" cached-arrow fallback also failed: {cache_err}."
            raise RuntimeError(
                f"failed to load Hugging Face dataset '{hf_name}'."
                f"{cache_msg} "
                "Check internet access / HF auth / dataset name, "
                "or pre-download into local cache and set HF_CACHE_DIR."
            ) from e

    ent_vocab, rel_vocab = _build_vocab(ds_dict)

    split_map = {
        "train": "train",
        "validation": "dev",
        "test": "test",
    }

    counts = {}
    skip_split_conversion = ds_name == "cwq" and cwq_vocab_only
    if not skip_split_conversion:
        for hf_split, out_split in split_map.items():
            if hf_split not in ds_dict:
                continue
            rows = (
                _convert_example(ex, ent_vocab, rel_vocab)
                for ex in tqdm(ds_dict[hf_split], desc=f"convert:{ds_name}:{out_split}", unit="ex")
            )

            if ds_name == "cwq":
                if out_split == "train":
                    out_path = os.path.join(out_root, "data", "CWQ", "train_split.jsonl")
                elif out_split == "dev":
                    out_path = os.path.join(out_root, "data", "CWQ", "dev_split.jsonl")
                else:
                    out_path = os.path.join(out_root, "data", "CWQ", "test_split.jsonl")
            else:
                # Keep legacy file names expected by current config.
                if out_split == "train":
                    out_path = os.path.join(out_root, "data", "webqsp", "train.json")
                elif out_split == "dev":
                    out_path = os.path.join(out_root, "data", "webqsp", "dev.json")
                else:
                    out_path = os.path.join(out_root, "data", "webqsp", "test.json")

            counts[out_split] = _write_jsonl(out_path, rows)
            print(f"[write] {out_path} rows={counts[out_split]}")
    else:
        counts["mode"] = "cwq_vocab_only"
        print("[mode] cwq vocab-only: split conversion skipped")

    if ds_name == "cwq":
        cwq_root = os.path.join(out_root, "data", "CWQ")
        _write_list(os.path.join(cwq_root, "entities.txt"), ent_vocab.items_in_order())
        _write_list(os.path.join(cwq_root, "relations.txt"), rel_vocab.items_in_order())

        if not cwq_vocab_only:
            emb_root = os.path.join(out_root, "data", "CWQ", "embeddings_output", "CWQ", "e5")
            os.makedirs(emb_root, exist_ok=True)
            _write_list(os.path.join(emb_root, "entity_ids.txt"), ent_vocab.items_in_order())
            _write_list(os.path.join(emb_root, "relation_ids.txt"), rel_vocab.items_in_order())

            # Compatibility path used by existing cwq config.
            test_src = os.path.join(out_root, "data", "CWQ", "test_split.jsonl")
            test_dst = os.path.join(out_root, "data", "data", "CWQ", "test.json")
            os.makedirs(os.path.dirname(test_dst), exist_ok=True)
            shutil.copy2(test_src, test_dst)
            counts["test_compat_path"] = test_dst
    else:
        web_root = os.path.join(out_root, "data", "webqsp")
        _write_list(os.path.join(web_root, "entities.txt"), ent_vocab.items_in_order())
        _write_list(os.path.join(web_root, "relations.txt"), rel_vocab.items_in_order())

    meta = {
        "dataset": ds_name,
        "hf_name": hf_name,
        "entities": len(ent_vocab),
        "relations": len(rel_vocab),
        "splits": counts,
    }
    print(f"[done] {ds_name}: {meta}")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Prepare RoG HF datasets for GRAPH-TRAVERSE pipeline")
    ap.add_argument("--dataset", choices=["cwq", "webqsp", "all"], default="all")
    ap.add_argument(
        "--cwq_vocab_only",
        action="store_true",
        help="For CWQ only: write entities.txt and relations.txt, and skip split JSONL conversion.",
    )
    ap.add_argument("--cwq_name", default="rmanluo/RoG-cwq")
    ap.add_argument("--webqsp_name", default="rmanluo/RoG-webqsp")
    ap.add_argument("--out_root", default=".")
    ap.add_argument("--cache_dir", default="")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    if args.cwq_vocab_only and args.dataset != "cwq":
        raise SystemExit("--cwq_vocab_only can be used only with --dataset cwq")

    metas = []
    if args.dataset in {"cwq", "all"}:
        metas.append(
            _convert_dataset(
                "cwq",
                args.cwq_name,
                out_root,
                cache_dir=args.cache_dir,
                cwq_vocab_only=args.cwq_vocab_only,
            )
        )
    if args.dataset in {"webqsp", "all"}:
        metas.append(_convert_dataset("webqsp", args.webqsp_name, out_root, cache_dir=args.cache_dir))

    meta_path = os.path.join(out_root, "data", ".downloads", "rog_hf_prepare_meta.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as w:
        json.dump({"runs": metas}, w, ensure_ascii=False, indent=2)
    print(f"[meta] {meta_path}")


if __name__ == "__main__":
    main()
