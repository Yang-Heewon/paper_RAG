import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm


_PP_CFG = {}


def _safe_warn_text(raw) -> str:
    msg = str(raw)
    enc = sys.stdout.encoding or "utf-8"
    return msg.encode(enc, errors="backslashreplace").decode(enc, errors="ignore")


def iter_json_records(path: str) -> Iterable[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        if not first:
            return
        f.seek(0)
        if first == '[':
            arr = json.load(f)
            for ex in arr:
                yield ex
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def load_kb_map(entities_txt: str) -> Dict[str, int]:
    kb2idx = {}
    with open(entities_txt, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.rstrip('\n').split('\t')
            if parts:
                kb2idx[parts[0]] = i
    return kb2idx


def load_rel_map(relations_txt: str) -> Dict[str, int]:
    rel2idx = {}
    with open(relations_txt, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            rel = line.strip()
            if rel:
                rel2idx[rel] = i
    return rel2idx


def build_adj_from_tuples(tuples, rel2idx=None):
    adj = defaultdict(list)
    for s, r, o in tuples:
        try:
            s_id = int(s)
            o_id = int(o)
        except Exception:
            continue
        if isinstance(r, int):
            r_id = int(r)
        elif isinstance(r, str) and rel2idx is not None:
            if r not in rel2idx:
                continue
            r_id = rel2idx[r]
        else:
            continue
        adj[s_id].append((r_id, o_id))
    return adj


def mine_valid_paths(
    tuples,
    start_entities: List[int],
    goal_entities: List[int],
    max_steps: int = 4,
    max_paths: int = 4,
    max_neighbors: int = 128,
):
    if not tuples or not start_entities or not goal_entities:
        return []

    goal_set = set(int(x) for x in goal_entities)
    adj = build_adj_from_tuples(tuples)

    found = []
    for s in start_entities:
        q = deque()
        q.append((int(s), [int(s)], []))

        while q and len(found) < max_paths:
            cur, visited, triples = q.popleft()
            if len(triples) >= max_steps:
                continue
            cand = adj.get(cur, [])
            if max_neighbors > 0 and len(cand) > max_neighbors:
                cand = cand[:max_neighbors]
            for r, nxt in cand:
                if nxt in visited:
                    continue
                new_triples = triples + [[cur, int(r), int(nxt)]]
                if nxt in goal_set:
                    found.append(new_triples)
                    if len(found) >= max_paths:
                        break
                q.append((int(nxt), visited + [int(nxt)], new_triples))
    return found


def _apply_path_policy(valid_paths, path_policy: str = "all", shortest_k: int = 1):
    paths = [p for p in valid_paths if isinstance(p, list) and p]
    if not paths:
        return []

    policy = str(path_policy or "all").strip().lower()
    if policy in {"all", "keep_all"}:
        return paths

    if policy in {"shortest_only", "shortest"}:
        min_len = min(len(p) for p in paths)
        for p in paths:
            if len(p) == min_len:
                return [p]
        return [paths[0]]

    if policy in {"shortest_k", "k_shortest"}:
        k = max(1, int(shortest_k))
        # Stable tiebreak by original order.
        ranked = sorted(enumerate(paths), key=lambda x: (len(x[1]), x[0]))
        return [p for _, p in ranked[:k]]

    return paths


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_line_offsets(jsonl_path: str, is_main: bool = True) -> List[int]:
    offsets = []
    with open(jsonl_path, "rb") as f:
        off = f.tell()
        line = f.readline()
        while line:
            offsets.append(off)
            off = f.tell()
            line = f.readline()
    if is_main:
        print(f"[ok] Indexed {len(offsets)} lines from {jsonl_path}")
    return offsets


def read_jsonl_by_offset(jsonl_path: str, offsets: List[int], idx: int) -> dict:
    with open(jsonl_path, "rb") as f:
        f.seek(offsets[idx])
        line = f.readline()
    return json.loads(line.decode("utf-8"))


def _normalize_cwq(
    ex: dict,
    kb2idx: Dict[str, int],
    max_steps: int,
    max_paths: int,
    max_neighbors: int,
    mine_paths: bool = True,
    path_policy: str = "all",
    shortest_k: int = 1,
) -> dict:
    entities = [int(x) for x in ex.get('entities', []) if isinstance(x, int)]
    answers = ex.get('answers', [])
    answers_cid = [int(x) for x in ex.get('answers_cid', []) if isinstance(x, int)]
    tuples = ex.get('subgraph', {}).get('tuples', [])
    valid_paths = ex.get('valid_paths', [])

    # Fallback for CWQ raws shaped like WebQSP: derive answers_cid/valid_paths.
    if (not answers_cid) and isinstance(answers, list):
        for a in answers:
            kb_id = a.get('kb_id') if isinstance(a, dict) else None
            if kb_id in kb2idx:
                answers_cid.append(int(kb2idx[kb_id]))
    if mine_paths and (not valid_paths) and tuples and entities and answers_cid:
        valid_paths = mine_valid_paths(
            tuples=tuples,
            start_entities=entities,
            goal_entities=answers_cid,
            max_steps=max_steps,
            max_paths=max_paths,
            max_neighbors=max_neighbors,
        )
    if not mine_paths:
        valid_paths = []

    out = {
        'orig_id': ex.get('orig_id', ex.get('id', '')),
        'question': ex.get('question', ''),
        'entities': entities,
        'answers_cid': answers_cid,
        'answers': answers,
        'subgraph': {'tuples': tuples},
        'valid_paths': valid_paths,
        'relation_paths': [],
    }
    if max_steps > 0 and out['valid_paths']:
        out['valid_paths'] = [p[:max_steps] for p in out['valid_paths']]
    if mine_paths and out['valid_paths']:
        out['valid_paths'] = _apply_path_policy(
            out['valid_paths'],
            path_policy=path_policy,
            shortest_k=shortest_k,
        )
    out['relation_paths'] = [
        [int(step[1]) for step in p if isinstance(step, list) and len(step) == 3]
        for p in out['valid_paths']
    ]
    return out


def _normalize_webqsp(
    ex: dict,
    kb2idx: Dict[str, int],
    max_steps: int,
    max_paths: int,
    max_neighbors: int,
    mine_paths: bool = True,
    path_policy: str = "all",
    shortest_k: int = 1,
) -> dict:
    answers = ex.get('answers', [])
    answers_cid = []
    for a in answers:
        kb_id = a.get('kb_id') if isinstance(a, dict) else None
        if kb_id in kb2idx:
            answers_cid.append(int(kb2idx[kb_id]))

    entities = [int(x) for x in ex.get('entities', []) if isinstance(x, int)]
    tuples = ex.get('subgraph', {}).get('tuples', [])
    valid_paths = []
    if mine_paths:
        valid_paths = mine_valid_paths(
            tuples=tuples,
            start_entities=entities,
            goal_entities=answers_cid,
            max_steps=max_steps,
            max_paths=max_paths,
            max_neighbors=max_neighbors,
        )

    if max_steps > 0 and valid_paths:
        valid_paths = [p[:max_steps] for p in valid_paths]
    if mine_paths and valid_paths:
        valid_paths = _apply_path_policy(
            valid_paths,
            path_policy=path_policy,
            shortest_k=shortest_k,
        )
    relation_paths = [[int(step[1]) for step in p if isinstance(step, list) and len(step) == 3] for p in valid_paths]

    return {
        'orig_id': ex.get('id', ''),
        'question': ex.get('question', ''),
        'entities': entities,
        'answers_cid': answers_cid,
        'answers': answers,
        'subgraph': {'tuples': tuples},
        'valid_paths': valid_paths,
        'relation_paths': relation_paths,
    }


def preprocess_split(
    dataset: str,
    input_path: str,
    output_path: str,
    entities_txt: str,
    max_steps: int = 4,
    max_paths: int = 4,
    max_neighbors: int = 128,
    mine_paths: bool = True,
    require_valid_paths: bool = True,
    preprocess_workers: int = 1,
    path_policy: str = "all",
    shortest_k: int = 1,
    progress_desc: str = "preprocess",
):
    dataset = dataset.lower()
    kb2idx = load_kb_map(entities_txt)

    total_hint = _estimate_total_records(input_path)
    kept = 0
    total = 0
    workers = max(1, int(preprocess_workers))

    with open(output_path, 'w', encoding='utf-8') as w, tqdm(total=total_hint, desc=progress_desc, unit='ex') as pbar:
        if workers == 1:
            for ex in iter_json_records(input_path):
                total += 1
                obj = _normalize_one(
                    ex, dataset, kb2idx, max_steps, max_paths, max_neighbors, mine_paths,
                    path_policy=path_policy, shortest_k=shortest_k
                )
                if require_valid_paths and not obj['valid_paths']:
                    pbar.update(1)
                    continue
                w.write(json.dumps(obj, ensure_ascii=False) + '\n')
                kept += 1
                pbar.update(1)
        else:
            try:
                with mp.Pool(
                    processes=workers,
                    initializer=_init_preprocess_worker,
                    initargs=(dataset, kb2idx, max_steps, max_paths, max_neighbors, mine_paths, path_policy, shortest_k),
                ) as pool:
                    for obj in pool.imap(_normalize_one_mp, iter_json_records(input_path), chunksize=32):
                        total += 1
                        if require_valid_paths and not obj['valid_paths']:
                            pbar.update(1)
                            continue
                        w.write(json.dumps(obj, ensure_ascii=False) + '\n')
                        kept += 1
                        pbar.update(1)
            except (PermissionError, OSError) as e:
                print(f"[warn] multiprocessing disabled ({_safe_warn_text(e)}); fallback to single worker.")
                for ex in iter_json_records(input_path):
                    total += 1
                    obj = _normalize_one(
                        ex, dataset, kb2idx, max_steps, max_paths, max_neighbors, mine_paths,
                        path_policy=path_policy, shortest_k=shortest_k
                    )
                    if require_valid_paths and not obj['valid_paths']:
                        pbar.update(1)
                        continue
                    w.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    kept += 1
                    pbar.update(1)

    return {'total': total, 'kept': kept, 'output': output_path}


def _normalize_one(ex, dataset, kb2idx, max_steps, max_paths, max_neighbors, mine_paths, path_policy="all", shortest_k=1):
    if dataset == 'cwq':
        return _normalize_cwq(
            ex, kb2idx, max_steps, max_paths, max_neighbors,
            mine_paths=mine_paths, path_policy=path_policy, shortest_k=shortest_k
        )
    if dataset == 'webqsp':
        return _normalize_webqsp(
            ex, kb2idx, max_steps, max_paths, max_neighbors,
            mine_paths=mine_paths, path_policy=path_policy, shortest_k=shortest_k
        )
    raise ValueError(f'Unsupported dataset: {dataset}')


def _init_preprocess_worker(dataset, kb2idx, max_steps, max_paths, max_neighbors, mine_paths, path_policy, shortest_k):
    global _PP_CFG
    _PP_CFG = {
        'dataset': dataset,
        'kb2idx': kb2idx,
        'max_steps': max_steps,
        'max_paths': max_paths,
        'max_neighbors': max_neighbors,
        'mine_paths': mine_paths,
        'path_policy': path_policy,
        'shortest_k': shortest_k,
    }


def _normalize_one_mp(ex):
    return _normalize_one(
        ex=ex,
        dataset=_PP_CFG['dataset'],
        kb2idx=_PP_CFG['kb2idx'],
        max_steps=_PP_CFG['max_steps'],
        max_paths=_PP_CFG['max_paths'],
        max_neighbors=_PP_CFG['max_neighbors'],
        mine_paths=_PP_CFG['mine_paths'],
        path_policy=_PP_CFG['path_policy'],
        shortest_k=_PP_CFG['shortest_k'],
    )


def _estimate_total_records(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
    if first == '[':
        return None
    n = 0
    with open(path, 'rb') as f:
        for _ in f:
            n += 1
    return n
