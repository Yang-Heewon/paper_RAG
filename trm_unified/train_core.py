import importlib
import math
import os
import re
import json
import sys
import random
from datetime import timedelta
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .data import build_adj_from_tuples, iter_json_records, load_kb_map, load_rel_map, read_jsonl_by_offset, build_line_offsets
from .tokenization import load_tokenizer


def _set_global_seed(seed: int, deterministic: bool = False):
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _setup_wandb(args, is_main: bool):
    if not is_main:
        return None
    mode = str(getattr(args, "wandb_mode", "disabled") or "disabled").lower()
    if mode in {"off", "false", "none", "disabled", "disable"}:
        return None
    try:
        import importlib
        import sys
        import wandb  # type: ignore
        if not hasattr(wandb, "init"):
            # Local `./wandb` directory can shadow the pip package.
            sys.modules.pop("wandb", None)
            cwd = os.getcwd()
            old_path = list(sys.path)
            try:
                sys.path = [p for p in sys.path if p not in {"", cwd}]
                wandb = importlib.import_module("wandb")
            finally:
                sys.path = old_path
        if not hasattr(wandb, "init"):
            raise AttributeError("imported wandb module does not provide init()")
    except Exception as e:
        print(f"[warn] wandb import failed: {e}; continue without wandb logging")
        return None

    run = wandb.init(
        project=getattr(args, "wandb_project", None) or "graph-traverse",
        entity=getattr(args, "wandb_entity", None) or None,
        name=getattr(args, "wandb_run_name", None) or None,
        group=getattr(args, "wandb_group", None) or None,
        mode=mode,
        config={
            "dataset": os.path.basename(getattr(args, "train_json", "")),
            "model_impl": getattr(args, "model_impl", ""),
            "batch_size": int(getattr(args, "batch_size", 0)),
            "epochs": int(getattr(args, "epochs", 0)),
            "lr": float(getattr(args, "lr", 0.0)),
            "max_steps": int(getattr(args, "max_steps", 0)),
            "beam": int(getattr(args, "beam", 0)),
            "start_topk": int(getattr(args, "start_topk", 0)),
        },
    )
    return run


def _progress_write_line(message: str, last_chars: int = 0) -> int:
    msg = str(message)
    width = max(int(last_chars), len(msg))
    sys.stdout.write("\r" + msg.ljust(width))
    sys.stdout.flush()
    return width


def _progress_finish_line(last_chars: int = 0) -> None:
    if last_chars > 0:
        sys.stdout.write("\r" + (" " * int(last_chars)) + "\r")
    sys.stdout.write("\n")
    sys.stdout.flush()


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def prune_rel_mix_cpu(q_emb, cand_rels, rel_mem, keep_k, rand_k, rand_pool_mult=4):
    if keep_k <= 0 and rand_k <= 0:
        return cand_rels
    if len(cand_rels) <= keep_k + rand_k:
        return cand_rels
    r_ids = np.array(cand_rels, dtype=np.int64)
    r = l2_normalize_np(np.asarray(rel_mem[r_ids], dtype=np.float32))
    q = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    score = r @ q
    idx_sorted = np.argsort(-score)
    keep = idx_sorted[:max(0, keep_k)].tolist()
    start = len(keep)
    end = min(len(idx_sorted), start + rand_pool_mult * max(0, rand_k))
    mid = idx_sorted[start:end]
    rand = []
    if rand_k > 0 and len(mid) > 0:
        rand = np.random.choice(mid, size=min(rand_k, len(mid)), replace=False).tolist()
    sel = list(dict.fromkeys(keep + rand))
    return [cand_rels[i] for i in sel]


def _build_rel_to_nodes(edges):
    rel_to_nodes = {}
    rel_seen = {}
    for r, n in edges:
        rr = int(r)
        nn = int(n)
        if rr not in rel_to_nodes:
            rel_to_nodes[rr] = []
            rel_seen[rr] = set()
        if nn not in rel_seen[rr]:
            rel_seen[rr].add(nn)
            rel_to_nodes[rr].append(nn)
    return rel_to_nodes


def _select_rel_candidates(
    rel_to_nodes: dict,
    q_emb: np.ndarray,
    rel_mem,
    max_relations: int,
    prune_keep: int,
    prune_rand: int,
):
    rel_cands = list(rel_to_nodes.keys())
    if not rel_cands:
        return rel_cands
    if max_relations > 0 and len(rel_cands) > max_relations:
        rel_cands = prune_rel_mix_cpu(q_emb, rel_cands, rel_mem, int(max_relations), 0)
    if prune_keep > 0 or prune_rand > 0:
        rel_cands = prune_rel_mix_cpu(q_emb, rel_cands, rel_mem, int(prune_keep), int(prune_rand))
    return rel_cands


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n <= eps:
        return x
    return (x / (n + eps)).astype(np.float32, copy=False)


def _apply_query_residual_np(
    q_emb: np.ndarray,
    rel_id: int,
    rel_mem,
    enabled: bool = False,
    alpha: float = 0.0,
    mode: str = "sub_rel",
) -> np.ndarray:
    if (not enabled) or alpha == 0.0:
        return q_emb
    m = str(mode or "sub_rel").strip().lower()
    if m in {"none", "off", "disabled"}:
        return q_emb

    rid = int(rel_id)
    if rid < 0 or rid >= int(rel_mem.shape[0]):
        return q_emb

    rel_vec = np.asarray(rel_mem[rid], dtype=np.float32)
    q = np.asarray(q_emb, dtype=np.float32)
    if m in {"add", "plus", "add_rel"}:
        out = q + float(alpha) * rel_vec
    else:
        # Default: remove used-relation direction from query state.
        out = q - float(alpha) * rel_vec
    return _l2_normalize_np(out)


def _infer_vocab_size_from_state_dict(sd: dict) -> Optional[int]:
    if not isinstance(sd, dict):
        return None
    cand_keys = [
        "inner.embed_tokens.embedding_weight",
        "module.inner.embed_tokens.embedding_weight",
        "inner.lm_head.weight",
        "module.inner.lm_head.weight",
    ]
    for k in cand_keys:
        v = sd.get(k, None)
        if hasattr(v, "shape") and len(v.shape) >= 2:
            try:
                return int(v.shape[0])
            except Exception:
                continue
    return None


class PathDataset(Dataset):
    def __init__(self, jsonl_path: str, max_steps: int, min_path_hops: int = 1):
        self.jsonl_path = jsonl_path
        self.max_steps = max_steps
        self.min_path_hops = max(1, int(min_path_hops))
        self.offsets = build_line_offsets(jsonl_path, is_main=True)
        self.flat = []
        for i in tqdm(range(len(self.offsets)), desc='Flattening'):
            ex = read_jsonl_by_offset(self.jsonl_path, self.offsets, i)
            vps = ex.get('valid_paths', [])
            for pidx, p in enumerate(vps):
                if self.min_path_hops > 1:
                    if (not isinstance(p, list)) or (len(p) < self.min_path_hops):
                        continue
                self.flat.append((i, pidx))

    def __len__(self):
        return len(self.flat)

    def __getitem__(self, idx):
        li, pi = self.flat[idx]
        ex = read_jsonl_by_offset(self.jsonl_path, self.offsets, li)
        path = ex.get('valid_paths', [])[pi][:self.max_steps]
        if not path:
            segs = [0, 0, 0]
        else:
            segs = [path[0][0]]
            for s, r, o in path:
                segs.append(r)
                segs.append(o)
        answers_cid = []
        for x in ex.get('answers_cid', []):
            try:
                answers_cid.append(int(x))
            except Exception:
                pass
        return {
            'q_text': ex.get('question', ''),
            'tuples': ex.get('subgraph', {}).get('tuples', []),
            'path_segments': segs,
            'answers_cid': answers_cid,
            'ex_line': li,
        }


def make_collate(
    tokenizer_name,
    rel_npy,
    q_npy,
    max_neighbors,
    prune_keep,
    prune_rand,
    max_q_len,
    max_steps,
    endpoint_aux=False,
    entity_vocab_size: Optional[int] = None,
    query_residual_enabled: bool = False,
    query_residual_alpha: float = 0.0,
    query_residual_mode: str = "sub_rel",
):
    tok = load_tokenizer(tokenizer_name)
    rel_mem = np.load(rel_npy, mmap_mode='r')
    if not q_npy:
        raise RuntimeError("query embedding path is empty for training collate (q_npy).")
    if not os.path.exists(q_npy):
        raise FileNotFoundError(
            f"query embedding file not found for training: {q_npy}. "
            "Run embed stage first and ensure query_train.npy exists."
        )
    q_mem = np.load(q_npy, mmap_mode='r')
    if int(q_mem.shape[1]) != int(rel_mem.shape[1]):
        raise RuntimeError(
            f"embedding dim mismatch (train): query_dim={int(q_mem.shape[1])}, "
            f"relation_dim={int(rel_mem.shape[1])}. "
            "Rebuild query/relation embeddings with the same embedding model/style."
        )

    def _can_reach_goal(adj, node: int, goals: set, rem_steps: int, memo: dict) -> bool:
        key = (int(node), int(rem_steps))
        if key in memo:
            return memo[key]
        n = int(node)
        if n in goals:
            memo[key] = True
            return True
        if rem_steps <= 0:
            memo[key] = False
            return False
        for _, nxt in adj.get(n, []):
            if _can_reach_goal(adj, int(nxt), goals, rem_steps - 1, memo):
                memo[key] = True
                return True
        memo[key] = False
        return False

    def _shortest_dist_to_goals(adj, goals: set, max_dist: int):
        rev = {}
        for s, edges in adj.items():
            ss = int(s)
            for _, o in edges:
                oo = int(o)
                rev.setdefault(oo, []).append(ss)
        dist = {}
        q = deque()
        for g in goals:
            gg = int(g)
            dist[gg] = 0
            q.append(gg)
        while q:
            cur = q.popleft()
            d = dist[cur]
            if d >= max_dist:
                continue
            for prv in rev.get(cur, []):
                if prv in dist:
                    continue
                dist[prv] = d + 1
                q.append(prv)
        return dist

    def _fn(batch):
        toks = tok([b['q_text'] for b in batch], padding=True, truncation=True, max_length=max_q_len, return_tensors='pt')
        sample_ctx = []
        for b in batch:
            adj = build_adj_from_tuples(b['tuples'])
            qi = int(b['ex_line'])
            if qi < 0 or qi >= int(q_mem.shape[0]):
                raise RuntimeError(
                    f"query_train.npy index out of range: ex_line={qi}, "
                    f"q_mem_rows={int(q_mem.shape[0])}. "
                    "Rebuild embeddings from the same processed train.jsonl."
                )
            q_emb = np.asarray(q_mem[qi], dtype=np.float32)
            gold_set = set(int(x) for x in b.get('answers_cid', []))
            dist_to_goal = _shortest_dist_to_goals(adj, gold_set, max_steps) if gold_set else {}
            sample_ctx.append(
                {
                    'adj': adj,
                    'q_emb_init': np.asarray(q_emb, dtype=np.float32).copy(),
                    'q_emb': q_emb,
                    'gold_set': gold_set,
                    'reach_memo': {},
                    'dist_to_goal': dist_to_goal,
                }
            )

        seq_batches = []
        for t in range(max_steps):
            puzs, rels, labels, cmask, vmask, endpoint_targets, policy_targets = [], [], [], [], [], [], []
            for i, b in enumerate(batch):
                segs = b['path_segments']
                idx = t * 2
                if idx + 2 >= len(segs):
                    puzs.append([0]); rels.append([0]); labels.append(-100); cmask.append([False]); vmask.append(False); endpoint_targets.append([0.0]); policy_targets.append([0.0])
                    continue
                cur = int(segs[idx])
                gold_r = int(segs[idx + 1])
                ctx_i = sample_ctx[i]
                adj = ctx_i['adj']
                q_emb = ctx_i['q_emb']
                gold_set = ctx_i['gold_set']
                reach_memo = ctx_i['reach_memo']
                dist_to_goal = ctx_i['dist_to_goal']
                edges = adj.get(cur, [])
                rel_to_nodes = _build_rel_to_nodes(edges)
                rel_cands = _select_rel_candidates(
                    rel_to_nodes=rel_to_nodes,
                    q_emb=q_emb,
                    rel_mem=rel_mem,
                    max_relations=int(max_neighbors),
                    prune_keep=int(prune_keep),
                    prune_rand=int(prune_rand),
                )
                if gold_r not in rel_cands:
                    rel_cands.append(gold_r)
                np.random.shuffle(rel_cands)
                cur_pid = int(cur)
                if cur_pid < 0 or (entity_vocab_size is not None and cur_pid >= int(entity_vocab_size)):
                    cur_pid = 0
                puzs.append([cur_pid] * len(rel_cands))
                rels.append(rel_cands)
                labels.append(rel_cands.index(gold_r))
                cmask.append([True] * len(rel_cands))
                vmask.append(True)
                cur_dist = dist_to_goal.get(int(cur), None)
                remain = max_steps - t
                pos_rels_policy = set()
                if cur_dist is not None and cur_dist > 0 and cur_dist <= remain:
                    for rr, next_nodes in rel_to_nodes.items():
                        if any(dist_to_goal.get(int(nxt), None) == cur_dist - 1 for nxt in next_nodes):
                            pos_rels_policy.add(rr)
                pt_row = [1.0 if rr in pos_rels_policy else 0.0 for rr in rel_cands]
                if sum(pt_row) <= 0.0:
                    pt_row[rel_cands.index(gold_r)] = 1.0
                policy_targets.append(pt_row)
                if endpoint_aux:
                    rem_steps = max(0, max_steps - (t + 1))
                    pos_rels = set()
                    if gold_set:
                        for rr, next_nodes in rel_to_nodes.items():
                            if rr in pos_rels:
                                continue
                            if any(_can_reach_goal(adj, int(nn), gold_set, rem_steps, reach_memo) for nn in next_nodes):
                                pos_rels.add(rr)
                    endpoint_targets.append([1.0 if rr in pos_rels else 0.0 for rr in rel_cands])
                else:
                    endpoint_targets.append([0.0] * len(rel_cands))
                ctx_i['q_emb'] = _apply_query_residual_np(
                    q_emb=q_emb,
                    rel_id=gold_r,
                    rel_mem=rel_mem,
                    enabled=bool(query_residual_enabled),
                    alpha=float(query_residual_alpha),
                    mode=str(query_residual_mode),
                )

            cmax = max(len(x) for x in puzs)
            B = len(batch)
            pt = torch.zeros((B, cmax), dtype=torch.long)
            rt = torch.zeros((B, cmax), dtype=torch.long)
            mt = torch.zeros((B, cmax), dtype=torch.bool)
            et = torch.zeros((B, cmax), dtype=torch.float32)
            ptt = torch.zeros((B, cmax), dtype=torch.float32)
            for k in range(B):
                L = len(puzs[k])
                pt[k, :L] = torch.tensor(puzs[k], dtype=torch.long)
                rt[k, :L] = torch.tensor(rels[k], dtype=torch.long)
                mt[k, :L] = torch.tensor(cmask[k], dtype=torch.bool)
                et[k, :L] = torch.tensor(endpoint_targets[k], dtype=torch.float32)
                ptt[k, :L] = torch.tensor(policy_targets[k], dtype=torch.float32)
            seq_batches.append({
                'puzzle_identifiers': pt,
                'relation_identifiers': rt,
                'candidate_mask': mt,
                'endpoint_targets': et,
                'policy_targets': ptt,
                'labels': torch.tensor(labels, dtype=torch.long),
                'valid_mask': torch.tensor(vmask, dtype=torch.bool),
            })

        # Halt supervision:
        # - valid steps before the last hop: 0
        # - last valid hop of each path: 1
        # - padded steps: excluded from halt loss
        for t in range(max_steps):
            vm = seq_batches[t]['valid_mask']
            if t + 1 < max_steps:
                next_vm = seq_batches[t + 1]['valid_mask']
            else:
                next_vm = torch.zeros_like(vm)
            halt_targets = (vm & (~next_vm)).to(torch.float32)
            halt_mask = vm.clone()
            seq_batches[t]['halt_targets'] = halt_targets
            seq_batches[t]['halt_mask'] = halt_mask

        # For RL/SCST rollouts, start from the original query embedding.
        # Using teacher-forced residual-updated q_emb leaks gold-path information.
        rl_q = np.stack(
            [ctx.get('q_emb_init', ctx['q_emb']) for ctx in sample_ctx],
            axis=0,
        ).astype(np.float32)
        rl_starts = []
        rl_gold = []
        rl_tuples = []
        for b in batch:
            segs = b.get('path_segments', [])
            rl_starts.append(int(segs[0]) if len(segs) > 0 else 0)
            rl_gold.append([int(x) for x in b.get('answers_cid', [])])
            rl_tuples.append(b.get('tuples', []))
        return {
            'input_ids': toks['input_ids'],
            'attention_mask': toks['attention_mask'],
            'seq_batches': seq_batches,
            'rl_tuples': rl_tuples,
            'rl_start_nodes': rl_starts,
            'rl_gold_answers': rl_gold,
            'rl_q_embs': torch.tensor(rl_q, dtype=torch.float32),
        }

    return _fn


def _setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        timeout_min = int(os.environ.get("DDP_TIMEOUT_MINUTES", "30"))
        timeout_min = max(1, timeout_min)
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            timeout=timedelta(minutes=timeout_min),
        )
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
        return True, rank, local_rank, world_size, device
    return False, 0, 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(model_impl: str, trm_root: str, cfg: dict):
    if not os.path.isdir(trm_root):
        raise FileNotFoundError(
            f"TRM root not found: {trm_root}. "
            "Set --trm_root or TRM_ROOT to a valid model source path."
        )
    if trm_root not in os.sys.path:
        os.sys.path.append(trm_root)
    mod = importlib.import_module(f'models.recursive_reasoning.{model_impl}')
    cls = getattr(mod, 'TinyRecursiveReasoningModel_ACTV1')
    carry_cls = getattr(mod, 'TinyRecursiveReasoningModel_ACTV1Carry')
    return cls(cfg), carry_cls


def _clone_carry(carry):
    inner = carry.inner_carry
    inner_type = type(inner)
    carry_type = type(carry)
    inner_kwargs = {}
    for k in inner.__dataclass_fields__.keys():
        v = getattr(inner, k)
        inner_kwargs[k] = v.clone()
    current_data = {}
    for k, v in carry.current_data.items():
        current_data[k] = v.clone() if torch.is_tensor(v) else v
    return carry_type(
        inner_carry=inner_type(**inner_kwargs),
        steps=carry.steps.clone(),
        halted=carry.halted.clone(),
        current_data=current_data,
    )


def _parse_eval_example(ex: dict, kb2idx: Dict[str, int], rel2idx: Dict[str, int]):
    raw_tuples = ex.get('subgraph', {}).get('tuples', [])
    if not raw_tuples and 'subgraph_edges' in ex:
        raw_tuples = ex.get('subgraph_edges', [])

    final_tuples = []
    for s, r, o in raw_tuples:
        try:
            s_id = int(s)
            o_id = int(o)
        except Exception:
            continue
        if isinstance(r, int):
            r_id = int(r)
        elif isinstance(r, str):
            if r not in rel2idx:
                continue
            r_id = int(rel2idx[r])
        else:
            continue
        final_tuples.append((s_id, r_id, o_id))

    starts = []
    # Prefer canonical CID starts when available.
    for x in ex.get('entities_cid', []):
        try:
            starts.append(int(x))
        except Exception:
            pass

    if 'seed_entities' in ex:
        for se in ex.get('seed_entities', []):
            if isinstance(se, dict) and 'node_id' in se:
                try:
                    starts.append(int(se['node_id']))
                except Exception:
                    pass
    if not starts:
        for x in ex.get('entities', []):
            try:
                starts.append(int(x))
            except Exception:
                pass

    gold = set()
    for x in ex.get('answers_cid', []):
        try:
            gold.add(int(x))
        except Exception:
            pass
    for a in ex.get('answers', []):
        if isinstance(a, dict):
            kb_id = a.get('kb_id')
            if kb_id in kb2idx:
                gold.add(int(kb2idx[kb_id]))
    for a in ex.get('gold_answers', []):
        if isinstance(a, dict) and 'node_id' in a:
            try:
                gold.add(int(a['node_id']))
            except Exception:
                pass

    return final_tuples, starts, gold


def _load_entity_name_map(entity_names_json: str) -> Dict[str, str]:
    if not entity_names_json or not os.path.exists(entity_names_json):
        return {}
    try:
        with open(entity_names_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _load_entity_labels(entities_txt: str, entity_names_json: str = "") -> List[str]:
    name_map = _load_entity_name_map(entity_names_json)
    out = []
    with open(entities_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            eid = parts[0].strip() if parts else ""
            name = parts[1].strip() if len(parts) > 1 else ""
            if eid in name_map and str(name_map[eid]).strip():
                name = str(name_map[eid]).strip()
            if name and name != eid:
                out.append(f"{eid}:{name}")
            else:
                out.append(eid)
    return out


def _load_relation_labels(relations_txt: str) -> List[str]:
    out = []
    with open(relations_txt, "r", encoding="utf-8") as f:
        for line in f:
            out.append(line.strip())
    return out


def _idx_to_label(labels: List[str], idx: int) -> str:
    i = int(idx)
    if 0 <= i < len(labels):
        return labels[i]
    return str(i)


def _find_gold_path(adj, starts: List[int], gold_set: set, max_steps: int):
    starts = [int(x) for x in starts]
    if not starts or not gold_set:
        return None

    for s in starts:
        if s in gold_set:
            return {"nodes": [s], "rels": []}

    q = deque()
    parent = {}  # node -> (prev_node, rel_id)
    dist = {}
    for s in starts:
        q.append(int(s))
        dist[int(s)] = 0

    target = None
    while q:
        cur = q.popleft()
        d = dist[cur]
        if d >= max_steps:
            continue
        for r, nxt in adj.get(cur, []):
            nxt = int(nxt)
            if nxt in dist:
                continue
            dist[nxt] = d + 1
            parent[nxt] = (cur, int(r))
            if nxt in gold_set:
                target = nxt
                q.clear()
                break
            q.append(nxt)

    if target is None:
        return None

    rels = []
    nodes = [int(target)]
    cur = int(target)
    while cur in parent:
        prev, rel = parent[cur]
        rels.append(int(rel))
        nodes.append(int(prev))
        cur = int(prev)
    nodes.reverse()
    rels.reverse()
    return {"nodes": nodes, "rels": rels}


def diagnose_oracle_reachability(
    eval_json: str,
    kb2idx: Dict[str, int],
    rel2idx: Dict[str, int],
    max_steps: int,
    max_neighbors: int,
    start_topk: int,
    eval_limit: int = -1,
):
    data = list(iter_json_records(eval_json))
    total = len(data) if eval_limit < 0 else min(len(data), eval_limit)
    steps_cap = max(0, int(max_steps))
    topk = max(1, int(start_topk))
    neighbor_cap = int(max_neighbors)

    usable = 0
    answer_in_subgraph = 0
    reachable_at_k = 0
    reachable_any = 0
    need_more_steps = 0
    no_start = 0
    no_gold = 0
    no_tuple = 0
    shortest_hops = []
    hop_hist = [0 for _ in range(steps_cap + 1)]

    for ex_idx in tqdm(range(total), desc='OracleDiag'):
        ex = data[ex_idx]
        tuples, starts_raw, gold = _parse_eval_example(ex, kb2idx, rel2idx)
        if not tuples:
            no_tuple += 1
            continue
        starts = list(dict.fromkeys(starts_raw))[:topk]
        if not starts:
            no_start += 1
            continue
        if not gold:
            no_gold += 1
            continue

        usable += 1
        nodes_in_subgraph = set()
        for s, _, o in tuples:
            nodes_in_subgraph.add(int(s))
            nodes_in_subgraph.add(int(o))
        if bool(gold & nodes_in_subgraph):
            answer_in_subgraph += 1

        adj = build_adj_from_tuples(tuples)
        dist = {}
        q = deque()
        for s in starts:
            ss = int(s)
            if ss in dist:
                continue
            dist[ss] = 0
            q.append(ss)

        best = None
        while q:
            cur = q.popleft()
            d = dist[cur]
            if cur in gold:
                best = d
                break
            cand = adj.get(cur, [])
            if neighbor_cap > 0 and len(cand) > neighbor_cap:
                cand = cand[:neighbor_cap]
            for _, nxt in cand:
                nn = int(nxt)
                if nn in dist:
                    continue
                dist[nn] = d + 1
                q.append(nn)

        if best is not None:
            shortest_hops.append(int(best))
            reachable_any += 1
            if best <= steps_cap:
                reachable_at_k += 1
                hop_hist[int(best)] += 1
            else:
                need_more_steps += 1

    def _rate(n: int, d: int) -> float:
        return float(n) / float(max(1, d))

    avg_hops_any = float(sum(shortest_hops)) / float(max(1, len(shortest_hops)))
    out = {
        'total': int(total),
        'usable': int(usable),
        'no_tuple': int(no_tuple),
        'no_start': int(no_start),
        'no_gold': int(no_gold),
        'answer_in_subgraph': int(answer_in_subgraph),
        'answer_in_subgraph_rate_total': _rate(answer_in_subgraph, total),
        'answer_in_subgraph_rate_usable': _rate(answer_in_subgraph, usable),
        'reachable_at_k': int(reachable_at_k),
        'reachable_at_k_rate_total': _rate(reachable_at_k, total),
        'reachable_at_k_rate_usable': _rate(reachable_at_k, usable),
        'reachable_any': int(reachable_any),
        'reachable_any_rate_total': _rate(reachable_any, total),
        'reachable_any_rate_usable': _rate(reachable_any, usable),
        'need_more_steps': int(need_more_steps),
        'need_more_steps_rate_usable': _rate(need_more_steps, usable),
        'avg_shortest_hops_any': float(avg_hops_any),
        'hop_hist_at_k': {str(i): int(hop_hist[i]) for i in range(len(hop_hist))},
        'max_steps': int(steps_cap),
        'start_topk': int(topk),
        'max_neighbors': int(neighbor_cap),
    }
    return out


@torch.no_grad()
def evaluate_relation_beam(
    model,
    carry_init_fn,
    eval_json: str,
    kb2idx: Dict[str, int],
    rel2idx: Dict[str, int],
    device,
    tokenizer_name: str,
    q_npy: str,
    rel_npy: str,
    max_steps: int,
    max_neighbors: int,
    prune_keep: int,
    start_topk: int,
    beam: int,
    max_q_len: int,
    eval_limit: int,
    debug_n: int,
    no_cycle: bool = False,
    eval_pred_topk: int = 1,
    eval_use_halt: bool = False,
    eval_min_hops_before_stop: int = 1,
    query_residual_enabled: bool = False,
    query_residual_alpha: float = 0.0,
    query_residual_mode: str = "sub_rel",
    entity_labels: Optional[List[str]] = None,
    relation_labels: Optional[List[str]] = None,
    eval_dump_jsonl: str = "",
    eval_random_sample_size: int = 0,
    eval_random_seed: int = 42,
):
    model.eval()
    tok = load_tokenizer(tokenizer_name)
    rel_mem = np.load(rel_npy, mmap_mode='r')
    if not q_npy:
        raise RuntimeError(
            f"query embedding path is empty for evaluation (eval_json={eval_json}). "
            "Provide query_emb_dev_npy/query_emb_eval_npy."
        )
    if not os.path.exists(q_npy):
        raise FileNotFoundError(
            f"query embedding file not found for evaluation: {q_npy} "
            f"(eval_json={eval_json})."
        )
    q_mem = np.load(q_npy, mmap_mode='r')
    if int(q_mem.shape[1]) != int(rel_mem.shape[1]):
        raise RuntimeError(
            f"embedding dim mismatch (eval): query_dim={int(q_mem.shape[1])}, "
            f"relation_dim={int(rel_mem.shape[1])}, eval_json={eval_json}. "
            "Rebuild query/relation embeddings with the same embedding model/style."
        )

    data = list(iter_json_records(eval_json))
    n_data = int(len(data))
    rand_n = int(max(0, int(eval_random_sample_size)))
    if rand_n > 0:
        k = min(n_data, rand_n)
        rng = np.random.default_rng(int(eval_random_seed))
        eval_indices = rng.choice(n_data, size=k, replace=False).astype(np.int64).tolist()
    else:
        total_seq = n_data if eval_limit < 0 else min(n_data, int(eval_limit))
        eval_indices = list(range(total_seq))
    total = int(len(eval_indices))
    if total <= 0:
        return 0.0, 0.0, 0
    max_qi = int(max(eval_indices))
    if max_qi >= int(q_mem.shape[0]):
        raise RuntimeError(
            f"query embedding rows mismatch: max_eval_index={max_qi}, q_mem_rows={int(q_mem.shape[0])}, "
            f"eval_json={eval_json}, q_npy={q_npy}. "
            "Use query_dev.npy for dev eval and query_test.npy for test eval."
        )
    hit1 = []
    f1s = []
    skip = 0
    debugged = 0
    pred_topk = max(1, int(eval_pred_topk))
    min_hops_before_stop = max(0, int(eval_min_hops_before_stop))
    dump_fp = None
    if eval_dump_jsonl:
        dump_dir = os.path.dirname(str(eval_dump_jsonl))
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        dump_fp = open(eval_dump_jsonl, "w", encoding="utf-8")

    entity_vocab_size = None
    try:
        inner = model.module.inner if hasattr(model, "module") else model.inner
        emb = getattr(inner, "puzzle_emb_data", None)
        if emb is not None and hasattr(emb, "shape") and len(emb.shape) >= 1:
            entity_vocab_size = int(emb.shape[0])
    except Exception:
        entity_vocab_size = None

    def _safe_entity_id(x: int) -> int:
        i = int(x)
        if i < 0:
            return 0
        if entity_vocab_size is None:
            return i
        return i if i < entity_vocab_size else 0

    def _top_pred_entities(sorted_beams, k: int) -> List[int]:
        out = []
        seen = set()
        for b in sorted_beams:
            n = int(b['nodes'][-1])
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
            if len(out) >= k:
                break
        return out

    for eval_pos in tqdm(range(total), desc='Eval'):
        ex_idx = int(eval_indices[eval_pos])
        ex = data[ex_idx]
        tuples, starts_raw, gold = _parse_eval_example(ex, kb2idx, rel2idx)
        adj = build_adj_from_tuples(tuples)
        starts = list(dict.fromkeys(starts_raw))[:max(1, start_topk)]
        if not starts or not tuples:
            skip += 1
            continue
        if not gold:
            skip += 1
            continue

        qi = int(ex_idx)
        if qi < 0 or qi >= int(q_mem.shape[0]):
            raise RuntimeError(
                f"query eval index out of range: ex_idx={qi}, q_mem_rows={int(q_mem.shape[0])}, "
                f"eval_json={eval_json}, q_npy={q_npy}"
            )
        q_emb = np.asarray(q_mem[qi], dtype=np.float32)

        q_toks = tok(ex.get('question', ''), return_tensors='pt', truncation=True, max_length=max_q_len)
        q_toks = {k: v.to(device) for k, v in q_toks.items()}

        beams = []
        for s in starts:
            init_carry = carry_init_fn({'input_ids': q_toks['input_ids']}, device)
            beams.append(
                {
                    'score': 0.0,
                    'nodes': [int(s)],
                    'rels': [],
                    'carry': init_carry,
                    'q_emb': np.asarray(q_emb, dtype=np.float32).copy(),
                    'stopped': False,
                }
            )

        beams = beams[:max(1, beam)]

        for _ in range(max_steps):
            new_beams = []
            for b in beams:
                if bool(b.get('stopped', False)):
                    new_beams.append(b)
                    continue
                cur = int(b['nodes'][-1])
                cand = adj.get(cur, [])
                if not cand:
                    nb = dict(b)
                    nb['stopped'] = True
                    new_beams.append(nb)
                    continue
                if no_cycle:
                    cand = [(r, nxt) for r, nxt in cand if int(nxt) not in b['nodes']]
                    if not cand:
                        nb = dict(b)
                        nb['stopped'] = True
                        new_beams.append(nb)
                        continue

                rel_to_nodes = _build_rel_to_nodes(cand)
                rel_cands = list(rel_to_nodes.keys())
                if not rel_cands:
                    nb = dict(b)
                    nb['stopped'] = True
                    new_beams.append(nb)
                    continue
                rel_cands = _select_rel_candidates(
                    rel_to_nodes=rel_to_nodes,
                    q_emb=np.asarray(b.get('q_emb', q_emb), dtype=np.float32),
                    rel_mem=rel_mem,
                    max_relations=int(max_neighbors),
                    prune_keep=int(prune_keep),
                    prune_rand=0,
                )
                if not rel_cands:
                    nb = dict(b)
                    nb['stopped'] = True
                    new_beams.append(nb)
                    continue

                cur_pid = _safe_entity_id(cur)
                p_ids = torch.full((1, len(rel_cands)), cur_pid, dtype=torch.long, device=device)
                r_ids = torch.tensor([rel_cands], dtype=torch.long, device=device)
                c_mask = torch.ones_like(r_ids, dtype=torch.bool)
                inp = {
                    'input_ids': q_toks['input_ids'],
                    'attention_mask': q_toks['attention_mask'],
                    'puzzle_identifiers': p_ids,
                    'relation_identifiers': r_ids,
                    'candidate_mask': c_mask,
                }

                next_carry, out = model(b['carry'], inp)
                logp = torch.log_softmax(out['scores'][0], dim=0)
                log_continue = 0.0
                if bool(eval_use_halt) and len(b['rels']) >= min_hops_before_stop:
                    halt_pair = torch.stack(
                        [out['q_continue_logits'][0], out['q_halt_logits'][0]], dim=0
                    )
                    halt_logp = torch.log_softmax(halt_pair, dim=0)
                    log_continue = float(halt_logp[0].item())
                    log_halt = float(halt_logp[1].item())
                    new_beams.append({
                        'score': float(b['score'] + log_halt),
                        'nodes': list(b['nodes']),
                        'rels': list(b['rels']),
                        'carry': _clone_carry(next_carry),
                        'q_emb': np.asarray(b.get('q_emb', q_emb), dtype=np.float32).copy(),
                        'stopped': True,
                    })
                topk = min(max(1, beam), len(rel_cands))
                topv, topi = torch.topk(logp, k=topk)
                for v, i in zip(topv.tolist(), topi.tolist()):
                    r_sel = int(rel_cands[int(i)])
                    q_next = _apply_query_residual_np(
                        q_emb=np.asarray(b.get('q_emb', q_emb), dtype=np.float32),
                        rel_id=r_sel,
                        rel_mem=rel_mem,
                        enabled=bool(query_residual_enabled),
                        alpha=float(query_residual_alpha),
                        mode=str(query_residual_mode),
                    )
                    next_nodes = rel_to_nodes.get(r_sel, [])
                    if not next_nodes:
                        if no_cycle:
                            continue
                        new_beams.append({
                            'score': float(b['score'] + log_continue + v),
                            'nodes': list(b['nodes']),
                            'rels': list(b['rels']) + [r_sel],
                            'carry': _clone_carry(next_carry),
                            'q_emb': np.asarray(q_next, dtype=np.float32).copy(),
                            'stopped': False,
                        })
                    else:
                        for nxt in next_nodes:
                            new_beams.append({
                                'score': float(b['score'] + log_continue + v),
                                'nodes': list(b['nodes']) + [int(nxt)],
                                'rels': list(b['rels']) + [r_sel],
                                'carry': _clone_carry(next_carry),
                                'q_emb': np.asarray(q_next, dtype=np.float32).copy(),
                                'stopped': False,
                            })

            new_beams.sort(key=lambda x: x['score'], reverse=True)
            beams = new_beams[:max(1, beam)]
            if not beams:
                break
            if all(bool(x.get('stopped', False)) for x in beams):
                break

        if not beams:
            skip += 1
            continue

        best = beams[0]
        pred = int(best['nodes'][-1])
        pred_entities = _top_pred_entities(beams, pred_topk)
        pred_set = set(pred_entities)
        gold_path = None
        if gold:
            h = 1.0 if pred in gold else 0.0
            hit1.append(h)
            inter = len(pred_set & gold)
            if inter == 0:
                f1_i = 0.0
                f1s.append(f1_i)
            else:
                precision = inter / max(1, len(pred_set))
                recall = inter / max(1, len(gold))
                f1_i = (2.0 * precision * recall) / max(1e-12, precision + recall)
                f1s.append(f1_i)
            if dump_fp is not None or debugged < max(0, debug_n):
                gold_path = _find_gold_path(adj, starts_raw, gold, max_steps=max_steps)
            if dump_fp is not None:
                rec = {
                    "index": int(ex_idx),
                    "question": ex.get("question", ""),
                    "hit1": float(h),
                    "f1": float(f1_i),
                    "pred_entity": int(pred),
                    "pred_top_entities": [int(x) for x in pred_entities],
                    "pred_path_nodes": [int(x) for x in best.get("nodes", [])],
                    "pred_path_relations": [int(x) for x in best.get("rels", [])],
                    "pred_path_hops": int(len(best.get("rels", []))),
                    "gold_entities": sorted(int(x) for x in gold),
                    "num_gold": int(len(gold)),
                    "gold_path_hops_within_max_steps": (
                        int(len(gold_path.get("rels", []))) if gold_path is not None else None
                    ),
                    "gold_reachable_within_max_steps": bool(gold_path is not None),
                    "first_pred_relation": (
                        int(best.get("rels", [])[0]) if len(best.get("rels", [])) > 0 else None
                    ),
                    "first_pred_relation_text": (
                        _idx_to_label(relation_labels, best.get("rels", [])[0])
                        if relation_labels and len(best.get("rels", [])) > 0
                        else None
                    ),
                    "pred_entity_text": (
                        _idx_to_label(entity_labels, pred) if entity_labels else str(pred)
                    ),
                }
                dump_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if debugged < max(0, debug_n):
            debugged += 1
            rel_path = ' -> '.join(str(x) for x in best['rels'])
            node_path = ' -> '.join(str(x) for x in best['nodes'])
            print(f"[EvalQ] {ex.get('question', '')}")
            print(f"  relation_path: {rel_path}")
            print(f"  node_path: {node_path}")
            if relation_labels:
                rel_text_path = ' -> '.join(_idx_to_label(relation_labels, x) for x in best['rels'])
                print(f"  relation_text_path: {rel_text_path}")
            if entity_labels:
                node_text_path = ' -> '.join(_idx_to_label(entity_labels, x) for x in best['nodes'])
                print(f"  node_text_path: {node_text_path}")
            print(f"  pred_entity: {pred} | gold_n={len(gold)}")
            print(f"  pred_top{pred_topk}_entities: {pred_entities}")
            if entity_labels:
                pred_top_txt = ' | '.join(_idx_to_label(entity_labels, x) for x in pred_entities)
                print(f"  pred_top{pred_topk}_text: {pred_top_txt}")
            if gold_path is None:
                gold_path = _find_gold_path(adj, starts_raw, gold, max_steps=max_steps)
            if gold_path is None:
                print("  gold_path: <not found within max_steps>")
            else:
                g_rels = ' -> '.join(str(x) for x in gold_path["rels"])
                g_nodes = ' -> '.join(str(x) for x in gold_path["nodes"])
                print(f"  gold_relation_path: {g_rels}")
                print(f"  gold_node_path: {g_nodes}")
                if relation_labels:
                    g_rel_txt = ' -> '.join(_idx_to_label(relation_labels, x) for x in gold_path["rels"])
                    print(f"  gold_relation_text_path: {g_rel_txt}")
                if entity_labels:
                    g_node_txt = ' -> '.join(_idx_to_label(entity_labels, x) for x in gold_path["nodes"])
                    print(f"  gold_node_text_path: {g_node_txt}")

    if dump_fp is not None:
        dump_fp.close()
    m_hit = float(np.mean(hit1)) if hit1 else 0.0
    m_f1 = float(np.mean(f1s)) if f1s else 0.0
    return m_hit, m_f1, skip


def train(args):
    is_ddp, rank, local_rank, world_size, device = _setup_ddp()
    is_main = rank == 0
    base_seed = int(getattr(args, "seed", 42))
    det = bool(getattr(args, "deterministic", False))
    seed_this_rank = int(base_seed + rank)
    _set_global_seed(seed_this_rank, deterministic=det)
    if is_main:
        print(
            f"[Reproducibility] seed={base_seed} per_rank_seed=seed+rank deterministic={det}"
        )
    wb = _setup_wandb(args, is_main)
    subgraph_reader_enabled = str(getattr(args, "subgraph_reader_enabled", False)).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if subgraph_reader_enabled:
        from .subgraph_reader import train_subgraph_reader

        train_subgraph_reader(
            args,
            is_ddp=is_ddp,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            wb=wb,
        )
        if is_ddp:
            dist.destroy_process_group()
        if wb is not None:
            wb.finish()
        return
    raise RuntimeError(
        "This workspace is configured for subgraph-reader-only training "
        "(D / D+latent). Set subgraph_reader_enabled=true."
    )

    # Full-dev evaluation on rank0 while other DDP ranks wait at barrier can
    # hit process-group timeout when eval_limit=-1 and timeout is too small.
    if is_ddp and str(getattr(args, 'dev_json', '')).strip():
        eval_limit_cfg = int(getattr(args, 'eval_limit', -1))
        timeout_min_cfg = int(os.environ.get("DDP_TIMEOUT_MINUTES", "30"))
        if eval_limit_cfg < 0 and timeout_min_cfg <= 90:
            raise RuntimeError(
                "DDP risk: eval_limit=-1 (full dev eval) with DDP_TIMEOUT_MINUTES<=90 may timeout "
                "while non-rank0 workers wait at barrier. "
                "Set EVAL_LIMIT=200 (recommended) or DDP_TIMEOUT_MINUTES>=180."
            )

    kb2idx = load_kb_map(args.entities_txt)
    rel2idx = load_rel_map(args.relations_txt)
    entity_labels = _load_entity_labels(args.entities_txt, getattr(args, 'entity_names_json', ''))
    relation_labels = _load_relation_labels(args.relations_txt)
    tok = load_tokenizer(args.trm_tokenizer)
    ent_mem = np.load(args.entity_emb_npy, mmap_mode='r')
    rel_mem = np.load(args.relation_emb_npy, mmap_mode='r')
    preloaded_sd = None
    ckpt_vocab_size = None
    if args.ckpt and os.path.exists(args.ckpt):
        preloaded_sd = torch.load(args.ckpt, map_location='cpu')
        ckpt_vocab_size = _infer_vocab_size_from_state_dict(preloaded_sd)
    vocab_size = int(tok.vocab_size)
    if ckpt_vocab_size is not None and int(ckpt_vocab_size) > 0 and int(ckpt_vocab_size) != int(vocab_size):
        if is_main:
            print(
                f"[warn] tokenizer vocab_size({vocab_size}) != ckpt vocab_size({int(ckpt_vocab_size)}); "
                "using ckpt vocab_size for model build."
            )
        vocab_size = int(ckpt_vocab_size)

    cfg = {
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'vocab_size': vocab_size,
        'hidden_size': args.hidden_size,
        'num_heads': args.num_heads,
        'expansion': args.expansion,
        'H_cycles': args.H_cycles,
        'L_cycles': args.L_cycles,
        'L_layers': args.L_layers,
        'H_layers': 0,
        'puzzle_emb_ndim': int(ent_mem.shape[1]),
        'relation_emb_ndim': int(rel_mem.shape[1]),
        'puzzle_emb_len': args.puzzle_emb_len,
        'pos_encodings': args.pos_encodings,
        'forward_dtype': args.forward_dtype,
        'halt_max_steps': args.halt_max_steps,
        'halt_exploration_prob': args.halt_exploration_prob,
        'no_ACT_continue': True,
        'mlp_t': False,
        'num_puzzle_identifiers': len(kb2idx) + 1000,
        'num_relation_identifiers': len(rel2idx) + 1000,
    }

    model, carry_cls = build_model(args.model_impl, args.trm_root, cfg)
    model.inner.puzzle_emb_data = ent_mem
    model.inner.relation_emb_data = rel_mem
    if bool(getattr(args, 'freeze_lm_head', True)):
        for p in model.inner.lm_head.parameters():
            p.requires_grad = False

    if preloaded_sd is not None:
        model.load_state_dict(preloaded_sd, strict=False)

    model.to(device)
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=bool(getattr(args, 'ddp_find_unused', False)),
        )

    def carry_init_fn(batch, dev):
        B = int(batch['input_ids'].shape[0])
        inner = model.module.inner if hasattr(model, 'module') else model.inner
        return carry_cls(inner.empty_carry(B), torch.zeros(B, device=dev), torch.ones(B, dtype=torch.bool, device=dev), {})

    if is_main and getattr(args, 'dev_json', '') and bool(getattr(args, 'oracle_diag_enabled', True)):
        diag_max_steps = int(getattr(args, 'eval_max_steps', args.max_steps))
        diag_max_neighbors = int(getattr(args, 'eval_max_neighbors', args.max_neighbors))
        diag_start_topk = int(getattr(args, 'eval_start_topk', getattr(args, 'start_topk', 5)))
        diag_limit = int(getattr(args, 'oracle_diag_limit', -1))
        diag = diagnose_oracle_reachability(
            eval_json=args.dev_json,
            kb2idx=kb2idx,
            rel2idx=rel2idx,
            max_steps=diag_max_steps,
            max_neighbors=diag_max_neighbors,
            start_topk=diag_start_topk,
            eval_limit=diag_limit,
        )
        print(
            "[OracleDiag:Dev] "
            f"usable={diag['usable']}/{diag['total']} "
            f"answer_in_subgraph={diag['answer_in_subgraph_rate_usable']:.4f} "
            f"reachable@{diag_max_steps}={diag['reachable_at_k_rate_usable']:.4f} "
            f"reachable_any={diag['reachable_any_rate_usable']:.4f} "
            f"need_more_steps={diag['need_more_steps_rate_usable']:.4f} "
            f"avg_shortest_hops_any={diag['avg_shortest_hops_any']:.2f}"
        )
        if wb is not None:
            wb.log(
                {
                    "oracle/dev_usable": int(diag['usable']),
                    "oracle/dev_total": int(diag['total']),
                    "oracle/dev_answer_in_subgraph_rate_usable": float(diag['answer_in_subgraph_rate_usable']),
                    "oracle/dev_reachable_at_k_rate_usable": float(diag['reachable_at_k_rate_usable']),
                    "oracle/dev_reachable_any_rate_usable": float(diag['reachable_any_rate_usable']),
                    "oracle/dev_need_more_steps_rate_usable": float(diag['need_more_steps_rate_usable']),
                    "oracle/dev_avg_shortest_hops_any": float(diag['avg_shortest_hops_any']),
                },
                step=0,
            )
        fail_thr = float(getattr(args, 'oracle_diag_fail_threshold', -1.0))
        if fail_thr >= 0.0 and diag['reachable_at_k_rate_usable'] < fail_thr:
            raise RuntimeError(
                f"[OracleDiag:Dev] reachable@{diag_max_steps}={diag['reachable_at_k_rate_usable']:.4f} < threshold={fail_thr:.4f}. "
                "Increase max_steps / neighbors or rebuild preprocessing before training."
            )

    if bool(getattr(args, 'oracle_diag_only', False)):
        if is_ddp:
            dist.destroy_process_group()
        if wb is not None:
            wb.finish()
        return

    train_min_path_hops = max(1, int(getattr(args, 'train_min_path_hops', 1)))
    ds = PathDataset(args.train_json, args.max_steps, min_path_hops=train_min_path_hops)
    if len(ds) <= 0:
        raise RuntimeError(
            f"Empty supervised dataset after flattening paths: train_json={args.train_json}, "
            f"max_steps={int(args.max_steps)}, train_min_path_hops={int(train_min_path_hops)}. "
            "Lower train_min_path_hops, increase max_steps, or rebuild preprocessing with richer valid_paths."
        )
    if is_main and train_min_path_hops > 1:
        print(f"[TrainData] min_path_hops={train_min_path_hops} (short paths are skipped in supervised dataset)")

    def _parse_optional_float(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().lower()
        if s in {"", "none", "null", "nan"}:
            return None
        return float(v)

    def _parse_optional_int(v):
        if v is None:
            return None
        if isinstance(v, int):
            return int(v)
        s = str(v).strip().lower()
        if s in {"", "none", "null", "nan"}:
            return None
        return int(v)

    def _as_bool(v, default=False):
        if v is None:
            return bool(default)
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    endpoint_aux_weight = float(getattr(args, 'endpoint_aux_weight', 0.0))
    metric_align_aux_weight = float(getattr(args, 'metric_align_aux_weight', getattr(args, 'policy_aux_weight', 0.0)))
    endpoint_loss_mode = str(getattr(args, 'endpoint_loss_mode', 'aux')).strip().lower()
    relation_aux_weight = float(getattr(args, 'relation_aux_weight', 1.0))
    halt_aux_weight = float(getattr(args, 'halt_aux_weight', 0.0))

    phase2_start_epoch = int(getattr(args, 'phase2_start_epoch', 0))
    phase2_endpoint_loss_mode = str(getattr(args, 'phase2_endpoint_loss_mode', '')).strip().lower()
    phase2_relation_aux_weight = _parse_optional_float(getattr(args, 'phase2_relation_aux_weight', None))
    phase2_endpoint_aux_weight = _parse_optional_float(getattr(args, 'phase2_endpoint_aux_weight', None))
    phase2_metric_align_aux_weight = _parse_optional_float(getattr(args, 'phase2_metric_align_aux_weight', None))
    phase2_halt_aux_weight = _parse_optional_float(getattr(args, 'phase2_halt_aux_weight', None))
    phase2_auto_enabled = bool(getattr(args, 'phase2_auto_enabled', False))
    phase2_auto_metric = str(getattr(args, 'phase2_auto_metric', 'dev_f1')).strip().lower()
    phase2_auto_threshold = _parse_optional_float(getattr(args, 'phase2_auto_threshold', None))
    phase2_auto_patience = int(_parse_optional_int(getattr(args, 'phase2_auto_patience', 0)) or 0)
    phase2_auto_min_epoch = int(_parse_optional_int(getattr(args, 'phase2_auto_min_epoch', 1)) or 1)
    phase2_auto_min_delta = float(getattr(args, 'phase2_auto_min_delta', 1e-4))
    phase2_rl_reward_metric = str(getattr(args, 'phase2_rl_reward_metric', 'f1')).strip().lower()
    phase2_rl_entropy_weight = float(getattr(args, 'phase2_rl_entropy_weight', 0.0))
    phase2_rl_sample_temp = float(getattr(args, 'phase2_rl_sample_temp', 1.0))
    phase2_rl_use_greedy_baseline = bool(getattr(args, 'phase2_rl_use_greedy_baseline', True))
    phase2_rl_no_cycle = bool(getattr(args, 'phase2_rl_no_cycle', getattr(args, 'eval_no_cycle', False)))
    phase2_rl_adv_clip = _parse_optional_float(getattr(args, 'phase2_rl_adv_clip', None))
    train_acc_mode = str(getattr(args, 'train_acc_mode', 'endpoint_proxy')).strip().lower()
    if train_acc_mode not in {'auto', 'endpoint_proxy', 'relation'}:
        train_acc_mode = 'endpoint_proxy'
    if phase2_auto_metric not in {'dev_f1', 'dev_hit1', 'train_acc', 'train_loss'}:
        phase2_auto_metric = 'dev_f1'
    if phase2_rl_reward_metric not in {'f1', 'hit1', 'hit'}:
        phase2_rl_reward_metric = 'f1'
    if phase2_rl_sample_temp <= 0.0:
        phase2_rl_sample_temp = 1.0
    phase2_auto_active = False
    phase2_auto_bad_count = 0
    phase2_auto_best = None

    train_sanity_eval_every_pct = int(getattr(args, 'train_sanity_eval_every_pct', 0))
    train_sanity_eval_limit = int(getattr(args, 'train_sanity_eval_limit', 5))
    train_sanity_eval_beam = int(getattr(args, 'train_sanity_eval_beam', 5))
    train_sanity_eval_start_topk = int(getattr(args, 'train_sanity_eval_start_topk', getattr(args, 'start_topk', 5)))
    train_sanity_eval_pred_topk = int(getattr(args, 'train_sanity_eval_pred_topk', 1))
    train_sanity_eval_no_cycle = _as_bool(getattr(args, 'train_sanity_eval_no_cycle', getattr(args, 'eval_no_cycle', False)))
    train_sanity_eval_use_halt = _as_bool(getattr(args, 'train_sanity_eval_use_halt', False))
    train_sanity_eval_max_neighbors = int(getattr(args, 'train_sanity_eval_max_neighbors', getattr(args, 'eval_max_neighbors', args.max_neighbors)))
    train_sanity_eval_prune_keep = int(getattr(args, 'train_sanity_eval_prune_keep', getattr(args, 'eval_prune_keep', args.prune_keep)))
    query_residual_enabled = _as_bool(getattr(args, 'query_residual_enabled', False))
    query_residual_alpha = float(getattr(args, 'query_residual_alpha', 0.0))
    query_residual_mode = str(getattr(args, 'query_residual_mode', 'sub_rel')).strip().lower()
    eval_random_sample_size = int(getattr(args, 'eval_random_sample_size', 0))
    eval_random_seed = int(getattr(args, 'eval_random_seed', 42))
    eval_random_resample_each_eval = _as_bool(getattr(args, 'eval_random_resample_each_eval', False))
    if train_sanity_eval_every_pct < 0:
        train_sanity_eval_every_pct = 0
    if train_sanity_eval_limit < 0:
        train_sanity_eval_limit = 0

    endpoint_main_modes = {'main', 'endpoint_main', 'endpoint-first', 'endpoint_first'}
    rl_scst_modes = {'rl_scst', 'scst', 'reinforce', 'policy_gradient'}
    entity_dist_main_modes = {
        'entity_dist_main',
        'entity_main',
        'answer_dist_main',
        'dist_main',
        'gnn_dist_main',
    }
    metric_align_main_modes = {
        'metric_align_main',
        'metric_main',
        'policy_main',
        'hitf1_main',
        'align_main',
    }

    collate = make_collate(
        args.trm_tokenizer,
        args.relation_emb_npy,
        args.query_emb_train_npy,
        args.max_neighbors,
        args.prune_keep,
        args.prune_rand,
        args.max_q_len,
        args.max_steps,
        endpoint_aux=True,
        entity_vocab_size=int(ent_mem.shape[0]),
        query_residual_enabled=query_residual_enabled,
        query_residual_alpha=query_residual_alpha,
        query_residual_mode=query_residual_mode,
    )
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=args.num_workers, drop_last=True, collate_fn=collate, pin_memory=torch.cuda.is_available(), persistent_workers=(args.num_workers > 0))
    if len(loader) <= 0:
        raise RuntimeError(
            f"Empty training loader: dataset_size={len(ds)}, batch_size={int(args.batch_size)}, drop_last=True. "
            "Reduce batch_size or adjust train_min_path_hops/preprocessing."
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr)
    ce = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    bce_endpoint = nn.BCEWithLogitsLoss(reduction='none')
    ce_halt = nn.CrossEntropyLoss(reduction='none')
    last_phase2_state = None

    def _rl_reward_from_pred(pred_node: int, gold_set: set, metric: str) -> float:
        if not gold_set:
            return 0.0
        hit = 1.0 if int(pred_node) in gold_set else 0.0
        if metric in {'hit', 'hit1'}:
            return hit
        if hit <= 0.0:
            return 0.0
        precision = 1.0
        recall = 1.0 / float(max(1, len(gold_set)))
        return float((2.0 * precision * recall) / max(1e-12, precision + recall))

    def _rollout_single_policy(
        input_ids_1,
        attn_1,
        adj: dict,
        start_node: int,
        q_emb_np: np.ndarray,
        sample_action: bool,
        no_cycle: bool,
        temperature: float,
        with_grad: bool,
        query_residual_enabled: bool = False,
        query_residual_alpha: float = 0.0,
        query_residual_mode: str = "sub_rel",
    ):
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            carry_1 = carry_init_fn({'input_ids': input_ids_1}, device)
            cur = int(start_node)
            q_emb_cur = np.asarray(q_emb_np, dtype=np.float32)
            visited = {cur}
            logprob_sum = torch.zeros((), device=device, requires_grad=with_grad)
            entropy_sum = torch.zeros((), device=device, requires_grad=with_grad)
            taken = 0
            for _ in range(int(args.max_steps)):
                edges = adj.get(cur, [])
                if no_cycle:
                    edges = [(r, n) for r, n in edges if int(n) not in visited]
                if not edges:
                    break
                rel_to_nodes = _build_rel_to_nodes(edges)
                rel_cands = _select_rel_candidates(
                    rel_to_nodes=rel_to_nodes,
                    q_emb=q_emb_cur,
                    rel_mem=rel_mem,
                    max_relations=int(args.max_neighbors),
                    prune_keep=int(args.prune_keep),
                    prune_rand=int(args.prune_rand),
                )
                if not rel_cands:
                    break
                cur_pid = int(cur)
                if cur_pid < 0 or cur_pid >= int(ent_mem.shape[0]):
                    cur_pid = 0
                p_ids = torch.full((1, len(rel_cands)), cur_pid, dtype=torch.long, device=device)
                r_ids = torch.tensor([rel_cands], dtype=torch.long, device=device)
                c_mask = torch.ones_like(r_ids, dtype=torch.bool)
                carry_1, out = model(
                    carry_1,
                    {
                        'input_ids': input_ids_1,
                        'attention_mask': attn_1,
                        'puzzle_identifiers': p_ids,
                        'relation_identifiers': r_ids,
                        'candidate_mask': c_mask,
                    },
                )
                logits_1 = out['scores'][0].masked_fill(~c_mask[0], -1e4)
                logits_1 = logits_1 / float(temperature)
                logp_1 = torch.log_softmax(logits_1, dim=0)
                prob_1 = torch.softmax(logits_1, dim=0)
                entropy_sum = entropy_sum + (-(prob_1 * logp_1).sum())
                if sample_action:
                    idx = int(torch.multinomial(prob_1, num_samples=1).item())
                else:
                    idx = int(torch.argmax(logp_1).item())
                logprob_sum = logprob_sum + logp_1[idx]
                r_sel = int(rel_cands[idx])
                q_emb_cur = _apply_query_residual_np(
                    q_emb=q_emb_cur,
                    rel_id=r_sel,
                    rel_mem=rel_mem,
                    enabled=bool(query_residual_enabled),
                    alpha=float(query_residual_alpha),
                    mode=str(query_residual_mode),
                )
                next_nodes = rel_to_nodes.get(r_sel, [])
                if not next_nodes:
                    break
                if sample_action and len(next_nodes) > 1:
                    nxt = int(next_nodes[np.random.randint(0, len(next_nodes))])
                else:
                    nxt = int(next_nodes[0])
                cur = nxt
                visited.add(cur)
                taken += 1
            return logprob_sum, entropy_sum, int(cur), int(taken)

    def _entity_dist_loss_from_steps(
        step_cache,
        tuples_list,
        starts_list,
        golds_list,
        *,
        compute_loss: bool = True,
        return_stats: bool = False,
    ):
        eps = 1e-12
        losses = []
        hit_sum = 0.0
        f1_sum = 0.0
        n_stat = 0
        B = int(len(starts_list))
        for i in range(B):
            gold_set = set(int(x) for x in golds_list[i])
            if not gold_set:
                continue

            raw_tuples = tuples_list[i]
            edges_by_src = {}
            nodes = set()
            for s, r, o in raw_tuples:
                try:
                    ss = int(s)
                    rr = int(r)
                    oo = int(o)
                except Exception:
                    continue
                edges_by_src.setdefault(ss, []).append((rr, oo))
                nodes.add(ss)
                nodes.add(oo)

            start_node = int(starts_list[i])
            nodes.add(start_node)
            nodes.update(gold_set)
            if not nodes:
                continue

            node_list = list(nodes)
            node2idx = {n: j for j, n in enumerate(node_list)}
            if start_node not in node2idx:
                continue
            N = len(node_list)
            dtype = step_cache[0]['probs'].dtype
            p = torch.zeros((N,), device=device, dtype=dtype)
            p[node2idx[start_node]] = 1.0

            for st in step_cache:
                if not bool(st['v_mask'][i].item()):
                    break
                rel_ids = st['r_ids'][i]
                rel_mask = st['c_mask'][i]
                rel_prob = st['probs'][i]

                rel_prob_map = {}
                valid_pos = torch.nonzero(rel_mask, as_tuple=False).squeeze(-1)
                for pos in valid_pos.tolist():
                    rr = int(rel_ids[pos].item())
                    rel_prob_map[rr] = rel_prob_map.get(rr, 0.0) + rel_prob[pos]

                p_next = torch.zeros_like(p)
                for u_node, u_idx in node2idx.items():
                    mass = p[u_idx]
                    out_edges = edges_by_src.get(int(u_node), [])
                    if not out_edges:
                        p_next[u_idx] = p_next[u_idx] + mass
                        continue

                    rel_to_targets = {}
                    for rr, vv in out_edges:
                        rel_to_targets.setdefault(int(rr), []).append(int(vv))

                    prob_used = torch.zeros((), device=device, dtype=dtype)
                    for rr, targets in rel_to_targets.items():
                        pr = rel_prob_map.get(rr, None)
                        if pr is None:
                            continue
                        prob_used = prob_used + pr
                        share = mass * pr / float(max(1, len(targets)))
                        for vv in targets:
                            if vv in node2idx:
                                p_next[node2idx[vv]] = p_next[node2idx[vv]] + share

                    # Keep residual mass to avoid collapsing when sampled relations
                    # are unavailable from this source node.
                    p_next[u_idx] = p_next[u_idx] + mass * torch.clamp(1.0 - prob_used, min=0.0)

                z = p_next.sum()
                if float(z.detach().item()) > 0.0:
                    p = p_next / (z + eps)
                else:
                    p = p_next

            if return_stats:
                pred_idx = int(torch.argmax(p).item())
                pred_node = int(node_list[pred_idx])
                inter = 1 if pred_node in gold_set else 0
                hit = float(inter)
                precision = float(inter)  # |pred_set|=1
                recall = float(inter) / float(max(1, len(gold_set)))
                f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
                hit_sum += hit
                f1_sum += f1
                n_stat += 1

            if compute_loss:
                gold = torch.zeros((N,), device=device, dtype=dtype)
                hit_gold = 0
                for g in gold_set:
                    if g in node2idx:
                        gold[node2idx[g]] = 1.0
                        hit_gold += 1
                if hit_gold <= 0:
                    continue
                gold = gold / gold.sum().clamp(min=1.0)
                loss_i = F.kl_div(torch.log(p.clamp(min=eps)), gold, reduction='sum')
                losses.append(loss_i)

        loss_out = torch.stack(losses).mean() if (compute_loss and losses) else torch.zeros((), device=device)
        if return_stats:
            return loss_out, float(hit_sum), float(f1_sum), int(n_stat)
        return loss_out

    for ep in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        use_phase2 = (phase2_start_epoch > 0 and ep >= phase2_start_epoch) or phase2_auto_active
        cur_endpoint_loss_mode = endpoint_loss_mode
        cur_relation_aux_weight = relation_aux_weight
        cur_endpoint_aux_weight = endpoint_aux_weight
        cur_metric_align_aux_weight = metric_align_aux_weight
        cur_halt_aux_weight = halt_aux_weight
        if use_phase2:
            if phase2_endpoint_loss_mode:
                cur_endpoint_loss_mode = phase2_endpoint_loss_mode
            if phase2_relation_aux_weight is not None:
                cur_relation_aux_weight = phase2_relation_aux_weight
            if phase2_endpoint_aux_weight is not None:
                cur_endpoint_aux_weight = phase2_endpoint_aux_weight
            if phase2_metric_align_aux_weight is not None:
                cur_metric_align_aux_weight = phase2_metric_align_aux_weight
            if phase2_halt_aux_weight is not None:
                cur_halt_aux_weight = phase2_halt_aux_weight
        cur_endpoint_main_mode = cur_endpoint_loss_mode in endpoint_main_modes
        cur_entity_dist_main_mode = cur_endpoint_loss_mode in entity_dist_main_modes
        cur_metric_align_main_mode = cur_endpoint_loss_mode in metric_align_main_modes
        epoch_rl_scst_mode = cur_endpoint_loss_mode in rl_scst_modes
        if train_acc_mode == 'auto':
            epoch_acc_mode = 'rl_reward' if epoch_rl_scst_mode else 'endpoint_proxy'
        else:
            epoch_acc_mode = train_acc_mode
        cur_endpoint_enabled = cur_endpoint_main_mode or (cur_endpoint_aux_weight > 0.0)
        cur_metric_align_enabled = cur_metric_align_main_mode or (cur_metric_align_aux_weight > 0.0)
        if is_main and (ep == 1 or last_phase2_state is None or bool(use_phase2) != bool(last_phase2_state)):
            print(
                "[TrainObjective] "
                f"ep={ep} mode={cur_endpoint_loss_mode} "
                f"rel_aux={cur_relation_aux_weight:.4f} "
                f"endpoint_aux={cur_endpoint_aux_weight:.4f} "
                f"metric_align_aux={cur_metric_align_aux_weight:.4f} "
                f"halt_aux={cur_halt_aux_weight:.4f} "
                f"train_acc_mode={epoch_acc_mode}"
            )
        last_phase2_state = bool(use_phase2)
        progress_single_line = _as_bool(getattr(args, 'progress_single_line', False), default=False)
        progress_log_every = max(1, int(getattr(args, 'progress_log_every', 50)))
        progress_mininterval = max(0.5, float(getattr(args, 'progress_mininterval', 1.0)))
        num_batches = len(loader)
        pbar = loader if progress_single_line else tqdm(
            loader,
            disable=not is_main,
            desc=f'Ep {ep}',
            dynamic_ncols=True,
            mininterval=progress_mininterval,
        )
        sanity_interval_steps = 0
        if train_sanity_eval_every_pct > 0:
            sanity_interval_steps = max(1, int(math.ceil((float(len(loader)) * float(train_sanity_eval_every_pct)) / 100.0)))
        tot_loss = 0.0
        tot_metric_sum = 0.0
        tot_metric_count = 0
        tot_metric_f1_sum = 0.0
        tot_rel_correct = 0
        tot_rel_count = 0
        steps = 0
        last_progress_chars = 0
        for batch in pbar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attn = batch['attention_mask'].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            bl = torch.zeros((), device=device)
            avg_reward = 0.0
            avg_greedy = 0.0
            avg_adv = 0.0
            cur_rl_scst_mode = cur_endpoint_loss_mode in rl_scst_modes
            if cur_rl_scst_mode:
                B = int(input_ids.shape[0])
                losses = []
                sample_rewards = []
                greedy_rewards = []
                advantages = []
                q_embs = batch['rl_q_embs'].cpu().numpy()
                rl_starts = batch['rl_start_nodes']
                rl_gold = batch['rl_gold_answers']
                rl_tuples = batch['rl_tuples']
                for i in range(B):
                    adj_i = build_adj_from_tuples(rl_tuples[i])
                    gold_set_i = set(int(x) for x in rl_gold[i])
                    q_emb_i = np.asarray(q_embs[i], dtype=np.float32)
                    in_i = input_ids[i:i + 1]
                    attn_i = attn[i:i + 1]
                    logp_s, ent_s, pred_s, taken_s = _rollout_single_policy(
                        input_ids_1=in_i,
                        attn_1=attn_i,
                        adj=adj_i,
                        start_node=int(rl_starts[i]),
                        q_emb_np=q_emb_i,
                        sample_action=True,
                        no_cycle=phase2_rl_no_cycle,
                        temperature=phase2_rl_sample_temp,
                        with_grad=True,
                        query_residual_enabled=query_residual_enabled,
                        query_residual_alpha=query_residual_alpha,
                        query_residual_mode=query_residual_mode,
                    )
                    r_s = _rl_reward_from_pred(pred_s, gold_set_i, phase2_rl_reward_metric)
                    if phase2_rl_use_greedy_baseline:
                        _, _, pred_g, _ = _rollout_single_policy(
                            input_ids_1=in_i,
                            attn_1=attn_i,
                            adj=adj_i,
                            start_node=int(rl_starts[i]),
                            q_emb_np=q_emb_i,
                            sample_action=False,
                            no_cycle=phase2_rl_no_cycle,
                            temperature=1.0,
                            with_grad=False,
                            query_residual_enabled=query_residual_enabled,
                            query_residual_alpha=query_residual_alpha,
                            query_residual_mode=query_residual_mode,
                        )
                        r_g = _rl_reward_from_pred(pred_g, gold_set_i, phase2_rl_reward_metric)
                    else:
                        r_g = 0.0
                    adv = float(r_s - r_g)
                    if phase2_rl_adv_clip is not None:
                        clip_v = float(abs(phase2_rl_adv_clip))
                        adv = float(max(-clip_v, min(clip_v, adv)))
                    loss_i = (-adv) * logp_s
                    if phase2_rl_entropy_weight > 0.0 and taken_s > 0:
                        loss_i = loss_i - float(phase2_rl_entropy_weight) * (ent_s / float(max(1, taken_s)))
                    losses.append(loss_i)
                    sample_rewards.append(float(r_s))
                    greedy_rewards.append(float(r_g))
                    advantages.append(float(adv))
                if losses:
                    bl = torch.stack(losses).mean()
                    avg_reward = float(sum(sample_rewards) / max(1, len(sample_rewards)))
                    avg_greedy = float(sum(greedy_rewards) / max(1, len(greedy_rewards)))
                    avg_adv = float(sum(advantages) / max(1, len(advantages)))
                    tot_metric_sum += avg_reward * float(len(sample_rewards))
                    tot_metric_count += int(len(sample_rewards))
                    tot_metric_f1_sum += avg_reward * float(len(sample_rewards))
                else:
                    avg_reward = 0.0
                    avg_greedy = 0.0
                    avg_adv = 0.0
            else:
                carry = carry_init_fn({'input_ids': input_ids}, device)
                T = len(batch['seq_batches'])
                need_endpoint_proxy_metric = (epoch_acc_mode == 'endpoint_proxy')
                entity_step_cache = []
                for t, step in enumerate(batch['seq_batches']):
                    p_ids = step['puzzle_identifiers'].to(device)
                    r_ids = step['relation_identifiers'].to(device)
                    c_mask = step['candidate_mask'].to(device)
                    endpoint_t = step['endpoint_targets'].to(device)
                    policy_t = step['policy_targets'].to(device)
                    halt_t = step['halt_targets'].to(device)
                    halt_m = step['halt_mask'].to(device)
                    labels = step['labels'].to(device)
                    v_mask = step['valid_mask'].to(device)
                    carry, out = model(carry, {'input_ids': input_ids, 'attention_mask': attn, 'puzzle_identifiers': p_ids, 'relation_identifiers': r_ids, 'candidate_mask': c_mask})
                    logits = out['scores'].masked_fill(~c_mask, -1e4)
                    if cur_entity_dist_main_mode or need_endpoint_proxy_metric:
                        probs = torch.softmax(logits, dim=1)
                        entity_step_cache.append(
                            {
                                'probs': probs,
                                'r_ids': r_ids,
                                'c_mask': c_mask,
                                'v_mask': v_mask,
                            }
                        )
                    lv = ce(logits, labels).masked_fill(~v_mask, 0.0)
                    valid = v_mask & (labels >= 0)
                    if valid.any():
                        pred = torch.argmax(logits, dim=1)
                        tot_rel_correct += int((pred[valid] == labels[valid]).sum().item())
                        tot_rel_count += int(valid.sum().item())
                    sc = v_mask.sum().clamp(min=1)
                    ce_loss = lv.sum() / sc
                    endpoint_loss = torch.zeros((), device=device)
                    if cur_endpoint_enabled:
                        ev = bce_endpoint(logits, endpoint_t).masked_fill(~c_mask, 0.0)
                        ev = ev.sum(dim=1) / c_mask.sum(dim=1).clamp(min=1).float()
                        ev = ev.masked_fill(~v_mask, 0.0)
                        endpoint_loss = ev.sum() / sc
                    policy_loss = torch.zeros((), device=device)
                    if cur_metric_align_enabled:
                        # Align train target with final answer reachability:
                        # maximize probability mass on relations that reduce shortest distance to gold.
                        logp = torch.log_softmax(logits, dim=1)
                        pos_mass = (torch.exp(logp) * policy_t).sum(dim=1)
                        policy_valid = v_mask & (policy_t.sum(dim=1) > 0.0)
                        pv = (-torch.log(pos_mass.clamp(min=1e-8))).masked_fill(~policy_valid, 0.0)
                        policy_sc = policy_valid.sum().clamp(min=1).float()
                        policy_loss = pv.sum() / policy_sc

                    if cur_metric_align_main_mode:
                        step_loss = policy_loss
                        if cur_relation_aux_weight > 0.0:
                            step_loss = step_loss + cur_relation_aux_weight * ce_loss
                    elif cur_entity_dist_main_mode:
                        step_loss = torch.zeros((), device=device)
                        if cur_relation_aux_weight > 0.0:
                            step_loss = step_loss + cur_relation_aux_weight * ce_loss
                    elif cur_endpoint_main_mode:
                        step_loss = endpoint_loss
                        if cur_relation_aux_weight > 0.0:
                            step_loss = step_loss + cur_relation_aux_weight * ce_loss
                    else:
                        step_loss = ce_loss
                    if (not cur_endpoint_main_mode) and cur_endpoint_aux_weight > 0.0:
                        step_loss = step_loss + cur_endpoint_aux_weight * endpoint_loss
                    if (not cur_metric_align_main_mode) and cur_metric_align_aux_weight > 0.0:
                        step_loss = step_loss + cur_metric_align_aux_weight * policy_loss
                    if cur_halt_aux_weight > 0.0:
                        halt_pair = torch.stack(
                            [out['q_continue_logits'].to(torch.float32), out['q_halt_logits'].to(torch.float32)],
                            dim=1,
                        )
                        halt_labels = halt_t.to(torch.long)
                        halt_vec = ce_halt(halt_pair, halt_labels).masked_fill(~halt_m, 0.0)
                        halt_l = halt_vec.sum() / halt_m.sum().clamp(min=1).float()
                        step_loss = step_loss + cur_halt_aux_weight * halt_l
                    bl += ((t + 1) / T) * step_loss
                if need_endpoint_proxy_metric and entity_step_cache:
                    with torch.no_grad():
                        _, proxy_hit_sum, proxy_f1_sum, proxy_n = _entity_dist_loss_from_steps(
                            step_cache=entity_step_cache,
                            tuples_list=batch['rl_tuples'],
                            starts_list=batch['rl_start_nodes'],
                            golds_list=batch['rl_gold_answers'],
                            compute_loss=False,
                            return_stats=True,
                        )
                    if proxy_n > 0:
                        tot_metric_sum += float(proxy_hit_sum)
                        tot_metric_f1_sum += float(proxy_f1_sum)
                        tot_metric_count += int(proxy_n)
                if cur_entity_dist_main_mode:
                    entity_dist_loss = _entity_dist_loss_from_steps(
                        step_cache=entity_step_cache,
                        tuples_list=batch['rl_tuples'],
                        starts_list=batch['rl_start_nodes'],
                        golds_list=batch['rl_gold_answers'],
                        compute_loss=True,
                        return_stats=False,
                    )
                    bl = bl + entity_dist_loss
            if not torch.isfinite(bl):
                if is_ddp:
                    raise RuntimeError('non-finite loss in DDP')
                continue
            bl.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += bl.item()
            steps += 1
            should_update_progress = (
                steps == 1
                or steps == num_batches
                or (progress_log_every > 0 and (steps % progress_log_every) == 0)
            )
            if is_main:
                rel_acc = 100.0 * (tot_rel_correct / max(1, tot_rel_count))
                endpoint_hit1_proxy = 100.0 * (tot_metric_sum / max(1, tot_metric_count))
                endpoint_f1_proxy = 100.0 * (tot_metric_f1_sum / max(1, tot_metric_count))
                if epoch_acc_mode == 'relation':
                    acc = rel_acc
                    acc_label = 'rel_acc'
                elif epoch_acc_mode == 'rl_reward':
                    acc = endpoint_hit1_proxy
                    acc_label = 'rl_reward'
                else:
                    # endpoint_proxy
                    if tot_metric_count > 0:
                        acc = endpoint_hit1_proxy
                        acc_label = 'endpoint_hit1'
                    else:
                        acc = rel_acc
                        acc_label = 'rel_acc_fallback'
                if should_update_progress and not progress_single_line:
                    if cur_rl_scst_mode:
                        pbar.set_postfix_str(
                            f'[step {steps}] loss={bl.item():.4f} avg={tot_loss/max(1,steps):.4f} '
                            f'reward={avg_reward:.4f} adv={avg_adv:.4f} acc({acc_label})={acc:.2f}% grad={float(grad_norm):.2e}'
                        )
                    else:
                        pbar.set_postfix_str(
                            f'[step {steps}] loss={bl.item():.4f} avg={tot_loss/max(1,steps):.4f} '
                            f'acc({acc_label})={acc:.2f}% rel_acc={rel_acc:.2f}% ep_f1_proxy={endpoint_f1_proxy:.2f}% grad={float(grad_norm):.2e}'
                        )
                if should_update_progress and progress_single_line:
                    if cur_rl_scst_mode:
                        msg = (
                            f"Ep {ep} {steps}/{num_batches} ({(100.0 * steps) / max(1, num_batches):.1f}%) "
                            f"loss={bl.item():.4f} avg={tot_loss/max(1,steps):.4f} "
                            f"reward={avg_reward:.4f} adv={avg_adv:.4f} "
                            f"acc({acc_label})={acc:.2f}% grad={float(grad_norm):.2e}"
                        )
                    else:
                        msg = (
                            f"Ep {ep} {steps}/{num_batches} ({(100.0 * steps) / max(1, num_batches):.1f}%) "
                            f"loss={bl.item():.4f} avg={tot_loss/max(1,steps):.4f} "
                            f"acc({acc_label})={acc:.2f}% rel_acc={rel_acc:.2f}% "
                            f"ep_f1_proxy={endpoint_f1_proxy:.2f}% grad={float(grad_norm):.2e}"
                        )
                    last_progress_chars = _progress_write_line(msg, last_progress_chars)
                if wb is not None:
                    wb_payload = {
                        "train/step_loss": float(bl.item()),
                        "train/step_avg_loss": float(tot_loss / max(1, steps)),
                        "train/step_acc": float(acc),
                        "train/step_rel_acc": float(rel_acc),
                        "train/step_endpoint_hit1_proxy": float(endpoint_hit1_proxy),
                        "train/step_endpoint_f1_proxy": float(endpoint_f1_proxy),
                        "train/grad_norm": float(grad_norm),
                        "train/epoch": int(ep),
                        "train/step": int(steps),
                    }
                    if cur_rl_scst_mode:
                        wb_payload.update(
                            {
                                "train/rl_reward_sample": float(avg_reward),
                                "train/rl_reward_greedy": float(avg_greedy),
                                "train/rl_advantage": float(avg_adv),
                            }
                        )
                    wb.log(
                        wb_payload,
                        step=(ep - 1) * max(1, len(loader)) + steps,
                    )
            should_sanity_eval = (
                train_sanity_eval_every_pct > 0
                and train_sanity_eval_limit > 0
                and sanity_interval_steps > 0
                and (steps % sanity_interval_steps == 0)
            )
            if should_sanity_eval:
                if is_ddp:
                    dist.barrier()
                if is_main:
                    eval_model = model.module if hasattr(model, 'module') else model
                    sh, sf, ss = evaluate_relation_beam(
                        model=eval_model,
                        carry_init_fn=carry_init_fn,
                        eval_json=args.train_json,
                        kb2idx=kb2idx,
                        rel2idx=rel2idx,
                        device=device,
                        tokenizer_name=args.trm_tokenizer,
                        q_npy=getattr(args, 'query_emb_train_npy', ''),
                        rel_npy=args.relation_emb_npy,
                        max_steps=int(getattr(args, 'eval_max_steps', args.max_steps)),
                        max_neighbors=int(train_sanity_eval_max_neighbors),
                        prune_keep=int(train_sanity_eval_prune_keep),
                        start_topk=int(max(1, train_sanity_eval_start_topk)),
                        beam=int(max(1, train_sanity_eval_beam)),
                        max_q_len=args.max_q_len,
                        eval_limit=int(train_sanity_eval_limit),
                        debug_n=0,
                        no_cycle=bool(train_sanity_eval_no_cycle),
                        eval_pred_topk=int(max(1, train_sanity_eval_pred_topk)),
                        eval_use_halt=bool(train_sanity_eval_use_halt),
                        eval_min_hops_before_stop=int(getattr(args, 'eval_min_hops_before_stop', 1)),
                        query_residual_enabled=bool(query_residual_enabled),
                        query_residual_alpha=float(query_residual_alpha),
                        query_residual_mode=str(query_residual_mode),
                        entity_labels=None,
                        relation_labels=None,
                        eval_random_sample_size=0,
                        eval_random_seed=42,
                    )
                    print(
                        f"[TrainSanity] ep={ep} step={steps}/{len(loader)} "
                        f"(~{int((100.0*steps)/max(1,len(loader)))}%) "
                        f"Hit@1={sh:.4f} F1={sf:.4f} Skip={ss} "
                        f"beam={int(max(1, train_sanity_eval_beam))} n={int(train_sanity_eval_limit)}"
                    )
                    if wb is not None:
                        wb.log(
                            {
                                "train_sanity/hit1": float(sh),
                                "train_sanity/f1": float(sf),
                                "train_sanity/skip": int(ss),
                                "train_sanity/epoch": int(ep),
                                "train_sanity/step": int(steps),
                            },
                            step=(ep - 1) * max(1, len(loader)) + steps,
                        )
                if is_ddp:
                    dist.barrier()
                model.train()

        epoch_tot_loss = float(tot_loss)
        epoch_steps = int(steps)
        epoch_tot_metric_sum = float(tot_metric_sum)
        epoch_tot_metric_count = int(tot_metric_count)
        epoch_tot_metric_f1_sum = float(tot_metric_f1_sum)
        epoch_tot_rel_correct = int(tot_rel_correct)
        epoch_tot_rel_count = int(tot_rel_count)
        if is_main and progress_single_line:
            _progress_finish_line(last_progress_chars)
        if is_ddp:
            # Aggregate epoch metrics across all ranks so train_acc is not rank0-local.
            epoch_reduce = torch.tensor(
                [
                    float(tot_loss),
                    float(steps),
                    float(tot_metric_sum),
                    float(tot_metric_count),
                    float(tot_metric_f1_sum),
                    float(tot_rel_correct),
                    float(tot_rel_count),
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(epoch_reduce, op=dist.ReduceOp.SUM)
            epoch_tot_loss = float(epoch_reduce[0].item())
            epoch_steps = int(round(float(epoch_reduce[1].item())))
            epoch_tot_metric_sum = float(epoch_reduce[2].item())
            epoch_tot_metric_count = int(round(float(epoch_reduce[3].item())))
            epoch_tot_metric_f1_sum = float(epoch_reduce[4].item())
            epoch_tot_rel_correct = int(round(float(epoch_reduce[5].item())))
            epoch_tot_rel_count = int(round(float(epoch_reduce[6].item())))
            dist.barrier()
        if is_main:
            save_obj = model.module if hasattr(model, 'module') else model
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt = os.path.join(args.out_dir, f'model_ep{ep}.pt')
            torch.save(save_obj.state_dict(), ckpt)
            print(f'Saved {ckpt}')
            dev_hit1 = None
            dev_f1 = None
            eval_every = max(1, int(getattr(args, 'eval_every_epochs', 1)))
            eval_start = max(1, int(getattr(args, 'eval_start_epoch', 1)))
            should_eval = ep >= eval_start and ((ep - eval_start) % eval_every == 0)
            if getattr(args, 'dev_json', '') and should_eval:
                # Dev evaluation uses the same endpoint traversal metric as test:
                # start-entity -> predicted end-entity, measured by Hit@1/F1.
                mh, mf, sk = evaluate_relation_beam(
                    model=save_obj,
                    carry_init_fn=carry_init_fn,
                    eval_json=args.dev_json,
                    kb2idx=kb2idx,
                    rel2idx=rel2idx,
                    device=device,
                    tokenizer_name=args.trm_tokenizer,
                    q_npy=getattr(args, 'query_emb_dev_npy', ''),
                    rel_npy=args.relation_emb_npy,
                    max_steps=int(getattr(args, 'eval_max_steps', args.max_steps)),
                    max_neighbors=int(getattr(args, 'eval_max_neighbors', args.max_neighbors)),
                    prune_keep=int(getattr(args, 'eval_prune_keep', args.prune_keep)),
                    start_topk=int(getattr(args, 'eval_start_topk', getattr(args, 'start_topk', 5))),
                    beam=int(getattr(args, 'eval_beam', getattr(args, 'beam', 5))),
                    max_q_len=args.max_q_len,
                    eval_limit=getattr(args, 'eval_limit', -1),
                    debug_n=getattr(args, 'debug_eval_n', 0),
                    no_cycle=bool(getattr(args, 'eval_no_cycle', False)),
                    eval_pred_topk=int(getattr(args, 'eval_pred_topk', 1)),
                    eval_use_halt=bool(getattr(args, 'eval_use_halt', False)),
                    eval_min_hops_before_stop=int(getattr(args, 'eval_min_hops_before_stop', 1)),
                    query_residual_enabled=bool(query_residual_enabled),
                    query_residual_alpha=float(query_residual_alpha),
                    query_residual_mode=str(query_residual_mode),
                    entity_labels=entity_labels,
                    relation_labels=relation_labels,
                    eval_random_sample_size=int(eval_random_sample_size),
                    eval_random_seed=int(
                        eval_random_seed + (ep if eval_random_resample_each_eval else 0)
                    ),
                )
                print(f'[Dev] Hit@1={mh:.4f} F1={mf:.4f} Skip={sk}')
                dev_hit1 = float(mh)
                dev_f1 = float(mf)
                if wb is not None:
                    wb.log(
                        {
                            "dev/hit1": float(mh),
                            "dev/f1": float(mf),
                            "dev/skip": int(sk),
                            "train/epoch": int(ep),
                        },
                        step=ep * max(1, len(loader)),
                    )
            elif getattr(args, 'dev_json', ''):
                print(f'[Dev] skip eval at ep{ep} (start={eval_start}, every={eval_every})')
            epoch_avg_loss = float(epoch_tot_loss / max(1, epoch_steps))
            epoch_rel_acc_ratio = float(epoch_tot_rel_correct / max(1, epoch_tot_rel_count))
            epoch_endpoint_hit_ratio = float(epoch_tot_metric_sum / max(1, epoch_tot_metric_count))
            epoch_endpoint_f1_ratio = float(epoch_tot_metric_f1_sum / max(1, epoch_tot_metric_count))
            if epoch_acc_mode == 'relation':
                epoch_acc_ratio = epoch_rel_acc_ratio
            elif epoch_acc_mode == 'rl_reward':
                epoch_acc_ratio = epoch_endpoint_hit_ratio
            else:
                epoch_acc_ratio = epoch_endpoint_hit_ratio if epoch_tot_metric_count > 0 else epoch_rel_acc_ratio
            if wb is not None:
                wb.log(
                    {
                        "train/epoch_avg_loss": float(epoch_avg_loss),
                        "train/epoch_acc": float(100.0 * epoch_acc_ratio),
                        "train/epoch_rel_acc": float(100.0 * epoch_rel_acc_ratio),
                        "train/epoch_endpoint_hit1_proxy": float(100.0 * epoch_endpoint_hit_ratio),
                        "train/epoch_endpoint_f1_proxy": float(100.0 * epoch_endpoint_f1_ratio),
                        "train/phase2_active": int(1 if use_phase2 else 0),
                        "train/epoch": int(ep),
                    },
                    step=ep * max(1, len(loader)),
                )
            metric_value = None
            maximize = True
            if phase2_auto_metric == 'dev_f1':
                metric_value = dev_f1
                maximize = True
            elif phase2_auto_metric == 'dev_hit1':
                metric_value = dev_hit1
                maximize = True
            elif phase2_auto_metric == 'train_acc':
                metric_value = epoch_acc_ratio
                maximize = True
            elif phase2_auto_metric == 'train_loss':
                metric_value = epoch_avg_loss
                maximize = False

            if phase2_auto_enabled and (not phase2_auto_active) and ep >= max(1, phase2_auto_min_epoch):
                trigger_reason = None
                if (phase2_auto_threshold is not None) and (metric_value is not None):
                    if maximize and metric_value >= phase2_auto_threshold:
                        trigger_reason = f"{phase2_auto_metric}={metric_value:.4f} >= {phase2_auto_threshold:.4f}"
                    elif (not maximize) and metric_value <= phase2_auto_threshold:
                        trigger_reason = f"{phase2_auto_metric}={metric_value:.4f} <= {phase2_auto_threshold:.4f}"
                if (trigger_reason is None) and phase2_auto_patience > 0 and (metric_value is not None):
                    improved = False
                    if phase2_auto_best is None:
                        improved = True
                    elif maximize:
                        improved = metric_value > float(phase2_auto_best) + phase2_auto_min_delta
                    else:
                        improved = metric_value < float(phase2_auto_best) - phase2_auto_min_delta
                    if improved:
                        phase2_auto_best = metric_value
                        phase2_auto_bad_count = 0
                    else:
                        phase2_auto_bad_count += 1
                        if phase2_auto_bad_count >= phase2_auto_patience:
                            trigger_reason = (
                                f"plateau: {phase2_auto_metric} no improvement for {phase2_auto_bad_count} evals"
                            )
                if trigger_reason is not None:
                    phase2_auto_active = True
                    print(f"[Phase2AutoSwitch] trigger at ep{ep}: {trigger_reason} (effective next epoch)")
                    if wb is not None:
                        wb.log(
                            {
                                "train/phase2_auto_trigger_epoch": int(ep),
                                "train/phase2_auto_metric_value": float(metric_value) if metric_value is not None else float('nan'),
                                "train/phase2_active": 1,
                            },
                            step=ep * max(1, len(loader)),
                        )
        if is_ddp:
            phase2_tensor = torch.tensor(
                [1 if (phase2_auto_active or (phase2_start_epoch > 0 and ep >= phase2_start_epoch)) else 0],
                dtype=torch.int64,
                device=device,
            )
            dist.broadcast(phase2_tensor, src=0)
            phase2_auto_active = bool(int(phase2_tensor.item()))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Keep all ranks in sync while rank0 runs (potentially long) dev eval.
        if is_ddp:
            dist.barrier()

    if is_ddp:
        dist.destroy_process_group()
    if wb is not None:
        wb.finish()


def test(args):
    subgraph_reader_enabled = str(getattr(args, "subgraph_reader_enabled", False)).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if subgraph_reader_enabled:
        from .subgraph_reader import test_subgraph_reader

        test_subgraph_reader(args)
        return
    raise RuntimeError(
        "This workspace is configured for subgraph-reader-only testing "
        "(D / D+latent). Set subgraph_reader_enabled=true."
    )

    # Full relation-path beam traversal evaluation:
    # start-entity -> predicted end-entity, measured by Hit@1/F1.
    tok = load_tokenizer(args.trm_tokenizer)
    ent_mem = np.load(args.entity_emb_npy, mmap_mode='r')
    rel_mem = np.load(args.relation_emb_npy, mmap_mode='r')
    sd = torch.load(args.ckpt, map_location='cpu')
    ckpt_vocab_size = _infer_vocab_size_from_state_dict(sd)
    vocab_size = int(tok.vocab_size)
    if ckpt_vocab_size is not None and int(ckpt_vocab_size) > 0 and int(ckpt_vocab_size) != int(vocab_size):
        print(
            f"[warn] tokenizer vocab_size({vocab_size}) != ckpt vocab_size({int(ckpt_vocab_size)}); "
            "using ckpt vocab_size for model build."
        )
        vocab_size = int(ckpt_vocab_size)
    kb2idx = load_kb_map(args.entities_txt)
    rel2idx = load_rel_map(args.relations_txt)
    entity_labels = _load_entity_labels(args.entities_txt, getattr(args, 'entity_names_json', ''))
    relation_labels = _load_relation_labels(args.relations_txt)
    cfg = {
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'vocab_size': vocab_size,
        'hidden_size': args.hidden_size,
        'num_heads': args.num_heads,
        'expansion': args.expansion,
        'H_cycles': args.H_cycles,
        'L_cycles': args.L_cycles,
        'L_layers': args.L_layers,
        'H_layers': 0,
        'puzzle_emb_ndim': int(ent_mem.shape[1]),
        'relation_emb_ndim': int(rel_mem.shape[1]),
        'puzzle_emb_len': args.puzzle_emb_len,
        'pos_encodings': args.pos_encodings,
        'forward_dtype': args.forward_dtype,
        'halt_max_steps': args.halt_max_steps,
        'halt_exploration_prob': args.halt_exploration_prob,
        'no_ACT_continue': True,
        'mlp_t': False,
        'num_puzzle_identifiers': len(kb2idx) + 1000,
        'num_relation_identifiers': len(rel2idx) + 1000,
    }
    model, _ = build_model(args.model_impl, args.trm_root, cfg)
    model.inner.puzzle_emb_data = ent_mem
    model.inner.relation_emb_data = rel_mem
    model.load_state_dict(sd, strict=False)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dev = next(model.parameters()).device

    carry_cls = type(model.initial_carry({'input_ids': torch.ones((1, 1), dtype=torch.long, device=dev)}))

    def carry_init_fn(batch, device):
        B = int(batch['input_ids'].shape[0])
        inner = model.inner
        return carry_cls(inner.empty_carry(B), torch.zeros(B, device=device), torch.ones(B, dtype=torch.bool, device=device), {})

    print("[ok] checkpoint loaded:", args.ckpt)
    if getattr(args, 'eval_json', ''):
        mh, mf, sk = evaluate_relation_beam(
            model=model,
            carry_init_fn=carry_init_fn,
            eval_json=args.eval_json,
            kb2idx=kb2idx,
            rel2idx=rel2idx,
            device=dev,
            tokenizer_name=args.trm_tokenizer,
            q_npy=getattr(args, 'query_emb_eval_npy', ''),
            rel_npy=args.relation_emb_npy,
            max_steps=int(getattr(args, 'eval_max_steps', getattr(args, 'max_steps', 4))),
            max_neighbors=int(getattr(args, 'eval_max_neighbors', getattr(args, 'max_neighbors', 256))),
            prune_keep=int(getattr(args, 'eval_prune_keep', getattr(args, 'prune_keep', 64))),
            start_topk=int(getattr(args, 'eval_start_topk', getattr(args, 'start_topk', 5))),
            beam=int(getattr(args, 'eval_beam', getattr(args, 'beam', 5))),
            max_q_len=getattr(args, 'max_q_len', 512),
            eval_limit=getattr(args, 'eval_limit', -1),
            debug_n=getattr(args, 'debug_eval_n', 5),
            no_cycle=bool(getattr(args, 'eval_no_cycle', False)),
            eval_pred_topk=int(getattr(args, 'eval_pred_topk', 1)),
            eval_use_halt=bool(getattr(args, 'eval_use_halt', False)),
            eval_min_hops_before_stop=int(getattr(args, 'eval_min_hops_before_stop', 1)),
            query_residual_enabled=bool(getattr(args, 'query_residual_enabled', False)),
            query_residual_alpha=float(getattr(args, 'query_residual_alpha', 0.0)),
            query_residual_mode=str(getattr(args, 'query_residual_mode', 'sub_rel')),
            entity_labels=entity_labels,
            relation_labels=relation_labels,
            eval_dump_jsonl=str(getattr(args, 'eval_dump_jsonl', '')),
            eval_random_sample_size=int(getattr(args, 'eval_random_sample_size', 0)),
            eval_random_seed=int(getattr(args, 'eval_random_seed', 42)),
        )
        print(f'[Test] Hit@1={mh:.4f} F1={mf:.4f} Skip={sk}')
