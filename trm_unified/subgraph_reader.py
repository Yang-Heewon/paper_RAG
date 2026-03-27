import math
import os
import re
import json
import sys
from collections import deque
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .data import build_line_offsets, load_rel_map, read_jsonl_by_offset


def _as_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


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


def _coerce_int_list(values) -> List[int]:
    out = []
    for x in values or []:
        try:
            out.append(int(x))
        except Exception:
            continue
    return out


def _parse_tuples(raw_tuples, rel2idx: Optional[Dict[str, int]] = None) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for tri in raw_tuples or []:
        if not isinstance(tri, (list, tuple)) or len(tri) != 3:
            continue
        s, r, o = tri
        try:
            ss = int(s)
            oo = int(o)
        except Exception:
            continue
        if isinstance(r, int):
            rr = int(r)
        elif isinstance(r, str) and rel2idx is not None and r in rel2idx:
            rr = int(rel2idx[r])
        else:
            continue
        out.append((ss, rr, oo))
    return out


def _extract_seed_entities(ex: dict, tuples: List[Tuple[int, int, int]]) -> List[int]:
    starts = _coerce_int_list(ex.get("entities_cid", []))
    if not starts:
        starts = _coerce_int_list(ex.get("entities", []))
    if not starts and tuples:
        starts = [int(tuples[0][0])]
    return list(dict.fromkeys(starts))


def _extract_gold_answers(ex: dict) -> List[int]:
    return list(dict.fromkeys(_coerce_int_list(ex.get("answers_cid", []))))


def _extract_candidate_entities(ex: dict) -> List[int]:
    return list(dict.fromkeys(_coerce_int_list(ex.get("candidate_cid", []))))


def _build_khop_subgraph(
    tuples: List[Tuple[int, int, int]],
    seed_nodes: Sequence[int],
    hops: int,
    max_nodes: int,
    max_edges: int,
    add_reverse_edges: bool,
    split_reverse_relations: bool = False,
    relation_offset: int = 0,
) -> Tuple[List[int], List[Tuple[int, int, int]]]:
    hops = max(0, int(hops))
    max_nodes = max(1, int(max_nodes))
    max_edges = max(0, int(max_edges))

    if not tuples:
        if seed_nodes:
            return [int(seed_nodes[0])], []
        return [0], []

    all_edges: List[Tuple[int, int, int]] = []
    adj: Dict[int, List[Tuple[int, int]]] = {}
    for s, r, o in tuples:
        ss = int(s)
        rr = int(r)
        oo = int(o)
        all_edges.append((ss, rr, oo))
        adj.setdefault(ss, []).append((rr, oo))
        if add_reverse_edges:
            rev_rr = int(rr + relation_offset) if split_reverse_relations else rr
            all_edges.append((oo, rev_rr, ss))
            adj.setdefault(oo, []).append((rev_rr, ss))

    seeds = [int(x) for x in seed_nodes if isinstance(x, (int, np.integer))]
    if not seeds:
        seeds = [int(tuples[0][0])]

    visited = set()
    node_order: List[int] = []
    frontier: List[int] = []
    for s in seeds:
        if s in visited:
            continue
        visited.add(s)
        node_order.append(s)
        frontier.append(s)
        if len(node_order) >= max_nodes:
            break

    traversed: List[Tuple[int, int, int]] = []
    traversed_set = set()
    for _ in range(hops):
        if not frontier:
            break
        new_frontier: List[int] = []
        for u in frontier:
            for rr, vv in adj.get(int(u), []):
                tri = (int(u), int(rr), int(vv))
                if tri not in traversed_set and (max_edges <= 0 or len(traversed) < max_edges):
                    traversed.append(tri)
                    traversed_set.add(tri)
                if vv in visited:
                    continue
                if len(node_order) >= max_nodes:
                    continue
                visited.add(int(vv))
                node_order.append(int(vv))
                new_frontier.append(int(vv))
        frontier = new_frontier

    if max_edges > 0 and len(traversed) < max_edges:
        for s, r, o in all_edges:
            if s not in visited or o not in visited:
                continue
            tri = (int(s), int(r), int(o))
            if tri in traversed_set:
                continue
            traversed.append(tri)
            traversed_set.add(tri)
            if len(traversed) >= max_edges:
                break

    if not node_order:
        node_order = [int(seeds[0])] if seeds else [0]

    node2idx = {n: i for i, n in enumerate(node_order)}
    edges_idx: List[Tuple[int, int, int]] = []
    for s, r, o in traversed:
        if s not in node2idx or o not in node2idx:
            continue
        edges_idx.append((node2idx[s], node2idx[o], int(r)))

    return node_order, edges_idx


class SubgraphExampleDataset(Dataset):
    def __init__(self, jsonl_path: str):
        if not jsonl_path:
            raise RuntimeError("subgraph reader requires a jsonl path.")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"jsonl not found: {jsonl_path}")
        self.jsonl_path = jsonl_path
        self.offsets = build_line_offsets(jsonl_path, is_main=True)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict:
        ex = read_jsonl_by_offset(self.jsonl_path, self.offsets, idx)
        return {
            "orig_id": ex.get("orig_id", ex.get("id", str(idx))),
            "question": ex.get("question", ""),
            "tuples": ex.get("subgraph", {}).get("tuples", []),
            "entities": ex.get("entities", []),
            "entities_cid": ex.get("entities_cid", []),
            "answers_cid": ex.get("answers_cid", []),
            "candidate_cid": ex.get("candidate_cid", []),
            "ex_line": idx,
        }


class SubgraphCollator:
    def __init__(
        self,
        entity_emb_npy: str,
        relation_emb_npy: str,
        query_emb_npy: str,
        rel2idx: Optional[Dict[str, int]],
        hops: int = 3,
        max_nodes: int = 256,
        max_edges: int = 2048,
        add_reverse_edges: bool = True,
        split_reverse_relations: bool = False,
    ):
        if not query_emb_npy:
            raise RuntimeError("query embedding path is empty for subgraph reader.")
        if not os.path.exists(query_emb_npy):
            raise FileNotFoundError(f"query embedding file not found: {query_emb_npy}")
        self.ent_mem = np.load(entity_emb_npy, mmap_mode="r")
        self.rel_mem = np.load(relation_emb_npy, mmap_mode="r")
        self.q_mem = np.load(query_emb_npy, mmap_mode="r")
        self.rel2idx = rel2idx
        self.hops = max(0, int(hops))
        self.max_nodes = max(1, int(max_nodes))
        self.max_edges = max(0, int(max_edges))
        self.add_reverse_edges = bool(add_reverse_edges)
        self.split_reverse_relations = bool(split_reverse_relations)
        self.entity_dim = int(self.ent_mem.shape[1])
        self.relation_dim = int(self.rel_mem.shape[1])
        self.query_dim = int(self.q_mem.shape[1])
        self.num_relations = int(self.rel_mem.shape[0])

    def _gather_entity_emb(self, node_ids: List[int]) -> np.ndarray:
        out = np.zeros((len(node_ids), self.entity_dim), dtype=np.float32)
        for i, nid in enumerate(node_ids):
            if 0 <= int(nid) < int(self.ent_mem.shape[0]):
                out[i] = np.asarray(self.ent_mem[int(nid)], dtype=np.float32)
        return out

    def _gather_relation_emb(self, rel_ids: List[int]) -> np.ndarray:
        out = np.zeros((len(rel_ids), self.relation_dim), dtype=np.float32)
        for i, rid in enumerate(rel_ids):
            if 0 <= int(rid) < int(self.rel_mem.shape[0]):
                out[i] = np.asarray(self.rel_mem[int(rid)], dtype=np.float32)
        return out

    def __call__(self, batch: List[dict]) -> dict:
        node_emb_list = []
        node_mask_list = []
        seed_mask_list = []
        node_label_list = []
        node_cids_list = []
        candidate_mask_list = []
        edge_src_list = []
        edge_dst_list = []
        edge_rel_emb_list = []
        edge_rel_id_list = []
        edge_dir_list = []
        edge_mask_list = []
        q_emb_list = []
        gold_list = []
        orig_ids = []

        max_n = 1
        max_e = 0
        for ex in batch:
            tuples = _parse_tuples(ex.get("tuples", []), self.rel2idx)
            seeds = _extract_seed_entities(ex, tuples)
            gold = _extract_gold_answers(ex)
            node_cids, edges_idx = _build_khop_subgraph(
                tuples=tuples,
                seed_nodes=seeds,
                hops=self.hops,
                max_nodes=self.max_nodes,
                max_edges=self.max_edges,
                add_reverse_edges=self.add_reverse_edges,
                split_reverse_relations=self.split_reverse_relations,
                relation_offset=self.num_relations,
            )
            if not node_cids:
                node_cids = [0]
            n = len(node_cids)
            e = len(edges_idx)
            max_n = max(max_n, n)
            max_e = max(max_e, e)

            node_emb = self._gather_entity_emb(node_cids)
            seed_set = set(int(x) for x in seeds)
            seed_mask = np.asarray([int(nid) in seed_set for nid in node_cids], dtype=np.bool_)
            # Fallback for malformed examples where no seed survives subgraph crop.
            if not bool(seed_mask.any()) and n > 0:
                seed_mask[0] = True
            gold_set = set(int(x) for x in gold)
            node_lbl = np.asarray([1.0 if int(nid) in gold_set else 0.0 for nid in node_cids], dtype=np.float32)
            candidate_ids = _extract_candidate_entities(ex)
            if candidate_ids:
                candidate_set = set(int(x) for x in candidate_ids)
                candidate_mask = np.asarray([int(nid) in candidate_set for nid in node_cids], dtype=np.bool_)
                if not bool(candidate_mask.any()):
                    candidate_mask = np.asarray([not is_seed for is_seed in seed_mask], dtype=np.bool_)
            else:
                candidate_mask = np.asarray([not is_seed for is_seed in seed_mask], dtype=np.bool_)
                if not bool(candidate_mask.any()):
                    candidate_mask = np.ones((n,), dtype=np.bool_)

            edge_src = np.zeros((e,), dtype=np.int64)
            edge_dst = np.zeros((e,), dtype=np.int64)
            edge_dir = np.zeros((e,), dtype=np.int64)
            edge_rel_ids = []
            for j, (sidx, didx, rid) in enumerate(edges_idx):
                edge_src[j] = int(sidx)
                edge_dst[j] = int(didx)
                rr = int(rid)
                if self.split_reverse_relations and rr >= self.num_relations:
                    rr = rr - self.num_relations
                    edge_dir[j] = 1
                rr = int(rr % max(1, self.num_relations))
                edge_rel_ids.append(rr)
            edge_rel_emb = self._gather_relation_emb(edge_rel_ids)

            qi = int(ex.get("ex_line", -1))
            if qi < 0 or qi >= int(self.q_mem.shape[0]):
                raise RuntimeError(
                    f"query embedding index out of range for subgraph reader: ex_line={qi}, "
                    f"q_rows={int(self.q_mem.shape[0])}"
                )
            q_emb = np.asarray(self.q_mem[qi], dtype=np.float32).copy()

            node_emb_list.append(node_emb)
            node_mask_list.append(np.ones((n,), dtype=np.bool_))
            seed_mask_list.append(seed_mask)
            node_label_list.append(node_lbl)
            node_cids_list.append(np.asarray(node_cids, dtype=np.int64))
            candidate_mask_list.append(candidate_mask)
            edge_src_list.append(edge_src)
            edge_dst_list.append(edge_dst)
            edge_rel_emb_list.append(edge_rel_emb)
            edge_rel_id_list.append(np.asarray(edge_rel_ids, dtype=np.int64))
            edge_dir_list.append(edge_dir)
            edge_mask_list.append(np.ones((e,), dtype=np.bool_))
            q_emb_list.append(q_emb)
            gold_list.append(gold)
            orig_ids.append(str(ex.get("orig_id", "")))

        bsz = len(batch)
        node_emb_t = torch.zeros((bsz, max_n, self.entity_dim), dtype=torch.float32)
        node_mask_t = torch.zeros((bsz, max_n), dtype=torch.bool)
        seed_mask_t = torch.zeros((bsz, max_n), dtype=torch.bool)
        node_label_t = torch.zeros((bsz, max_n), dtype=torch.float32)
        node_cids_t = torch.full((bsz, max_n), -1, dtype=torch.long)
        candidate_mask_t = torch.zeros((bsz, max_n), dtype=torch.bool)

        edge_src_t = torch.zeros((bsz, max_e), dtype=torch.long)
        edge_dst_t = torch.zeros((bsz, max_e), dtype=torch.long)
        edge_rel_emb_t = torch.zeros((bsz, max_e, self.relation_dim), dtype=torch.float32)
        edge_rel_id_t = torch.zeros((bsz, max_e), dtype=torch.long)
        edge_dir_t = torch.zeros((bsz, max_e), dtype=torch.long)
        edge_mask_t = torch.zeros((bsz, max_e), dtype=torch.bool)

        q_emb_t = torch.zeros((bsz, self.query_dim), dtype=torch.float32)

        for i in range(bsz):
            n = node_emb_list[i].shape[0]
            e = edge_rel_emb_list[i].shape[0]
            node_emb_t[i, :n] = torch.from_numpy(node_emb_list[i])
            node_mask_t[i, :n] = torch.from_numpy(node_mask_list[i])
            seed_mask_t[i, :n] = torch.from_numpy(seed_mask_list[i])
            node_label_t[i, :n] = torch.from_numpy(node_label_list[i])
            node_cids_t[i, :n] = torch.from_numpy(node_cids_list[i])
            candidate_mask_t[i, :n] = torch.from_numpy(candidate_mask_list[i])
            if e > 0:
                edge_src_t[i, :e] = torch.from_numpy(edge_src_list[i])
                edge_dst_t[i, :e] = torch.from_numpy(edge_dst_list[i])
                edge_rel_emb_t[i, :e] = torch.from_numpy(edge_rel_emb_list[i])
                edge_rel_id_t[i, :e] = torch.from_numpy(edge_rel_id_list[i])
                edge_dir_t[i, :e] = torch.from_numpy(edge_dir_list[i])
                edge_mask_t[i, :e] = torch.from_numpy(edge_mask_list[i])
            q_emb_t[i] = torch.from_numpy(q_emb_list[i])

        return {
            "node_emb": node_emb_t,
            "node_mask": node_mask_t,
            "seed_mask": seed_mask_t,
            "node_labels": node_label_t,
            "node_cids": node_cids_t,
            "candidate_mask": candidate_mask_t,
            "edge_src": edge_src_t,
            "edge_dst": edge_dst_t,
            "edge_rel_emb": edge_rel_emb_t,
            "edge_rel_ids": edge_rel_id_t,
            "edge_dir": edge_dir_t,
            "edge_mask": edge_mask_t,
            "q_emb": q_emb_t,
            "gold_answers": gold_list,
            "orig_ids": orig_ids,
        }


class _ReaRevFusion(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.r = nn.Linear(hidden_size * 3, hidden_size, bias=False)
        self.g = nn.Linear(hidden_size * 3, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = torch.cat([x, y, x - y], dim=-1)
        r_val = self.r(xy)
        g_val = torch.sigmoid(self.g(xy))
        return g_val * r_val + (1.0 - g_val) * x


class _ReaRevQueryReform(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fusion = _ReaRevFusion(hidden_size)

    def forward(self, q_node: torch.Tensor, ent_emb: torch.Tensor, seed_dist: torch.Tensor) -> torch.Tensor:
        seed_retrieve = torch.matmul(seed_dist.unsqueeze(0), ent_emb).squeeze(0)
        return self.fusion(q_node, seed_retrieve)


class RecursiveSubgraphReader(nn.Module):
    def __init__(
        self,
        entity_dim: int,
        relation_dim: int,
        query_dim: int,
        hidden_size: int,
        recursion_steps: int = 8,
        dropout: float = 0.1,
        use_direction_embedding: bool = False,
        outer_reasoning_enabled: bool = False,
        outer_reasoning_steps: int = 3,
        gnn_variant: str = "rearev_bfs",
        rearev_num_instructions: int = 3,
        rearev_adapt_stages: int = 1,
        rearev_normalized_gnn: bool = False,
        rearev_latent_reasoning_enabled: bool = False,
        rearev_latent_residual_alpha: float = 0.25,
        rearev_latent_update_mode: str = "gru",
        rearev_global_gate_enabled: bool = False,
        rearev_logit_global_fusion_enabled: bool = False,
        rearev_dynamic_halting_enabled: bool = False,
        rearev_dynamic_halting_threshold: float = 0.9,
        rearev_dynamic_halting_min_steps: int = 1,
        rearev_trm_style_enabled: bool = False,
        rearev_trm_tminus1_no_grad: bool = True,
        rearev_trm_detach_carry: bool = True,
        rearev_trm_supervise_all_stages: bool = False,
        rearev_act_stop_in_train: bool = False,
        rearev_asymmetric_yz_enabled: bool = False,
        rearev_asym_inner_y_ema_enabled: bool = False,
        rearev_asym_inner_y_ema_alpha: float = 0.0,
        trm_rel_topk_relations: int = 0,
        trm_rel_score_alpha: float = 1.0,
        trm_rel_use_relid_policy: bool = True,
    ):
        super().__init__()
        self.entity_dim = int(entity_dim)
        self.relation_dim = int(relation_dim)
        self.query_dim = int(query_dim)
        self.hidden_size = int(hidden_size)
        self.recursion_steps = max(0, int(recursion_steps))
        self.use_direction_embedding = bool(use_direction_embedding)
        self.outer_reasoning_enabled = bool(outer_reasoning_enabled)
        self.outer_reasoning_steps = max(1, int(outer_reasoning_steps))
        variant = str(gnn_variant or "rearev_bfs").strip().lower()
        allowed_variants = {
            "rearev_bfs",
            "rearev_dplus",
        }
        if variant not in allowed_variants:
            raise ValueError(
                f"Unsupported gnn_variant={variant!r}. allowed={sorted(allowed_variants)}"
            )
        self.gnn_variant = variant
        self.rearev_num_instructions = max(1, int(rearev_num_instructions))
        self.rearev_adapt_stages = max(1, int(rearev_adapt_stages))
        self.rearev_normalized_gnn = bool(rearev_normalized_gnn)
        self.rearev_latent_reasoning_enabled = bool(rearev_latent_reasoning_enabled)
        self.rearev_latent_residual_alpha = float(max(0.0, rearev_latent_residual_alpha))
        latent_mode = str(rearev_latent_update_mode or "gru").strip().lower()
        # D+ alias: same core operator as D(rearev_bfs), but uses attention-memory latent update.
        if self.gnn_variant == "rearev_dplus" and latent_mode == "gru":
            latent_mode = "attn"
        if latent_mode not in {"gru", "attn"}:
            raise ValueError(
                f"Unsupported rearev_latent_update_mode={latent_mode!r}. allowed=['gru','attn']"
            )
        self.rearev_latent_update_mode = latent_mode
        self.rearev_global_gate_enabled = bool(rearev_global_gate_enabled)
        self.rearev_logit_global_fusion_enabled = bool(rearev_logit_global_fusion_enabled)
        self.rearev_dynamic_halting_enabled = bool(rearev_dynamic_halting_enabled)
        self.rearev_dynamic_halting_threshold = float(
            min(1.0, max(0.0, rearev_dynamic_halting_threshold))
        )
        self.rearev_dynamic_halting_min_steps = max(1, int(rearev_dynamic_halting_min_steps))
        self.rearev_trm_style_enabled = bool(rearev_trm_style_enabled)
        self.rearev_trm_tminus1_no_grad = bool(rearev_trm_tminus1_no_grad)
        self.rearev_trm_detach_carry = bool(rearev_trm_detach_carry)
        self.rearev_trm_supervise_all_stages = bool(rearev_trm_supervise_all_stages)
        self.rearev_act_stop_in_train = bool(rearev_act_stop_in_train)
        self.rearev_asymmetric_yz_enabled = bool(rearev_asymmetric_yz_enabled)
        self.rearev_asym_inner_y_ema_enabled = bool(rearev_asym_inner_y_ema_enabled)
        self.rearev_asym_inner_y_ema_alpha = float(
            min(1.0, max(0.0, rearev_asym_inner_y_ema_alpha))
        )
        self.trm_rel_topk_relations = max(0, int(trm_rel_topk_relations))
        self.trm_rel_score_alpha = float(max(0.0, trm_rel_score_alpha))
        self.trm_rel_use_relid_policy = bool(trm_rel_use_relid_policy)

        self.node_proj = nn.Linear(self.entity_dim, self.hidden_size)
        self.rel_proj = nn.Linear(self.relation_dim, self.hidden_size)
        self.rel_proj_inv = nn.Linear(self.relation_dim, self.hidden_size)
        self.q_proj = nn.Linear(self.query_dim, self.hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        # Shared score function (ReaRev score_func).
        self.out_head = nn.Linear(self.hidden_size, 1)
        self.rearev_ins_proj = nn.Linear(self.hidden_size, self.hidden_size * self.rearev_num_instructions)
        self.rearev_rel_linears = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.recursion_steps)]
        )
        self.rearev_e2e_linears = nn.ModuleList(
            [
                nn.Linear(
                    self.hidden_size * (1 + 2 * self.rearev_num_instructions),
                    self.hidden_size,
                )
                for _ in range(self.recursion_steps)
            ]
        )
        # ReaRev reform blocks are only needed when we actually perform stage-to-stage
        # instruction updates (adapt_stages > 1). Keeping them absent for single-stage
        # runs avoids DDP unused-parameter failures with find_unused_parameters=False.
        self.rearev_reforms = (
            nn.ModuleList([_ReaRevQueryReform(self.hidden_size) for _ in range(self.rearev_num_instructions)])
            if self.rearev_adapt_stages > 1
            else None
        )
        if self.rearev_global_gate_enabled:
            self.rearev_global_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        else:
            self.rearev_global_gate = None
        if self.rearev_logit_global_fusion_enabled:
            self.rearev_score_fusion = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, 1),
            )
        else:
            self.rearev_score_fusion = None
        self.rearev_latent_gru = None
        self.rearev_latent_norm = None
        self.rearev_latent_to_ins = None
        self.rearev_latent_attn_q_proj = None
        self.rearev_latent_attn_k_proj = None
        self.rearev_latent_attn_v_proj = None
        self.rearev_latent_attn_gate = None
        if self.rearev_latent_reasoning_enabled:
            self.rearev_latent_norm = nn.LayerNorm(self.hidden_size)
            self.rearev_latent_to_ins = nn.Linear(
                self.hidden_size,
                self.hidden_size * self.rearev_num_instructions,
            )
            # Zero-init keeps step-0 behavior equivalent to vanilla ReaRev, then learns deltas.
            nn.init.zeros_(self.rearev_latent_to_ins.weight)
            nn.init.zeros_(self.rearev_latent_to_ins.bias)
            if self.rearev_latent_update_mode == "gru":
                self.rearev_latent_gru = nn.GRUCell(self.hidden_size, self.hidden_size)
            elif self.rearev_latent_update_mode == "attn":
                self.rearev_latent_attn_q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.rearev_latent_attn_k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.rearev_latent_attn_v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.rearev_latent_attn_gate = nn.Linear(self.hidden_size * 2, 1)
        if self.rearev_dynamic_halting_enabled or self.rearev_trm_style_enabled:
            self.rearev_halt_proj = nn.Linear(self.hidden_size, 1)
        else:
            self.rearev_halt_proj = None
        if self.rearev_asymmetric_yz_enabled:
            # Asymmetric y-update gate: combines current latent state z and previous y-context.
            self.rearev_yz_update_gate = nn.Linear(self.hidden_size * 2, 1)
            # Residual y-update head: y_new = y_old + delta(h_final, z_final, y_old).
            self.rearev_y_delta_head = nn.Sequential(
                nn.Linear(self.hidden_size * 2 + 1, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, 1),
            )
            # Optional cycle-level halt head over (z_global, y_context).
            if self.rearev_dynamic_halting_enabled or self.rearev_trm_style_enabled:
                self.rearev_halt_with_y_proj = nn.Linear(self.hidden_size * 2, 1)
            else:
                self.rearev_halt_with_y_proj = None
        else:
            self.rearev_yz_update_gate = None
            self.rearev_y_delta_head = None
            self.rearev_halt_with_y_proj = None
        if self.outer_reasoning_enabled:
            self.outer_state_update = nn.GRUCell(self.hidden_size, self.hidden_size)
            self.outer_state_norm = nn.LayerNorm(self.hidden_size)
        if self.use_direction_embedding:
            self.edge_dir_emb = nn.Embedding(2, self.hidden_size)
        if self.gnn_variant in {"trm_rel_recursive", "trm_frontier_recursive", "trm_frontier_rearev1"}:
            self.trm_rel_query_proj = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.trm_rel_score_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.trm_rel_gate_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.trm_rel_node_update = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.trm_rel_h_norm = nn.LayerNorm(self.hidden_size)
            self.trm_rel_y_update_gate = nn.Linear(self.hidden_size * 3, 1)
            self.trm_rel_z_gru = nn.GRUCell(self.hidden_size, self.hidden_size)
            self.trm_rel_z_norm = nn.LayerNorm(self.hidden_size)
        else:
            self.trm_rel_query_proj = None
            self.trm_rel_score_proj = None
            self.trm_rel_gate_proj = None
            self.trm_rel_node_update = None
            self.trm_rel_h_norm = None
            self.trm_rel_y_update_gate = None
            self.trm_rel_z_gru = None
            self.trm_rel_z_norm = None

    def _score_nodes(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        base_score = self.out_head(self.dropout(h)).squeeze(-1)
        if self.rearev_logit_global_fusion_enabled and self.rearev_score_fusion is not None:
            z_expand = z.unsqueeze(0).expand(h.shape[0], -1)
            fuse_in = torch.cat([h, z_expand, h * z_expand], dim=-1)
            fusion_delta = self.rearev_score_fusion(self.dropout(fuse_in)).squeeze(-1)
            return base_score + fusion_delta
        return base_score

    def _halt_logit(
        self,
        z: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        y_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            self.rearev_asymmetric_yz_enabled
            and self.rearev_halt_with_y_proj is not None
            and h is not None
            and y_dist is not None
            and int(h.shape[0]) > 0
            and int(y_dist.shape[0]) == int(h.shape[0])
        ):
            y_ctx = (h * y_dist.unsqueeze(-1)).sum(dim=0)
            halt_in = torch.cat([z, y_ctx], dim=-1).unsqueeze(0)
            return self.rearev_halt_with_y_proj(halt_in).squeeze()
        if self.rearev_halt_proj is not None:
            return self.rearev_halt_proj(z).squeeze(-1)
        return z.new_zeros(())

    def _seed_distribution(
        self,
        seed_mask: Optional[torch.Tensor],
        n: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if seed_mask is None:
            p = torch.zeros((n,), dtype=dtype, device=device)
            p[0] = 1.0
            return p
        sm = seed_mask[:n].to(device=device)
        p = sm.to(dtype=dtype)
        denom = p.sum()
        if float(denom.item()) <= 0.0:
            p = torch.zeros((n,), dtype=dtype, device=device)
            p[0] = 1.0
            return p
        return p / denom

    def _edge_weights(
        self,
        src: Optional[torch.Tensor],
        n: int,
        e: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if e <= 0 or src is None:
            return None
        if not self.rearev_normalized_gnn:
            return torch.ones((e,), dtype=dtype, device=device)
        deg = torch.zeros((n,), dtype=dtype, device=device)
        deg.index_add_(0, src, torch.ones((e,), dtype=dtype, device=device))
        return 1.0 / deg[src].clamp(min=1.0)

    def _safe_prob_normalize(
        self,
        x: torch.Tensor,
        *,
        fallback: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        # Avoid exploding gradients from division by near-zero probability mass.
        if x.numel() <= 0:
            return x
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        total = x.sum()
        if float(total.detach().item()) > float(eps):
            out = x / total
            return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        if fallback is not None and fallback.numel() == x.numel():
            fb = torch.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0)
            fb_total = fb.sum()
            if float(fb_total.detach().item()) > float(eps):
                return fb / fb_total
        # Final fallback: put all mass on a deterministic index to keep the graph alive.
        out = torch.zeros_like(x)
        out.view(-1)[0] = 1.0
        return out

    def _update_latent_state(
        self,
        prev_latent: torch.Tensor,
        h_state: torch.Tensor,
        node_dist: Optional[torch.Tensor],
        *,
        use_plain_context: bool = False,
    ) -> torch.Tensor:
        if not self.rearev_latent_reasoning_enabled:
            return prev_latent
        if int(h_state.shape[0]) <= 0:
            return prev_latent
        if self.rearev_latent_norm is None:
            return prev_latent

        if self.rearev_latent_update_mode == "gru":
            if self.rearev_latent_gru is None:
                return prev_latent
            if use_plain_context:
                ctx = h_state.mean(dim=0)
            elif node_dist is not None and int(node_dist.numel()) == int(h_state.shape[0]):
                prob = self._safe_prob_normalize(node_dist, eps=1e-8)
                ctx = (h_state * prob.unsqueeze(-1)).sum(dim=0)
            else:
                ctx = h_state.mean(dim=0)
            latent = self.rearev_latent_gru(
                ctx.unsqueeze(0),
                prev_latent.unsqueeze(0),
            ).squeeze(0)
            return self.rearev_latent_norm(latent)

        if self.rearev_latent_update_mode == "attn":
            if (
                self.rearev_latent_attn_q_proj is None
                or self.rearev_latent_attn_k_proj is None
                or self.rearev_latent_attn_v_proj is None
                or self.rearev_latent_attn_gate is None
            ):
                return prev_latent
            q = self.rearev_latent_attn_q_proj(prev_latent).unsqueeze(0)  # [1, d]
            k = self.rearev_latent_attn_k_proj(h_state)  # [n, d]
            v = self.rearev_latent_attn_v_proj(h_state)  # [n, d]
            logits = (k * q).sum(dim=-1) / math.sqrt(float(max(1, self.hidden_size)))
            if (not use_plain_context) and node_dist is not None and int(node_dist.numel()) == int(h_state.shape[0]):
                prob = self._safe_prob_normalize(node_dist, eps=1e-8).clamp(min=1e-8)
                logits = logits + torch.log(prob)
            attn = torch.softmax(logits, dim=0)
            ctx = (v * attn.unsqueeze(-1)).sum(dim=0)
            gate_in = torch.cat([prev_latent, ctx], dim=-1).unsqueeze(0)
            beta = torch.sigmoid(self.rearev_latent_attn_gate(gate_in)).view(())
            latent = (1.0 - beta) * prev_latent + beta * ctx
            return self.rearev_latent_norm(latent)

        return prev_latent

    def _reason_layer_pair(
        self,
        curr_dist: torch.Tensor,
        instruction: torch.Tensor,
        rel_linear: nn.Linear,
        rel_h: Optional[torch.Tensor],
        rel_h_inv: Optional[torch.Tensor],
        src: Optional[torch.Tensor],
        dst: Optional[torch.Tensor],
        edge_w: Optional[torch.Tensor],
        n: int,
        e: int,
        ref_h: torch.Tensor,
        return_edge_strength: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if e <= 0 or src is None or dst is None or rel_h is None or rel_h_inv is None:
            z = torch.zeros_like(ref_h)
            return z, z, None
        fact_query = instruction.unsqueeze(0).expand(e, -1)

        fact_val_fwd = F.relu(rel_linear(rel_h) * fact_query)
        fact_val_inv = F.relu(rel_linear(rel_h_inv) * fact_query)
        if edge_w is not None:
            ww = edge_w.unsqueeze(-1)
            fact_val_fwd = fact_val_fwd * ww
            fact_val_inv = fact_val_inv * ww

        msg_fwd = fact_val_fwd * curr_dist[src].unsqueeze(-1)
        msg_inv = fact_val_inv * curr_dist[dst].unsqueeze(-1)

        agg_fwd = torch.zeros((n, self.hidden_size), dtype=ref_h.dtype, device=ref_h.device)
        agg_inv = torch.zeros((n, self.hidden_size), dtype=ref_h.dtype, device=ref_h.device)
        agg_fwd.index_add_(0, dst, msg_fwd)
        agg_inv.index_add_(0, src, msg_inv)
        edge_strength = None
        if return_edge_strength:
            edge_strength = (
                fact_val_fwd.mean(dim=-1) * curr_dist[src]
                + fact_val_inv.mean(dim=-1) * curr_dist[dst]
            )
        return agg_fwd, agg_inv, edge_strength

    def _trm_rel_edge_policy(
        self,
        z: torch.Tensor,
        q_state: torch.Tensor,
        rel_h: Optional[torch.Tensor],
        edge_rel_ids: Optional[torch.Tensor],
        e: int,
        ref_h: torch.Tensor,
    ) -> torch.Tensor:
        if (
            e <= 0
            or rel_h is None
            or self.trm_rel_query_proj is None
            or self.trm_rel_score_proj is None
        ):
            return torch.zeros((max(0, e),), dtype=ref_h.dtype, device=ref_h.device)

        zq = torch.cat([z, q_state], dim=-1).unsqueeze(0)
        rel_query = torch.tanh(self.trm_rel_query_proj(zq)).squeeze(0)
        rel_query = self.trm_rel_score_proj(rel_query)
        edge_raw = (rel_h * rel_query.unsqueeze(0)).sum(dim=-1)
        edge_raw = torch.nan_to_num(edge_raw, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
            min=-20.0, max=20.0
        )

        if (
            self.trm_rel_use_relid_policy
            and edge_rel_ids is not None
            and int(edge_rel_ids.numel()) >= e
            and bool((edge_rel_ids[:e] >= 0).any().item())
        ):
            rel_ids = edge_rel_ids[:e].to(torch.long)
            valid = rel_ids >= 0
            rel_ids_valid = rel_ids[valid]
            edge_raw_valid = edge_raw[valid]
            uniq_rel, inv = torch.unique(
                rel_ids_valid, sorted=False, return_inverse=True
            )
            rel_sum = edge_raw.new_zeros((int(uniq_rel.numel()),))
            rel_cnt = edge_raw.new_zeros((int(uniq_rel.numel()),))
            rel_sum.index_add_(0, inv, edge_raw_valid)
            rel_cnt.index_add_(0, inv, torch.ones_like(edge_raw_valid))
            rel_scores = rel_sum / rel_cnt.clamp(min=1.0)
            rel_scores = torch.nan_to_num(
                rel_scores, nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(min=-20.0, max=20.0)
            if 0 < self.trm_rel_topk_relations < int(rel_scores.numel()):
                kk = int(self.trm_rel_topk_relations)
                _, topi = torch.topk(rel_scores, k=kk, largest=True)
                masked = torch.full_like(rel_scores, -1e9)
                masked[topi] = rel_scores[topi]
                rel_scores = masked
            rel_policy = torch.softmax(rel_scores, dim=0)
            rel_policy = self._safe_prob_normalize(rel_policy, eps=1e-8)
            edge_w = edge_raw.new_zeros((e,))
            edge_w[valid] = rel_policy[inv]
        else:
            edge_w = torch.softmax(edge_raw, dim=0)
            if 0 < self.trm_rel_topk_relations < e:
                kk = int(self.trm_rel_topk_relations)
                topv, topi = torch.topk(edge_w, k=kk, largest=True)
                sparse = torch.zeros_like(edge_w)
                sparse[topi] = topv
                edge_w = self._safe_prob_normalize(sparse, eps=1e-8)
            else:
                edge_w = self._safe_prob_normalize(edge_w, eps=1e-8)
        return edge_w

    def _inner_recur_trm_rel(
        self,
        h: torch.Tensor,
        src: Optional[torch.Tensor],
        dst: Optional[torch.Tensor],
        rel_h: Optional[torch.Tensor],
        rel_h_inv: Optional[torch.Tensor],
        edge_rel_ids: Optional[torch.Tensor],
        q_inj: torch.Tensor,
        seed_mask: Optional[torch.Tensor],
        n: int,
        e: int,
        return_rel_trace: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        del rel_h_inv  # not used in this variant
        q_state = q_inj.squeeze(0)
        h_state = h
        z_state = q_state
        y_state = self._seed_distribution(seed_mask=seed_mask, n=n, dtype=h.dtype, device=h.device)
        score_tp = torch.log(y_state.clamp(min=1e-8))

        trm_step_logits: List[torch.Tensor] = []
        trm_step_halt_logits: List[torch.Tensor] = []
        trm_step_valid: List[torch.Tensor] = []
        trace_step_logits: List[torch.Tensor] = []
        trace_step_edge_importance: List[torch.Tensor] = []

        outer_should_break = False
        for stage_idx in range(self.rearev_adapt_stages):
            for step_idx in range(self.recursion_steps):
                edge_w = self._trm_rel_edge_policy(
                    z=z_state,
                    q_state=q_state,
                    rel_h=rel_h,
                    edge_rel_ids=edge_rel_ids,
                    e=e,
                    ref_h=h_state,
                )

                if e > 0 and src is not None and dst is not None:
                    y_src = y_state[src]
                    y_msg = y_src * edge_w
                    y_prop = y_state.new_zeros((n,))
                    y_prop.index_add_(0, dst, y_msg)
                    y_prop = self._safe_prob_normalize(y_prop, fallback=y_state, eps=1e-6)
                else:
                    y_prop = y_state

                if (
                    e > 0
                    and src is not None
                    and dst is not None
                    and rel_h is not None
                    and self.trm_rel_gate_proj is not None
                ):
                    h_src = h_state[src]
                    rel_gate = torch.sigmoid(self.trm_rel_gate_proj(self.dropout(rel_h)))
                    node_msg = h_src * rel_gate * edge_w.unsqueeze(-1)
                    agg = h_state.new_zeros((n, self.hidden_size))
                    agg.index_add_(0, dst, node_msg)
                else:
                    agg = h_state.new_zeros((n, self.hidden_size))

                if self.trm_rel_node_update is None or self.trm_rel_h_norm is None:
                    raise RuntimeError("trm_rel_recursive modules are not initialized.")
                delta_h = torch.tanh(
                    self.trm_rel_node_update(self.dropout(torch.cat([h_state, agg], dim=-1)))
                )
                h_candidate = self.trm_rel_h_norm(h_state + self.dropout(delta_h))

                if self.trm_rel_z_gru is None or self.trm_rel_z_norm is None:
                    raise RuntimeError("trm_rel_recursive latent modules are not initialized.")
                summary = (h_candidate * y_prop.unsqueeze(-1)).sum(dim=0)
                z_candidate = self.trm_rel_z_gru(
                    summary.unsqueeze(0), z_state.unsqueeze(0)
                ).squeeze(0)
                z_candidate = self.trm_rel_z_norm(z_candidate)

                if self.trm_rel_y_update_gate is None:
                    raise RuntimeError("trm_rel_recursive y-update modules are not initialized.")
                z_expand = z_candidate.unsqueeze(0).expand(n, -1)
                q_expand = q_state.unsqueeze(0).expand(n, -1)
                beta_in = torch.cat([h_candidate, z_expand, h_candidate * q_expand], dim=-1)
                beta = torch.sigmoid(self.trm_rel_y_update_gate(self.dropout(beta_in))).squeeze(-1)
                y_next = (1.0 - beta) * y_state + beta * y_prop
                y_next = self._safe_prob_normalize(y_next, fallback=y_state, eps=1e-6)

                score_base = self._score_nodes(h_candidate, z_candidate)
                score_tp = score_base + (self.trm_rel_score_alpha * torch.log(y_next.clamp(min=1e-8)))
                y_state = torch.softmax(score_tp, dim=0)
                h_state = h_candidate
                z_state = z_candidate

                halt_logit = self._halt_logit(z_state, h_state, y_state)
                if self.rearev_trm_style_enabled and (
                    self.rearev_trm_supervise_all_stages
                    or ((stage_idx + 1) >= self.rearev_adapt_stages)
                ):
                    trm_step_logits.append(score_tp)
                    trm_step_halt_logits.append(halt_logit)
                    trm_step_valid.append(torch.ones((), dtype=torch.bool, device=score_tp.device))

                if self.rearev_dynamic_halting_enabled and (
                    (not self.training) or self.rearev_act_stop_in_train
                ):
                    if (
                        (step_idx + 1) >= self.rearev_dynamic_halting_min_steps
                        and float(torch.sigmoid(halt_logit).detach().item())
                        >= self.rearev_dynamic_halting_threshold
                    ):
                        outer_should_break = True

                if return_rel_trace:
                    trace_step_logits.append(score_tp)
                    if e > 0:
                        trace_step_edge_importance.append(edge_w)
                    else:
                        trace_step_edge_importance.append(
                            score_tp.new_zeros((0,), dtype=score_tp.dtype, device=score_tp.device)
                        )

                if outer_should_break:
                    break
            if outer_should_break:
                break

        trm_aux = None
        if self.rearev_trm_style_enabled or return_rel_trace:
            trm_aux = {}
            if self.rearev_trm_style_enabled:
                if trm_step_logits:
                    trm_aux["step_logits"] = torch.stack(trm_step_logits, dim=0)
                    trm_aux["step_halt_logits"] = torch.stack(trm_step_halt_logits, dim=0)
                    trm_aux["step_valid_mask"] = torch.stack(trm_step_valid, dim=0)
                else:
                    trm_aux["step_logits"] = score_tp.new_full((0, n), -1e4)
                    trm_aux["step_halt_logits"] = score_tp.new_zeros((0,))
                    trm_aux["step_valid_mask"] = torch.zeros(
                        (0,), dtype=torch.bool, device=score_tp.device
                    )
            if return_rel_trace:
                if trace_step_logits:
                    trm_aux["trace_step_logits"] = torch.stack(trace_step_logits, dim=0)
                    trm_aux["trace_step_edge_importance"] = torch.stack(
                        trace_step_edge_importance, dim=0
                    )
                else:
                    trm_aux["trace_step_logits"] = score_tp.new_full((0, n), -1e4)
                    trm_aux["trace_step_edge_importance"] = score_tp.new_zeros((0, e))
                if edge_rel_ids is None or int(edge_rel_ids.numel()) <= 0:
                    trm_aux["trace_edge_rel_ids"] = score_tp.new_full((e,), -1, dtype=torch.long)
                else:
                    trm_aux["trace_edge_rel_ids"] = edge_rel_ids[:e].to(dtype=torch.long)

        return h_state, score_tp, trm_aux

    def _inner_recur_trm_frontier(
        self,
        h: torch.Tensor,
        src: Optional[torch.Tensor],
        dst: Optional[torch.Tensor],
        rel_h: Optional[torch.Tensor],
        rel_h_inv: Optional[torch.Tensor],
        edge_rel_ids: Optional[torch.Tensor],
        q_inj: torch.Tensor,
        seed_mask: Optional[torch.Tensor],
        n: int,
        e: int,
        return_rel_trace: bool = False,
        use_rearev1_operator: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # Frontier-recursive mode:
        # step core is transition/control over y (frontier), and final readout happens at the end.
        q_state = q_inj.squeeze(0)
        h_state = h
        z_state = q_state
        y_state = self._seed_distribution(seed_mask=seed_mask, n=n, dtype=h.dtype, device=h.device)
        score_tp = torch.log(y_state.clamp(min=1e-8))

        trm_step_logits: List[torch.Tensor] = []
        trm_step_halt_logits: List[torch.Tensor] = []
        trm_step_valid: List[torch.Tensor] = []
        trace_step_logits: List[torch.Tensor] = []
        trace_step_edge_importance: List[torch.Tensor] = []
        frontier_step_score_raw: List[torch.Tensor] = []
        frontier_step_score_frontier: List[torch.Tensor] = []
        frontier_step_y_dist: List[torch.Tensor] = []
        frontier_step_y_entropy: List[torch.Tensor] = []

        for step_idx in range(self.recursion_steps):
            edge_w = self._trm_rel_edge_policy(
                z=z_state,
                q_state=q_state,
                rel_h=rel_h,
                edge_rel_ids=edge_rel_ids,
                e=e,
                ref_h=h_state,
            )

            if e > 0 and src is not None and dst is not None:
                y_src = y_state[src]
                y_msg = y_src * edge_w
                y_prop = y_state.new_zeros((n,))
                y_prop.index_add_(0, dst, y_msg)
                y_prop = self._safe_prob_normalize(y_prop, fallback=y_state, eps=1e-6)
            else:
                y_prop = y_state

            if use_rearev1_operator:
                # ReaRev 1-step operator:
                # use relation-aware forward/inverse propagation and collapse to one message tensor.
                instruction = torch.tanh(q_state + z_state)
                if self.rearev_ins_proj is not None:
                    ins_all = self.rearev_ins_proj(instruction).view(
                        self.rearev_num_instructions, self.hidden_size
                    )
                    instruction = ins_all[0]
                rel_linear = self.rearev_rel_linears[step_idx % len(self.rearev_rel_linears)]
                agg_fwd, agg_inv, _ = self._reason_layer_pair(
                    curr_dist=y_state,
                    instruction=instruction,
                    rel_linear=rel_linear,
                    rel_h=rel_h,
                    rel_h_inv=rel_h_inv,
                    src=src,
                    dst=dst,
                    edge_w=edge_w if e > 0 else None,
                    n=n,
                    e=e,
                    ref_h=h_state,
                    return_edge_strength=False,
                )
                agg = 0.5 * (agg_fwd + agg_inv)
            elif (
                e > 0
                and src is not None
                and dst is not None
                and rel_h is not None
                and self.trm_rel_gate_proj is not None
            ):
                h_src = h_state[src]
                rel_gate = torch.sigmoid(self.trm_rel_gate_proj(self.dropout(rel_h)))
                node_msg = h_src * rel_gate * edge_w.unsqueeze(-1)
                agg = h_state.new_zeros((n, self.hidden_size))
                agg.index_add_(0, dst, node_msg)
            else:
                agg = h_state.new_zeros((n, self.hidden_size))

            if self.trm_rel_node_update is None or self.trm_rel_h_norm is None:
                raise RuntimeError("trm_frontier_recursive modules are not initialized.")
            delta_h = torch.tanh(
                self.trm_rel_node_update(self.dropout(torch.cat([h_state, agg], dim=-1)))
            )
            h_candidate = self.trm_rel_h_norm(h_state + self.dropout(delta_h))

            if self.trm_rel_z_gru is None or self.trm_rel_z_norm is None:
                raise RuntimeError("trm_frontier_recursive latent modules are not initialized.")
            summary = (h_candidate * y_prop.unsqueeze(-1)).sum(dim=0)
            z_candidate = self.trm_rel_z_gru(
                summary.unsqueeze(0), z_state.unsqueeze(0)
            ).squeeze(0)
            z_candidate = self.trm_rel_z_norm(z_candidate)

            if self.trm_rel_y_update_gate is None:
                raise RuntimeError("trm_frontier_recursive y-update modules are not initialized.")
            z_expand = z_candidate.unsqueeze(0).expand(n, -1)
            q_expand = q_state.unsqueeze(0).expand(n, -1)
            beta_in = torch.cat([h_candidate, z_expand, h_candidate * q_expand], dim=-1)
            beta = torch.sigmoid(self.trm_rel_y_update_gate(self.dropout(beta_in))).squeeze(-1)
            y_next = (1.0 - beta) * y_state + beta * y_prop
            y_next = self._safe_prob_normalize(y_next, fallback=y_state, eps=1e-6)

            score_base = self._score_nodes(h_candidate, z_candidate)
            # Keep frontier term bounded so it does not numerically dominate score_base.
            frontier_log = torch.log(y_next.clamp(min=1e-8)).clamp(min=-8.0, max=0.0)
            score_frontier = self.trm_rel_score_alpha * frontier_log
            score_tp = score_base + score_frontier

            # Frontier semantics: carry frontier state directly, not via softmax(score_tp).
            y_state = y_next
            h_state = h_candidate
            z_state = z_candidate
            frontier_step_score_raw.append(score_base)
            frontier_step_score_frontier.append(score_frontier)
            frontier_step_y_dist.append(y_state)
            y_entropy = -(y_state * torch.log(y_state.clamp(min=1e-12))).sum()
            frontier_step_y_entropy.append(y_entropy)

            halt_logit = self._halt_logit(z_state, h_state, y_state)
            trm_step_logits.append(score_tp)
            trm_step_halt_logits.append(halt_logit)
            trm_step_valid.append(torch.ones((), dtype=torch.bool, device=score_tp.device))
            if (
                self.rearev_trm_style_enabled
                and self.training
                and self.rearev_trm_detach_carry
                and ((step_idx + 1) < self.recursion_steps)
            ):
                h_state = h_state.detach()
                z_state = z_state.detach()
                y_state = y_state.detach()
                score_tp = score_tp.detach()

            if return_rel_trace:
                trace_step_logits.append(score_tp)
                if e > 0:
                    trace_step_edge_importance.append(edge_w)
                else:
                    trace_step_edge_importance.append(
                        score_tp.new_zeros((0,), dtype=score_tp.dtype, device=score_tp.device)
                    )

            if self.rearev_dynamic_halting_enabled and (
                (not self.training) or self.rearev_act_stop_in_train
            ):
                if (
                    (step_idx + 1) >= self.rearev_dynamic_halting_min_steps
                    and float(torch.sigmoid(halt_logit).detach().item())
                    >= self.rearev_dynamic_halting_threshold
                ):
                    break

        trm_aux = None
        if self.gnn_variant in {"trm_frontier_recursive", "trm_frontier_rearev1"} or self.rearev_trm_style_enabled or return_rel_trace:
            trm_aux = {}
            if trm_step_logits:
                trm_aux["step_logits"] = torch.stack(trm_step_logits, dim=0)
                trm_aux["step_halt_logits"] = torch.stack(trm_step_halt_logits, dim=0)
                trm_aux["step_valid_mask"] = torch.stack(trm_step_valid, dim=0)
            else:
                trm_aux["step_logits"] = score_tp.new_full((0, n), -1e4)
                trm_aux["step_halt_logits"] = score_tp.new_zeros((0,))
                trm_aux["step_valid_mask"] = torch.zeros(
                    (0,), dtype=torch.bool, device=score_tp.device
                )
            if frontier_step_score_raw:
                trm_aux["frontier_step_score_raw"] = torch.stack(frontier_step_score_raw, dim=0)
                trm_aux["frontier_step_score_frontier"] = torch.stack(frontier_step_score_frontier, dim=0)
                trm_aux["frontier_step_y_dist"] = torch.stack(frontier_step_y_dist, dim=0)
                trm_aux["frontier_step_y_entropy"] = torch.stack(frontier_step_y_entropy, dim=0)
            else:
                trm_aux["frontier_step_score_raw"] = score_tp.new_full((0, n), 0.0)
                trm_aux["frontier_step_score_frontier"] = score_tp.new_full((0, n), 0.0)
                trm_aux["frontier_step_y_dist"] = score_tp.new_full((0, n), 0.0)
                trm_aux["frontier_step_y_entropy"] = score_tp.new_zeros((0,))
            if return_rel_trace:
                if trace_step_logits:
                    trm_aux["trace_step_logits"] = torch.stack(trace_step_logits, dim=0)
                    trm_aux["trace_step_edge_importance"] = torch.stack(
                        trace_step_edge_importance, dim=0
                    )
                else:
                    trm_aux["trace_step_logits"] = score_tp.new_full((0, n), -1e4)
                    trm_aux["trace_step_edge_importance"] = score_tp.new_zeros((0, e))
                if edge_rel_ids is None or int(edge_rel_ids.numel()) <= 0:
                    trm_aux["trace_edge_rel_ids"] = score_tp.new_full((e,), -1, dtype=torch.long)
                else:
                    trm_aux["trace_edge_rel_ids"] = edge_rel_ids[:e].to(dtype=torch.long)

        return h_state, score_tp, trm_aux

    def _inner_recur_rearev(
        self,
        h: torch.Tensor,
        src: Optional[torch.Tensor],
        dst: Optional[torch.Tensor],
        rel_h: Optional[torch.Tensor],
        rel_h_inv: Optional[torch.Tensor],
        edge_rel_ids: Optional[torch.Tensor],
        q_inj: torch.Tensor,
        seed_mask: Optional[torch.Tensor],
        n: int,
        e: int,
        return_rel_trace: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        q_state = q_inj.squeeze(0)
        instructions = self.rearev_ins_proj(q_state).view(self.rearev_num_instructions, self.hidden_size)
        seed_dist = self._seed_distribution(seed_mask=seed_mask, n=n, dtype=h.dtype, device=h.device)
        edge_w = self._edge_weights(src=src, n=n, e=e, dtype=h.dtype, device=h.device)
        latent_state = q_state
        if self.rearev_asymmetric_yz_enabled:
            # TRM-style asymmetric mode uses explicit global latent scratchpad initialized at zero.
            latent_state = torch.zeros_like(q_state)

        h_stage = h
        if self.rearev_asymmetric_yz_enabled:
            # y is explicitly initialized from seed entities.
            score_tp = torch.log(seed_dist.clamp(min=1e-8))
            curr_dist = torch.softmax(score_tp, dim=0)
        else:
            score_tp = self._score_nodes(h_stage, latent_state)
            curr_dist = seed_dist
        trm_step_logits: List[torch.Tensor] = []
        trm_step_halt_logits: List[torch.Tensor] = []
        trm_step_valid: List[torch.Tensor] = []
        trace_step_logits: List[torch.Tensor] = []
        trace_step_edge_importance: List[torch.Tensor] = []
        outer_should_break = False
        for stage_idx in range(self.rearev_adapt_stages):
            y_prev_score = score_tp
            y_prev_dist = curr_dist
            y_inner_score = y_prev_score
            y_inner_dist = y_prev_dist
            halt_mass = 0.0
            alive_prob = h_stage.new_tensor(1.0)
            for step_idx in range(self.recursion_steps):
                rel_linear = self.rearev_rel_linears[step_idx]
                e2e_linear = self.rearev_e2e_linears[step_idx]
                reason_dist = y_inner_dist if self.rearev_asymmetric_yz_enabled else curr_dist
                step_instructions = instructions
                if self.rearev_latent_reasoning_enabled and self.rearev_latent_to_ins is not None:
                    ins_delta = self.rearev_latent_to_ins(latent_state).view(
                        self.rearev_num_instructions, self.hidden_size
                    )
                    step_instructions = instructions + (
                        self.rearev_latent_residual_alpha * torch.tanh(ins_delta)
                    )
                use_no_grad_prefix = (
                    self.rearev_trm_style_enabled
                    and self.training
                    and self.rearev_trm_tminus1_no_grad
                    and ((step_idx + 1) < self.recursion_steps)
                )
                step_ctx = torch.no_grad if use_no_grad_prefix else nullcontext
                should_break = False
                score_for_trace = score_tp
                with step_ctx():
                    neighbor_reps: List[torch.Tensor] = []
                    h_in = h_stage
                    if self.rearev_asymmetric_yz_enabled:
                        # Inject current y distribution into node state before message passing.
                        h_in = h_stage * (1.0 + reason_dist.unsqueeze(-1))
                    step_edge_importance = (
                        h_stage.new_zeros((e,), dtype=h_stage.dtype, device=h_stage.device)
                        if (return_rel_trace and e > 0)
                        else None
                    )
                    for ins_idx in range(self.rearev_num_instructions):
                        agg_fwd, agg_inv, edge_strength = self._reason_layer_pair(
                            curr_dist=reason_dist,
                            instruction=step_instructions[ins_idx],
                            rel_linear=rel_linear,
                            rel_h=rel_h,
                            rel_h_inv=rel_h_inv,
                            src=src,
                            dst=dst,
                            edge_w=edge_w,
                            n=n,
                            e=e,
                            ref_h=h_in,
                            return_edge_strength=return_rel_trace,
                        )
                        neighbor_reps.append(agg_fwd)
                        neighbor_reps.append(agg_inv)
                        if step_edge_importance is not None and edge_strength is not None:
                            step_edge_importance = step_edge_importance + edge_strength
                    prev_h = h_stage
                    prev_score = score_tp
                    prev_dist = curr_dist
                    prev_latent = latent_state
                    next_local_entity_emb = torch.cat([h_in] + neighbor_reps, dim=-1)
                    h_msg = F.relu(e2e_linear(self.dropout(next_local_entity_emb)))
                    if self.rearev_global_gate_enabled and self.rearev_global_gate is not None:
                        z_expand = prev_latent.unsqueeze(0).expand(prev_h.shape[0], -1)
                        gate_in = torch.cat([prev_h, z_expand], dim=-1)
                        gate = torch.sigmoid(self.rearev_global_gate(gate_in))
                        h_candidate = gate * h_msg + (1.0 - gate) * prev_h
                    else:
                        h_candidate = h_msg
                    score_candidate = self._score_nodes(h_candidate, prev_latent)
                    dist_candidate = torch.softmax(score_candidate, dim=0)
                    score_for_trace = score_candidate
                    latent_candidate = prev_latent
                    if (
                        self.rearev_latent_reasoning_enabled
                    ):
                        # D: GRU latent memory, D+: attention-memory latent update.
                        latent_candidate = self._update_latent_state(
                            prev_latent=prev_latent,
                            h_state=h_candidate,
                            node_dist=None if self.rearev_asymmetric_yz_enabled else dist_candidate,
                            use_plain_context=bool(self.rearev_asymmetric_yz_enabled),
                        )
                    if self.rearev_dynamic_halting_enabled and (not self.rearev_asymmetric_yz_enabled):
                        alive_before = alive_prob
                        p_halt = h_stage.new_tensor(0.0)
                        if self.rearev_halt_proj is not None:
                            p_halt = torch.sigmoid(self.rearev_halt_proj(latent_candidate)).squeeze(-1)
                            if (step_idx + 1) < self.rearev_dynamic_halting_min_steps:
                                p_halt = torch.zeros_like(p_halt)
                        p_halt = p_halt.clamp(min=0.0, max=1.0)
                        alive_prob = alive_before * (1.0 - p_halt)
                        inv_alive = 1.0 - alive_prob
                        h_stage = alive_prob * h_candidate + inv_alive * prev_h
                        if self.rearev_asymmetric_yz_enabled:
                            # Keep y fixed during inner z-updates; update y only once per outer stage.
                            score_tp = prev_score
                            curr_dist = prev_dist
                        else:
                            score_tp = alive_prob * score_candidate + inv_alive * prev_score
                            curr_dist = alive_prob * dist_candidate + inv_alive * prev_dist
                        latent_state = alive_prob * latent_candidate + inv_alive * prev_latent
                        if (not self.training) or self.rearev_act_stop_in_train:
                            halt_mass += float((alive_before * p_halt).detach().item())
                            if (
                                (step_idx + 1) >= self.rearev_dynamic_halting_min_steps
                                and halt_mass >= self.rearev_dynamic_halting_threshold
                            ):
                                should_break = True
                    else:
                        h_stage = h_candidate
                        if self.rearev_asymmetric_yz_enabled:
                            if self.rearev_asym_inner_y_ema_enabled and self.rearev_asym_inner_y_ema_alpha > 0.0:
                                ema_a = float(self.rearev_asym_inner_y_ema_alpha)
                                y_prev = y_inner_dist
                                y_inner_dist = ((1.0 - ema_a) * y_inner_dist) + (ema_a * dist_candidate)
                                y_inner_dist = self._safe_prob_normalize(
                                    y_inner_dist, fallback=y_prev, eps=1e-6
                                )
                                y_inner_score = torch.log(y_inner_dist.clamp(min=1e-8))
                            score_tp = y_inner_score
                            curr_dist = y_inner_dist
                        else:
                            score_tp = score_candidate
                            curr_dist = dist_candidate
                        latent_state = latent_candidate

                if return_rel_trace:
                    trace_step_logits.append(score_for_trace if self.rearev_asymmetric_yz_enabled else score_tp)
                    if step_edge_importance is None:
                        trace_step_edge_importance.append(
                            h_stage.new_zeros((e,), dtype=h_stage.dtype, device=h_stage.device)
                        )
                    else:
                        trace_step_edge_importance.append(
                            step_edge_importance / float(max(1, self.rearev_num_instructions))
                        )

                if self.rearev_trm_style_enabled and (not self.rearev_asymmetric_yz_enabled) and (
                    self.rearev_trm_supervise_all_stages
                    or ((stage_idx + 1) >= self.rearev_adapt_stages)
                ):
                    trm_step_logits.append(score_tp)
                    trm_step_halt_logits.append(self._halt_logit(latent_state, h_stage, curr_dist))
                    trm_step_valid.append(torch.ones((), dtype=torch.bool, device=score_tp.device))
                    if (
                        self.training
                        and self.rearev_trm_detach_carry
                        and ((step_idx + 1) < self.recursion_steps)
                    ):
                        h_stage = h_stage.detach()
                        score_tp = score_tp.detach()
                        curr_dist = curr_dist.detach()
                        latent_state = latent_state.detach()

                if should_break:
                    break

            if self.rearev_asymmetric_yz_enabled:
                if self.rearev_asym_inner_y_ema_enabled and self.rearev_asym_inner_y_ema_alpha > 0.0:
                    y_prev_score = y_inner_score
                    y_prev_dist = y_inner_dist
                # Outer y-update: after inner z refinement loops, update y exactly once.
                z_expand = latent_state.unsqueeze(0).expand(h_stage.shape[0], -1)
                y_prev_col = y_prev_score.unsqueeze(-1)
                if self.rearev_y_delta_head is not None:
                    delta_in = torch.cat([h_stage, z_expand, y_prev_col], dim=-1)
                    y_delta = self.rearev_y_delta_head(self.dropout(delta_in)).squeeze(-1)
                else:
                    y_delta = self._score_nodes(h_stage, latent_state)
                y_candidate_outer = y_prev_score + y_delta
                y_ctx = (h_stage * y_prev_dist.unsqueeze(-1)).sum(dim=0)
                if self.rearev_yz_update_gate is not None:
                    gate_in = torch.cat([latent_state, y_ctx], dim=-1).unsqueeze(0)
                    update_ratio = torch.sigmoid(self.rearev_yz_update_gate(gate_in)).view(())
                    update_ratio = update_ratio.clamp(min=0.0, max=1.0)
                else:
                    update_ratio = y_candidate_outer.new_tensor(1.0)
                score_tp = update_ratio * y_candidate_outer + (1.0 - update_ratio) * y_prev_score
                curr_dist = torch.softmax(score_tp, dim=0)
                halt_logit_cycle = self._halt_logit(latent_state, h_stage, curr_dist)

                if self.rearev_trm_style_enabled and (
                    self.rearev_trm_supervise_all_stages
                    or ((stage_idx + 1) >= self.rearev_adapt_stages)
                ):
                    trm_step_logits.append(score_tp)
                    trm_step_halt_logits.append(halt_logit_cycle)
                    trm_step_valid.append(torch.ones((), dtype=torch.bool, device=score_tp.device))
                    if (
                        self.training
                        and self.rearev_trm_detach_carry
                        and ((stage_idx + 1) < self.rearev_adapt_stages)
                    ):
                        h_stage = h_stage.detach()
                        score_tp = score_tp.detach()
                        curr_dist = curr_dist.detach()
                        latent_state = latent_state.detach()
                if (
                    self.rearev_dynamic_halting_enabled
                    and ((not self.training) or self.rearev_act_stop_in_train)
                    and ((stage_idx + 1) >= self.rearev_dynamic_halting_min_steps)
                    and float(torch.sigmoid(halt_logit_cycle).detach().item()) >= self.rearev_dynamic_halting_threshold
                ):
                    outer_should_break = True

            if stage_idx + 1 >= self.rearev_adapt_stages:
                break
            if outer_should_break:
                break
            if self.rearev_reforms is not None:
                updated: List[torch.Tensor] = []
                for ins_idx, reform in enumerate(self.rearev_reforms):
                    updated.append(reform(instructions[ins_idx], h_stage, seed_dist))
                instructions = torch.stack(updated, dim=0)

        trm_aux = None
        if self.rearev_trm_style_enabled or return_rel_trace:
            trm_aux = {}
            if self.rearev_trm_style_enabled:
                if trm_step_logits:
                    trm_aux["step_logits"] = torch.stack(trm_step_logits, dim=0)
                    trm_aux["step_halt_logits"] = torch.stack(trm_step_halt_logits, dim=0)
                    trm_aux["step_valid_mask"] = torch.stack(trm_step_valid, dim=0)
                else:
                    trm_aux["step_logits"] = score_tp.new_full((0, n), -1e4)
                    trm_aux["step_halt_logits"] = score_tp.new_zeros((0,))
                    trm_aux["step_valid_mask"] = torch.zeros(
                        (0,), dtype=torch.bool, device=score_tp.device
                    )
            if return_rel_trace:
                if trace_step_logits:
                    trm_aux["trace_step_logits"] = torch.stack(trace_step_logits, dim=0)
                    trm_aux["trace_step_edge_importance"] = torch.stack(
                        trace_step_edge_importance, dim=0
                    )
                else:
                    trm_aux["trace_step_logits"] = score_tp.new_full((0, n), -1e4)
                    trm_aux["trace_step_edge_importance"] = score_tp.new_zeros((0, e))
                if edge_rel_ids is None or int(edge_rel_ids.numel()) <= 0:
                    trm_aux["trace_edge_rel_ids"] = score_tp.new_full((e,), -1, dtype=torch.long)
                else:
                    trm_aux["trace_edge_rel_ids"] = edge_rel_ids[:e].to(dtype=torch.long)

        return h_stage, score_tp, trm_aux

    def _run_inner(
        self,
        h: torch.Tensor,
        src: Optional[torch.Tensor],
        dst: Optional[torch.Tensor],
        rel_h: Optional[torch.Tensor],
        rel_h_inv: Optional[torch.Tensor],
        edge_rel_ids: Optional[torch.Tensor],
        q_inj: torch.Tensor,
        seed_mask: Optional[torch.Tensor],
        n: int,
        e: int,
        return_rel_trace: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if self.gnn_variant == "trm_rel_recursive":
            return self._inner_recur_trm_rel(
                h=h,
                src=src,
                dst=dst,
                rel_h=rel_h,
                rel_h_inv=rel_h_inv,
                edge_rel_ids=edge_rel_ids,
                q_inj=q_inj,
                seed_mask=seed_mask,
                n=n,
                e=e,
                return_rel_trace=return_rel_trace,
            )
        if self.gnn_variant == "trm_frontier_recursive":
            return self._inner_recur_trm_frontier(
                h=h,
                src=src,
                dst=dst,
                rel_h=rel_h,
                rel_h_inv=rel_h_inv,
                edge_rel_ids=edge_rel_ids,
                q_inj=q_inj,
                seed_mask=seed_mask,
                n=n,
                e=e,
                return_rel_trace=return_rel_trace,
            )
        if self.gnn_variant == "trm_frontier_rearev1":
            return self._inner_recur_trm_frontier(
                h=h,
                src=src,
                dst=dst,
                rel_h=rel_h,
                rel_h_inv=rel_h_inv,
                edge_rel_ids=edge_rel_ids,
                q_inj=q_inj,
                seed_mask=seed_mask,
                n=n,
                e=e,
                return_rel_trace=return_rel_trace,
                use_rearev1_operator=True,
            )
        return self._inner_recur_rearev(
            h=h,
            src=src,
            dst=dst,
            rel_h=rel_h,
            rel_h_inv=rel_h_inv,
            edge_rel_ids=edge_rel_ids,
            q_inj=q_inj,
            seed_mask=seed_mask,
            n=n,
            e=e,
            return_rel_trace=return_rel_trace,
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        node_mask: torch.Tensor,
        seed_mask: Optional[torch.Tensor],
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel_emb: torch.Tensor,
        edge_rel_ids: Optional[torch.Tensor],
        edge_dir: Optional[torch.Tensor],
        edge_mask: torch.Tensor,
        q_emb: torch.Tensor,
        return_aux: bool = False,
        return_rel_trace: bool = False,
    ):
        bsz, max_n, _ = node_emb.shape
        qh = self.q_proj(q_emb)
        h_all = self.node_proj(node_emb)
        logits = node_emb.new_full((bsz, max_n), -1e4)
        need_aux = bool(return_aux or return_rel_trace)
        step_logits = None
        step_halt_logits = None
        step_valid_mask = None
        trace_step_logits = None
        trace_step_edge_importance = None
        trace_step_valid_mask = None
        trace_edge_rel_ids = None
        frontier_step_score_raw = None
        frontier_step_score_frontier = None
        frontier_step_y_dist = None
        frontier_step_y_entropy = None
        aux_steps = int(self.recursion_steps)
        frontier_like_variant = self.gnn_variant in {"trm_frontier_recursive", "trm_frontier_rearev1"}
        if frontier_like_variant:
            aux_steps = int(self.recursion_steps)
        if self.rearev_trm_supervise_all_stages:
            aux_steps = int(max(1, self.rearev_adapt_stages))
            if not self.rearev_asymmetric_yz_enabled:
                aux_steps = int(self.recursion_steps * max(1, self.rearev_adapt_stages))
        if frontier_like_variant:
            aux_steps = int(self.recursion_steps)
        else:
            if self.rearev_asymmetric_yz_enabled:
                aux_steps = 1
        if return_aux:
            step_logits = node_emb.new_full((bsz, aux_steps, max_n), -1e4)
            step_halt_logits = node_emb.new_zeros((bsz, aux_steps))
            step_valid_mask = torch.zeros(
                (bsz, aux_steps), dtype=torch.bool, device=node_emb.device
            )
            if frontier_like_variant:
                frontier_step_score_raw = node_emb.new_zeros((bsz, aux_steps, max_n))
                frontier_step_score_frontier = node_emb.new_zeros((bsz, aux_steps, max_n))
                frontier_step_y_dist = node_emb.new_zeros((bsz, aux_steps, max_n))
                frontier_step_y_entropy = node_emb.new_zeros((bsz, aux_steps))
        if return_rel_trace:
            trace_steps = int(self.recursion_steps * max(1, self.rearev_adapt_stages))
            if frontier_like_variant:
                trace_steps = int(self.recursion_steps)
            max_e = int(edge_mask.shape[1]) if edge_mask.ndim == 2 else 0
            trace_step_logits = node_emb.new_full((bsz, trace_steps, max_n), -1e4)
            trace_step_edge_importance = node_emb.new_zeros((bsz, trace_steps, max_e))
            trace_step_valid_mask = torch.zeros(
                (bsz, trace_steps), dtype=torch.bool, device=node_emb.device
            )
            trace_edge_rel_ids = torch.full(
                (bsz, max_e), -1, dtype=torch.long, device=node_emb.device
            )

        for i in range(bsz):
            n = int(node_mask[i].sum().item())
            if n <= 0:
                continue
            h = h_all[i, :n]
            qb = qh[i].unsqueeze(0)
            seed_i = seed_mask[i, :n] if seed_mask is not None else None

            e = int(edge_mask[i].sum().item()) if edge_mask.shape[1] > 0 else 0
            if e > 0:
                src = edge_src[i, :e].long()
                dst = edge_dst[i, :e].long()
                rel_h = self.rel_proj(edge_rel_emb[i, :e])
                rel_h_inv = self.rel_proj_inv(edge_rel_emb[i, :e])
                rel_ids = edge_rel_ids[i, :e].long() if edge_rel_ids is not None else None
                if self.use_direction_embedding and edge_dir is not None:
                    dir_ids = edge_dir[i, :e].long().clamp(min=0, max=1)
                    rel_h = rel_h + self.edge_dir_emb(dir_ids)
                    rel_h_inv = rel_h_inv + self.edge_dir_emb(1 - dir_ids)
            else:
                src = None
                dst = None
                rel_h = None
                rel_h_inv = None
                rel_ids = None

            if not self.outer_reasoning_enabled:
                h, y_logits, trm_aux = self._run_inner(
                    h=h,
                    src=src,
                    dst=dst,
                    rel_h=rel_h,
                    rel_h_inv=rel_h_inv,
                    edge_rel_ids=rel_ids,
                    q_inj=qb,
                    seed_mask=seed_i,
                    n=n,
                    e=e,
                    return_rel_trace=return_rel_trace,
                )
            else:
                z = qb.squeeze(0)
                y_logits = self._score_nodes(h, z)
                trm_aux = None
                for _ in range(self.outer_reasoning_steps):
                    y_prob = torch.softmax(y_logits, dim=0)
                    denom = y_prob.sum().clamp(min=1e-6)
                    ctx = (h * y_prob.unsqueeze(-1)).sum(dim=0) / denom
                    z = self.outer_state_update(ctx.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
                    z = self.outer_state_norm(z)
                    q_loop = qb + z.unsqueeze(0)
                    h, y_logits, _ = self._run_inner(
                        h=h,
                        src=src,
                        dst=dst,
                        rel_h=rel_h,
                        rel_h_inv=rel_h_inv,
                        edge_rel_ids=rel_ids,
                        q_inj=q_loop,
                        seed_mask=seed_i,
                        n=n,
                        e=e,
                        return_rel_trace=False,
                    )

            logits[i, :n] = y_logits
            if return_aux and trm_aux is not None and step_logits is not None:
                local_step_logits = trm_aux["step_logits"]
                local_halt = trm_aux["step_halt_logits"]
                local_valid = trm_aux["step_valid_mask"]
                s = min(int(step_logits.shape[1]), int(local_step_logits.shape[0]))
                if s > 0:
                    step_logits[i, :s, :n] = local_step_logits[:s, :n]
                    step_halt_logits[i, :s] = local_halt[:s]
                    step_valid_mask[i, :s] = local_valid[:s]
                    if (
                        frontier_like_variant
                        and frontier_step_score_raw is not None
                        and "frontier_step_score_raw" in trm_aux
                    ):
                        local_raw = trm_aux["frontier_step_score_raw"]
                        local_front = trm_aux["frontier_step_score_frontier"]
                        local_y = trm_aux["frontier_step_y_dist"]
                        local_ent = trm_aux["frontier_step_y_entropy"]
                        sf = min(int(s), int(local_raw.shape[0]))
                        if sf > 0:
                            frontier_step_score_raw[i, :sf, :n] = local_raw[:sf, :n]
                            frontier_step_score_frontier[i, :sf, :n] = local_front[:sf, :n]
                            frontier_step_y_dist[i, :sf, :n] = local_y[:sf, :n]
                            frontier_step_y_entropy[i, :sf] = local_ent[:sf]
            if return_rel_trace and trm_aux is not None and trace_step_logits is not None:
                local_trace_logits = trm_aux.get("trace_step_logits", None)
                local_trace_edges = trm_aux.get("trace_step_edge_importance", None)
                local_trace_rel_ids = trm_aux.get("trace_edge_rel_ids", None)
                if local_trace_logits is not None:
                    s_trace = min(int(trace_step_logits.shape[1]), int(local_trace_logits.shape[0]))
                    if s_trace > 0:
                        trace_step_logits[i, :s_trace, :n] = local_trace_logits[:s_trace, :n]
                        trace_step_valid_mask[i, :s_trace] = True
                        if local_trace_edges is not None and e > 0:
                            trace_step_edge_importance[i, :s_trace, :e] = local_trace_edges[:s_trace, :e]
                if local_trace_rel_ids is not None and e > 0:
                    trace_edge_rel_ids[i, :e] = local_trace_rel_ids[:e]

        if need_aux:
            out = {
                "logits": logits,
            }
            if return_aux:
                out["step_logits"] = step_logits
                out["step_halt_logits"] = step_halt_logits
                out["step_valid_mask"] = step_valid_mask
                if frontier_like_variant and frontier_step_score_raw is not None:
                    out["frontier_step_score_raw"] = frontier_step_score_raw
                    out["frontier_step_score_frontier"] = frontier_step_score_frontier
                    out["frontier_step_y_dist"] = frontier_step_y_dist
                    out["frontier_step_y_entropy"] = frontier_step_y_entropy
            if return_rel_trace:
                out["trace_step_logits"] = trace_step_logits
                out["trace_step_edge_importance"] = trace_step_edge_importance
                out["trace_step_valid_mask"] = trace_step_valid_mask
                out["trace_edge_rel_ids"] = trace_edge_rel_ids
            return out
        return logits


def _masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    loss = loss * mask.to(torch.float32)
    denom = mask.to(torch.float32).sum().clamp(min=1.0)
    return loss.sum() / denom


def _masked_rearev_kl_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    no_positive_mode: str = "uniform",
) -> Tuple[torch.Tensor, int]:
    # ReaRev-style objective:
    # minimize KL( target_answer_distribution || predicted_entity_distribution )
    # where both distributions are defined on valid subgraph nodes.
    valid = mask.to(torch.bool)
    logits_masked = logits.masked_fill(~valid, -1e9)
    log_pred = F.log_softmax(logits_masked, dim=1)

    tgt = targets.to(torch.float32) * valid.to(torch.float32)
    tgt_sum = tgt.sum(dim=1, keepdim=True)
    has_pos = tgt_sum.squeeze(-1) > 0

    mode = str(no_positive_mode or "uniform").strip().lower()
    if mode in {"skip", "mask", "drop"}:
        # Do not force uniform targets for rows without positives.
        # This avoids over-regularizing mid-search/no-answer rows.
        if not bool(has_pos.any().item()):
            return logits.new_tensor(0.0), 0
        tgt_dist = tgt / tgt_sum.clamp(min=1.0)
        tgt_dist = tgt_dist / tgt_dist.sum(dim=1, keepdim=True).clamp(min=1e-12)
        kl = F.kl_div(log_pred, tgt_dist, reduction="none").sum(dim=1)
        kl = kl[has_pos]
        return kl.mean(), int(has_pos.to(torch.int64).sum().item())

    # Legacy behavior: fallback keeps loss finite for rows with no positives.
    valid_cnt = valid.to(torch.float32).sum(dim=1, keepdim=True).clamp(min=1.0)
    uniform = valid.to(torch.float32) / valid_cnt
    tgt_dist = torch.where(has_pos.unsqueeze(-1), tgt / tgt_sum.clamp(min=1.0), uniform)
    tgt_dist = tgt_dist / tgt_dist.sum(dim=1, keepdim=True).clamp(min=1e-12)
    kl = F.kl_div(log_pred, tgt_dist, reduction="none").sum(dim=1)
    return kl.mean(), int(has_pos.to(torch.int64).sum().item())


def _masked_rearev_step_kl_loss(
    step_logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    step_valid_mask: torch.Tensor,
    no_positive_mode: str = "uniform",
) -> Tuple[torch.Tensor, int]:
    # Step-wise ReaRev KL:
    # apply KL(target_dist || pred_dist_t) at each supervised step and average uniformly.
    bsz = int(step_logits.shape[0])
    num_steps = int(step_logits.shape[1])
    losses: List[torch.Tensor] = []
    valid_rows = 0
    mode = str(no_positive_mode or "uniform").strip().lower()
    skip_no_pos = mode in {"skip", "mask", "drop"}

    for i in range(bsz):
        valid = mask[i].to(torch.bool)
        if int(valid.to(torch.int64).sum().item()) <= 0:
            continue
        tgt = targets[i].to(torch.float32) * valid.to(torch.float32)
        tgt_sum = float(tgt.sum().item())
        has_pos = tgt_sum > 0.0
        if (not has_pos) and skip_no_pos:
            continue
        if has_pos:
            tgt_dist = tgt / max(tgt_sum, 1.0)
            valid_rows_base = 1
        else:
            denom = valid.to(torch.float32).sum().clamp(min=1.0)
            tgt_dist = valid.to(torch.float32) / denom
            valid_rows_base = 0
        tgt_dist = tgt_dist / tgt_dist.sum().clamp(min=1e-12)

        for t in range(num_steps):
            if not bool(step_valid_mask[i, t].item()):
                continue
            log_pred_t = F.log_softmax(step_logits[i, t].masked_fill(~valid, -1e9), dim=0)
            losses.append(F.kl_div(log_pred_t, tgt_dist, reduction="sum"))
            valid_rows += int(valid_rows_base)

    if not losses:
        return step_logits.new_tensor(0.0), 0
    return torch.stack(losses).mean(), int(valid_rows)


def _masked_trm_step_ce_loss(
    step_logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    step_valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    # TRM-style node decision supervision:
    # CE over node logits at each supervised recursion step.
    bsz = int(step_logits.shape[0])
    num_steps = int(step_logits.shape[1])
    losses: List[torch.Tensor] = []
    valid_rows = 0
    for i in range(bsz):
        valid_nodes = mask[i].to(torch.bool)
        valid_idx = torch.nonzero(valid_nodes, as_tuple=False).squeeze(-1)
        if valid_idx.numel() <= 0:
            continue
        pos_idx = torch.nonzero((targets[i] > 0.5) & valid_nodes, as_tuple=False).squeeze(-1)
        target_idx = int(pos_idx[0].item()) if pos_idx.numel() > 0 else int(valid_idx[0].item())
        target_t = torch.tensor([target_idx], dtype=torch.long, device=step_logits.device)
        for t in range(num_steps):
            if not bool(step_valid_mask[i, t].item()):
                continue
            row_logits = step_logits[i, t].masked_fill(~valid_nodes, -1e4).unsqueeze(0)
            losses.append(F.cross_entropy(row_logits, target_t, reduction="mean"))
            valid_rows += 1
    if not losses:
        return step_logits.new_tensor(0.0), 0
    return torch.stack(losses).mean(), int(valid_rows)


def _trm_halt_bce_loss(
    step_halt_logits: torch.Tensor,
    step_valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    # TRM-style halt supervision:
    # for each sample, final valid recursion step => halt target 1, others 0.
    bsz = int(step_halt_logits.shape[0])
    num_steps = int(step_halt_logits.shape[1])
    targets = torch.zeros_like(step_halt_logits, dtype=torch.float32)
    valid_steps = int(step_valid_mask.to(torch.int64).sum().item())
    if valid_steps <= 0:
        return step_halt_logits.new_tensor(0.0), 0
    for i in range(bsz):
        valid_idx = torch.nonzero(step_valid_mask[i], as_tuple=False).squeeze(-1)
        if valid_idx.numel() <= 0:
            continue
        last_t = int(valid_idx[-1].item())
        if 0 <= last_t < num_steps:
            targets[i, last_t] = 1.0
    raw = F.binary_cross_entropy_with_logits(step_halt_logits, targets, reduction="none")
    weighted = raw * step_valid_mask.to(torch.float32)
    denom = step_valid_mask.to(torch.float32).sum().clamp(min=1.0)
    return weighted.sum() / denom, int(valid_steps)


def _masked_bce_hard_negative_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    hard_negative_enabled: bool = False,
    hard_negative_topk: int = 64,
) -> Tuple[torch.Tensor, int]:
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    if (not hard_negative_enabled) or int(hard_negative_topk) <= 0:
        weighted = raw * mask.to(torch.float32)
        denom = mask.to(torch.float32).sum().clamp(min=1.0)
        return weighted.sum() / denom, int(mask.to(torch.int64).sum().item())

    bsz = int(logits.shape[0])
    keep_mask = torch.zeros_like(mask, dtype=torch.bool)
    total_keep = 0
    topk = max(1, int(hard_negative_topk))

    with torch.no_grad():
        for i in range(bsz):
            valid = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
            if valid.numel() <= 0:
                continue
            y = targets[i, valid]
            pos = valid[y > 0.5]
            neg = valid[y <= 0.5]

            row_keep = torch.zeros_like(mask[i], dtype=torch.bool)
            if pos.numel() > 0:
                row_keep[pos] = True
            if neg.numel() > 0:
                k = min(topk, int(neg.numel()))
                hard = torch.topk(logits[i, neg].detach(), k=k, largest=True).indices
                hard_neg = neg[hard]
                row_keep[hard_neg] = True
            if not row_keep.any():
                row_keep[valid] = True
            keep_mask[i] = row_keep
            total_keep += int(row_keep.sum().item())

    weighted = raw * keep_mask.to(torch.float32)
    denom = keep_mask.to(torch.float32).sum().clamp(min=1.0)
    return weighted.sum() / denom, int(total_keep)


def _ranking_hard_negative_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.2,
    hard_negative_topk: int = 16,
) -> Tuple[torch.Tensor, int]:
    bsz = int(logits.shape[0])
    losses: List[torch.Tensor] = []
    pair_count = 0
    topk = max(1, int(hard_negative_topk))
    m = float(margin)

    for i in range(bsz):
        valid = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
        if valid.numel() <= 0:
            continue
        y = targets[i, valid]
        pos = valid[y > 0.5]
        neg = valid[y <= 0.5]
        if pos.numel() <= 0 or neg.numel() <= 0:
            continue

        pos_score = logits[i, pos].min()
        neg_scores = logits[i, neg]
        k = min(topk, int(neg_scores.numel()))
        hard_neg = torch.topk(neg_scores, k=k, largest=True).values
        losses.append(F.relu(m - pos_score + hard_neg).mean())
        pair_count += int(k)

    if not losses:
        return logits.new_tensor(0.0), 0
    return torch.stack(losses).mean(), int(pair_count)


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _sample_metrics_from_logits(
    logits_row: torch.Tensor,
    mask_row: torch.Tensor,
    node_cids_row: torch.Tensor,
    gold_answers: Sequence[int],
    candidate_mask_row: Optional[torch.Tensor],
    pred_topk: int,
    threshold: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    gold = set(int(x) for x in gold_answers)
    if not gold:
        return None, None, None, None

    valid_mask = mask_row
    if candidate_mask_row is not None:
        valid_mask = valid_mask & candidate_mask_row
    valid = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if valid.numel() <= 0:
        return None, None, None, None

    vals = torch.sigmoid(logits_row[valid])
    cids = node_cids_row[valid]
    keep = torch.nonzero(cids >= 0, as_tuple=False).squeeze(-1)
    if keep.numel() <= 0:
        return None, None, None, None
    vals = vals[keep]
    cids = cids[keep]

    order = torch.argsort(vals, descending=True)
    k = max(1, int(pred_topk))
    top_idx = order[: min(k, int(order.numel()))]
    top_nodes = [int(cids[j].item()) for j in top_idx]
    hit1 = 1.0 if top_nodes and int(top_nodes[0]) in gold else 0.0

    pred_set = set()
    thr = float(threshold)
    if thr > 0.0:
        sel = torch.nonzero(vals >= thr, as_tuple=False).squeeze(-1)
        pred_set = {int(cids[j].item()) for j in sel.tolist()}
    if not pred_set:
        pred_set = set(top_nodes)

    inter = len(pred_set & gold)
    precision = float(inter) / float(max(1, len(pred_set)))
    recall = float(inter) / float(max(1, len(gold)))
    f1 = 0.0
    if (precision + recall) > 0.0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    return float(hit1), float(f1), float(precision), float(recall)


@torch.no_grad()
def evaluate_subgraph_reader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pred_topk: int,
    threshold: float,
    is_main: bool,
    desc: str = "Eval-Subgraph",
    return_counts: bool = False,
) -> Tuple[float, float, float, float, int]:
    model.eval()
    hit_sum = 0.0
    f1_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    n_valid = 0
    skip = 0

    pbar = tqdm(loader, disable=not is_main, desc=desc)
    for batch in pbar:
        meta_gold = batch["gold_answers"]
        batch_dev = _move_batch_to_device(batch, device)
        logits = model(
            node_emb=batch_dev["node_emb"],
            node_mask=batch_dev["node_mask"],
            seed_mask=batch_dev.get("seed_mask", None),
            edge_src=batch_dev["edge_src"],
            edge_dst=batch_dev["edge_dst"],
            edge_rel_emb=batch_dev["edge_rel_emb"],
            edge_rel_ids=batch_dev.get("edge_rel_ids", None),
            edge_dir=batch_dev.get("edge_dir", None),
            edge_mask=batch_dev["edge_mask"],
            q_emb=batch_dev["q_emb"],
        )
        bsz = int(logits.shape[0])
        for i in range(bsz):
            hit, f1, precision, recall = _sample_metrics_from_logits(
                logits_row=logits[i],
                mask_row=batch_dev["node_mask"][i],
                node_cids_row=batch_dev["node_cids"][i],
                gold_answers=meta_gold[i],
                candidate_mask_row=batch_dev.get("candidate_mask", None)[i] if batch_dev.get("candidate_mask", None) is not None else None,
                pred_topk=pred_topk,
                threshold=threshold,
            )
            if hit is None or f1 is None or precision is None or recall is None:
                skip += 1
                continue
            hit_sum += float(hit)
            f1_sum += float(f1)
            precision_sum += float(precision)
            recall_sum += float(recall)
            n_valid += 1

    mean_hit = float(hit_sum / max(1, n_valid))
    mean_f1 = float(f1_sum / max(1, n_valid))
    mean_precision = float(precision_sum / max(1, n_valid))
    mean_recall = float(recall_sum / max(1, n_valid))
    if return_counts:
        return mean_hit, mean_f1, mean_precision, mean_recall, int(skip), int(n_valid)
    return mean_hit, mean_f1, mean_precision, mean_recall, int(skip)


def _load_relation_text_labels(relations_txt: str) -> Tuple[List[str], List[str]]:
    rel_ids: List[str] = []
    with open(relations_txt, "r", encoding="utf-8") as f:
        for line in f:
            rel_ids.append(line.strip())

    cand_paths: List[str] = []
    rel_dir = os.path.dirname(relations_txt)
    rel_base = os.path.basename(relations_txt)
    cand_paths.append(os.path.join(rel_dir, "relation_names_used.txt"))
    if rel_base.endswith("_ids.txt"):
        cand_paths.append(
            os.path.join(rel_dir, rel_base.replace("_ids.txt", "_names_used.txt"))
        )
    if rel_base.endswith("relations.txt"):
        cand_paths.append(os.path.join(rel_dir, "relations_names_used.txt"))

    rel_names: Optional[List[str]] = None
    for p in cand_paths:
        if not p or (not os.path.exists(p)):
            continue
        try:
            names = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    names.append(line.strip())
            if len(names) == len(rel_ids) and len(names) > 0:
                rel_names = names
                break
        except Exception:
            continue

    if rel_names is None:
        rel_names = []
        for rid in rel_ids:
            txt = rid.replace(".", " ").replace("_", " ").replace("/", " ").replace(":", " ")
            txt = re.sub(r"\s+", " ", txt).strip()
            rel_names.append(txt if txt else rid)

    return rel_ids, rel_names


def _load_entity_text_labels(entities_txt: str) -> List[str]:
    labels: List[str] = []
    with open(entities_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            ent_id = parts[0].strip() if parts else ""
            ent_name = parts[1].strip() if len(parts) > 1 else ""
            if ent_name and ent_name != ent_id:
                labels.append(f"{ent_name} <{ent_id}>")
            else:
                labels.append(ent_id)
    return labels


@torch.no_grad()
def debug_relation_topk_trace(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    relation_ids: Optional[List[str]],
    relation_labels: Optional[List[str]],
    entity_labels: Optional[List[str]],
    topk: int,
    log_examples: int,
    is_main: bool,
    path_dump_jsonl: str = "",
    dump_max_examples: int = 1000,
):
    if not is_main:
        return
    model.eval()
    k = max(1, int(topk))
    max_log_n = max(0, int(log_examples))
    max_dump_n = max(0, int(dump_max_examples))
    rid_list = relation_ids if isinstance(relation_ids, list) else []
    rlabel_list = relation_labels if isinstance(relation_labels, list) else []
    elabel_list = entity_labels if isinstance(entity_labels, list) else []
    dump_path = str(path_dump_jsonl or "").strip()
    dump_f = None
    if dump_path:
        dump_dir = os.path.dirname(dump_path)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        dump_f = open(dump_path, "w", encoding="utf-8")
    if max_log_n <= 0 and (dump_f is None or max_dump_n <= 0):
        if dump_f is not None:
            dump_f.close()
        return
    max_n = max(max_log_n, (max_dump_n if dump_f is not None else 0))

    def _ent_text(cid: int) -> str:
        if 0 <= int(cid) < len(elabel_list):
            return str(elabel_list[int(cid)])
        return str(cid)

    seen = 0
    try:
        for batch in loader:
            batch_dev = _move_batch_to_device(batch, device)
            out = model(
                node_emb=batch_dev["node_emb"],
                node_mask=batch_dev["node_mask"],
                seed_mask=batch_dev.get("seed_mask", None),
                edge_src=batch_dev["edge_src"],
                edge_dst=batch_dev["edge_dst"],
                edge_rel_emb=batch_dev["edge_rel_emb"],
                edge_rel_ids=batch_dev.get("edge_rel_ids", None),
                edge_dir=batch_dev.get("edge_dir", None),
                edge_mask=batch_dev["edge_mask"],
                q_emb=batch_dev["q_emb"],
                return_rel_trace=True,
            )
            trace_logits = out.get("trace_step_logits", None)
            trace_edge_importance = out.get("trace_step_edge_importance", None)
            trace_step_valid = out.get("trace_step_valid_mask", None)
            trace_edge_rel_ids = out.get("trace_edge_rel_ids", None)
            if (
                trace_logits is None
                or trace_edge_importance is None
                or trace_step_valid is None
                or trace_edge_rel_ids is None
            ):
                if seen < max_log_n:
                    print("[Trace-RelTopK] trace output unavailable.")
                return

            bsz = int(trace_logits.shape[0])
            for i in range(bsz):
                if seen >= max_n:
                    return
                do_log = seen < max_log_n
                do_dump = (dump_f is not None) and (seen < max_dump_n)
                n = int(batch_dev["node_mask"][i].sum().item())
                e = int(batch_dev["edge_mask"][i].sum().item()) if batch_dev["edge_mask"].shape[1] > 0 else 0
                orig_id = ""
                try:
                    orig_id = str(batch.get("orig_ids", [""])[i])
                except Exception:
                    orig_id = ""
                if do_log:
                    print(
                        f"[Trace-RelTopK] ex={seen} orig_id={orig_id} valid_nodes={n} "
                        f"valid_edges={e}"
                    )
                if n <= 0 or e <= 0:
                    if do_log:
                        print("  - no valid node/edge in sampled subgraph")
                    seen += 1
                    continue

                step_ids = torch.nonzero(trace_step_valid[i], as_tuple=False).squeeze(-1)
                if step_ids.numel() <= 0:
                    if do_log:
                        print("  - no valid recursion steps traced")
                    seen += 1
                    continue

                rel_ids = trace_edge_rel_ids[i, :e].to(torch.long)
                valid_rel_edge = rel_ids >= 0
                if not bool(valid_rel_edge.any()):
                    if do_log:
                        print("  - no valid relation ids on edges")
                    seen += 1
                    continue

                node_cids = batch_dev["node_cids"][i, :n].to(torch.long)
                edge_src = batch_dev["edge_src"][i, :e].to(torch.long)
                edge_dst = batch_dev["edge_dst"][i, :e].to(torch.long)
                edge_valid = batch_dev["edge_mask"][i, :e].to(torch.bool)

                step_topk_summary = []
                for t in step_ids.tolist():
                    edge_scores = trace_edge_importance[i, t, :e]
                    keep = valid_rel_edge
                    rel_ids_kept = rel_ids[keep]
                    score_kept = edge_scores[keep]
                    if rel_ids_kept.numel() <= 0:
                        continue

                    uniq_rel, inv = torch.unique(rel_ids_kept, sorted=False, return_inverse=True)
                    rel_score = torch.zeros((uniq_rel.numel(),), dtype=score_kept.dtype, device=score_kept.device)
                    rel_score.index_add_(0, inv, score_kept)
                    kk = min(k, int(uniq_rel.numel()))
                    topv, topi = torch.topk(rel_score, k=kk, largest=True)

                    node_logit = trace_logits[i, t, :n]
                    node_prob = torch.softmax(node_logit, dim=0)
                    node_kk = min(k, int(n))
                    node_topv, node_topi = torch.topk(node_logit, k=node_kk, largest=True)
                    best_local_idx = int(node_topi[0].item()) if node_kk > 0 else int(torch.argmax(node_prob).item())
                    best_cid = int(node_cids[best_local_idx].item()) if best_local_idx < int(node_cids.numel()) else -1
                    node_items = []
                    node_items_obj = []
                    for jj in range(node_kk):
                        local_idx = int(node_topi[jj].item())
                        cid = int(node_cids[local_idx].item()) if local_idx < int(node_cids.numel()) else -1
                        nlogit = float(node_topv[jj].item())
                        nprob = float(node_prob[local_idx].item()) if 0 <= local_idx < int(node_prob.numel()) else 0.0
                        ntext = _ent_text(cid)
                        node_items.append(f"{ntext}({nlogit:.4f}|p={nprob:.4f})")
                        node_items_obj.append(
                            {
                                "local_idx": int(local_idx),
                                "cid": int(cid),
                                "text": str(ntext),
                                "logit": float(nlogit),
                                "prob": float(nprob),
                            }
                        )

                    rel_items = []
                    rel_items_obj = []
                    for jj in range(kk):
                        rid = int(uniq_rel[topi[jj]].item())
                        rid_raw = rid_list[rid] if 0 <= rid < len(rid_list) else f"rid:{rid}"
                        rname = rlabel_list[rid] if 0 <= rid < len(rlabel_list) else rid_raw
                        rscore = float(topv[jj].item())
                        if rname != rid_raw:
                            rel_items.append(f"{rname} <{rid_raw}> ({rscore:.4f})")
                        else:
                            rel_items.append(f"{rname}({rscore:.4f})")
                        rel_items_obj.append(
                            {
                                "relation_index": int(rid),
                                "relation_id": str(rid_raw),
                                "relation_text": str(rname),
                                "score": float(rscore),
                            }
                        )
                    if do_log:
                        print(
                            f"  step={int(t)+1} best_node_cid={best_cid} "
                            f"top{k}_relations=[{', '.join(rel_items)}]"
                        )
                        print(f"          top{node_kk}_nodes=[{', '.join(node_items)}]")
                    step_topk_summary.append(
                        {
                            "step": int(t) + 1,
                            "best_node_cid": int(best_cid),
                            "best_node_text": _ent_text(best_cid),
                            "top_nodes": node_items_obj,
                            "top_relations": rel_items_obj,
                        }
                    )

                seed_local = torch.nonzero(batch_dev["seed_mask"][i, :n], as_tuple=False).squeeze(-1)
                if seed_local.numel() <= 0:
                    cur_local = 0
                else:
                    cur_local = int(seed_local[0].item())
                cur_cid = int(node_cids[cur_local].item()) if 0 <= cur_local < int(node_cids.numel()) else -1
                path_hops = []
                path_text_parts = [_ent_text(cur_cid)]
                for t in step_ids.tolist():
                    row_logits = trace_logits[i, t, :n]
                    row_prob = torch.softmax(row_logits, dim=0)
                    row_edge_score = trace_edge_importance[i, t, :e]
                    out_mask = edge_valid & valid_rel_edge & (edge_src == int(cur_local))
                    out_idx = torch.nonzero(out_mask, as_tuple=False).squeeze(-1)
                    if out_idx.numel() <= 0:
                        break
                    dst_nodes = edge_dst[out_idx].to(torch.long)
                    cand_score = row_edge_score[out_idx] + row_prob[dst_nodes]
                    best_pos = int(torch.argmax(cand_score).item())
                    edge_idx = int(out_idx[best_pos].item())
                    nxt_local = int(edge_dst[edge_idx].item())
                    nxt_cid = int(node_cids[nxt_local].item()) if 0 <= nxt_local < int(node_cids.numel()) else -1
                    rid = int(rel_ids[edge_idx].item())
                    rid_raw = rid_list[rid] if 0 <= rid < len(rid_list) else f"rid:{rid}"
                    rname = rlabel_list[rid] if 0 <= rid < len(rlabel_list) else rid_raw
                    hop = {
                        "step": int(t) + 1,
                        "from_local_idx": int(cur_local),
                        "from_cid": int(cur_cid),
                        "from_text": _ent_text(cur_cid),
                        "edge_idx": int(edge_idx),
                        "relation_index": int(rid),
                        "relation_id": str(rid_raw),
                        "relation_text": str(rname),
                        "to_local_idx": int(nxt_local),
                        "to_cid": int(nxt_cid),
                        "to_text": _ent_text(nxt_cid),
                        "edge_score": float(row_edge_score[edge_idx].item()),
                        "to_node_prob": float(row_prob[nxt_local].item()),
                        "combined_score": float(cand_score[best_pos].item()),
                    }
                    path_hops.append(hop)
                    if rname != rid_raw:
                        path_text_parts.append(f"--[{rname} <{rid_raw}>]--> {_ent_text(nxt_cid)}")
                    else:
                        path_text_parts.append(f"--[{rname}]--> {_ent_text(nxt_cid)}")
                    cur_local = nxt_local
                    cur_cid = nxt_cid
                if do_log:
                    if path_hops:
                        print(f"  explicit_path: {' '.join(path_text_parts)}")
                    else:
                        print("  explicit_path: (no outgoing edge from seed on traced steps)")

                if do_dump:
                    rec = {
                        "example_index": int(seen),
                        "orig_id": str(orig_id),
                        "seed_cid": int(path_hops[0]["from_cid"]) if path_hops else int(cur_cid),
                        "seed_text": (
                            str(path_hops[0]["from_text"])
                            if path_hops
                            else _ent_text(cur_cid)
                        ),
                        "path_hops": path_hops,
                        "path_text": " ".join(path_text_parts),
                        "step_topk_relations": step_topk_summary,
                    }
                    dump_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                seen += 1
    finally:
        if dump_f is not None:
            dump_f.close()

@torch.no_grad()
def debug_supervision_step_trace(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    is_main: bool,
    examples: int = 5,
    dump_jsonl: str = "",
    plot_png: str = "",
    log_prefix: str = "[Trace-Supervision]",
):
    if (not is_main) or int(examples) <= 0:
        return []

    model.eval()
    max_examples = max(1, int(examples))
    records: List[dict] = []

    for batch in loader:
        batch_dev = _move_batch_to_device(batch, device)
        out = model(
            node_emb=batch_dev["node_emb"],
            node_mask=batch_dev["node_mask"],
            seed_mask=batch_dev.get("seed_mask", None),
            edge_src=batch_dev["edge_src"],
            edge_dst=batch_dev["edge_dst"],
            edge_rel_emb=batch_dev["edge_rel_emb"],
            edge_rel_ids=batch_dev.get("edge_rel_ids", None),
            edge_dir=batch_dev.get("edge_dir", None),
            edge_mask=batch_dev["edge_mask"],
            q_emb=batch_dev["q_emb"],
            return_aux=True,
        )
        step_logits = out.get("step_logits", None)
        step_valid_mask = out.get("step_valid_mask", None)
        final_logits = out.get("logits", None)
        frontier_step_score_raw = out.get("frontier_step_score_raw", None)
        frontier_step_score_frontier = out.get("frontier_step_score_frontier", None)
        frontier_step_y_dist = out.get("frontier_step_y_dist", None)
        frontier_step_y_entropy = out.get("frontier_step_y_entropy", None)
        has_frontier_diag = (
            frontier_step_score_raw is not None
            and frontier_step_score_frontier is not None
            and frontier_step_y_dist is not None
            and frontier_step_y_entropy is not None
        )
        if step_logits is None or step_valid_mask is None or final_logits is None:
            print(f"{log_prefix} step outputs unavailable.")
            return records

        bsz = int(final_logits.shape[0])
        for i in range(bsz):
            if len(records) >= max_examples:
                break
            n = int(batch_dev["node_mask"][i].sum().item())
            if n <= 0:
                continue
            valid_nodes = batch_dev["node_mask"][i, :n].to(torch.bool)
            tgt = (batch_dev["node_labels"][i, :n].to(torch.float32) * valid_nodes.to(torch.float32))
            tgt_sum = float(tgt.sum().item())
            has_pos = tgt_sum > 0.0
            if not has_pos:
                # Supervision comparison is only meaningful when at least one GT node exists in subgraph.
                continue
            tgt_dist = tgt / max(tgt_sum, 1.0)
            pos_mask = (tgt > 0.0)
            dist_to_pos: Optional[torch.Tensor] = None
            if has_frontier_diag:
                dist_cpu = [n + 1] * n
                pos_locs = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1).tolist()
                if pos_locs:
                    for p in pos_locs:
                        dist_cpu[int(p)] = 0
                    adj: List[List[int]] = [[] for _ in range(n)]
                    e_i = int(batch_dev["edge_mask"][i].sum().item())
                    if e_i > 0:
                        src_row = batch_dev["edge_src"][i, :e_i]
                        dst_row = batch_dev["edge_dst"][i, :e_i]
                        for s_raw, d_raw in zip(src_row.tolist(), dst_row.tolist()):
                            s = int(s_raw)
                            d = int(d_raw)
                            if 0 <= s < n and 0 <= d < n:
                                # Undirected local distance for frontier-vs-gold proximity diagnostics.
                                adj[s].append(d)
                                adj[d].append(s)
                    q = deque(int(p) for p in pos_locs)
                    while q:
                        u = q.popleft()
                        nd = int(dist_cpu[u]) + 1
                        for v in adj[u]:
                            if nd < int(dist_cpu[v]):
                                dist_cpu[v] = nd
                                q.append(v)
                    dist_to_pos = torch.tensor(dist_cpu, dtype=torch.long, device=final_logits.device)

            orig_id = ""
            try:
                orig_id = str(batch.get("orig_ids", [""])[i])
            except Exception:
                orig_id = ""
            step_ids = torch.nonzero(step_valid_mask[i], as_tuple=False).squeeze(-1)
            rec = {
                "example_index": int(len(records)),
                "orig_id": str(orig_id),
                "has_positive": bool(has_pos),
                "supervision_steps": [],
            }
            for s_idx, t in enumerate(step_ids.tolist(), start=1):
                row_logits = step_logits[i, t, :n]
                masked = row_logits.masked_fill(~valid_nodes, -1e9)
                log_pred = F.log_softmax(masked, dim=0)
                prob = torch.softmax(masked, dim=0)
                kl_val = float(F.kl_div(log_pred, tgt_dist, reduction="sum").item())
                gt_mass = float(prob[pos_mask].sum().item())
                hit1, f1, precision, recall = _sample_metrics_from_logits(
                    logits_row=row_logits,
                    mask_row=batch_dev["node_mask"][i],
                    node_cids_row=batch_dev["node_cids"][i],
                    gold_answers=batch["gold_answers"][i],
                    candidate_mask_row=batch_dev.get("candidate_mask", None)[i] if batch_dev.get("candidate_mask", None) is not None else None,
                    pred_topk=1,
                    threshold=0.5,
                )
                rec["supervision_steps"].append(
                    {
                        "sup_step": int(s_idx),
                        "model_step_index": int(t) + 1,
                        "kl": float(kl_val),
                        "gt_mass": float(gt_mass),
                        "hit1": float(0.0 if hit1 is None else hit1),
                        "f1": float(0.0 if f1 is None else f1),
                        "precision": float(0.0 if precision is None else precision),
                        "recall": float(0.0 if recall is None else recall),
                    }
                )
                if has_frontier_diag:
                    raw_row = frontier_step_score_raw[i, t, :n]
                    front_row = frontier_step_score_frontier[i, t, :n]
                    y_row = frontier_step_y_dist[i, t, :n].clamp(min=0.0)
                    y_row = y_row / y_row.sum().clamp(min=1e-12)
                    raw_masked = raw_row.masked_fill(~valid_nodes, -1e9)
                    raw_prob = torch.softmax(raw_masked, dim=0)
                    raw_gt_mass = float(raw_prob[pos_mask].sum().item())
                    y_gt_mass = float(y_row[pos_mask].sum().item())
                    y_ent = float(frontier_step_y_entropy[i, t].item())
                    raw_abs = float(raw_row[valid_nodes].abs().mean().item())
                    front_abs = float(front_row[valid_nodes].abs().mean().item())

                    total_top_local = int(torch.argmax(prob).item())
                    raw_top_local = int(torch.argmax(raw_prob).item())
                    y_top_local = int(torch.argmax(y_row).item())
                    node_cids_row = batch_dev["node_cids"][i]
                    topk = min(5, n)
                    total_topk_local = torch.topk(prob, k=topk, dim=0).indices.tolist()
                    raw_topk_local = torch.topk(raw_prob, k=topk, dim=0).indices.tolist()
                    y_topk_local = torch.topk(y_row, k=topk, dim=0).indices.tolist()
                    total_top_cid = int(node_cids_row[total_top_local].item())
                    raw_top_cid = int(node_cids_row[raw_top_local].item())
                    y_top_cid = int(node_cids_row[y_top_local].item())
                    total_topk_cids = [int(node_cids_row[idx].item()) for idx in total_topk_local]
                    raw_topk_cids = [int(node_cids_row[idx].item()) for idx in raw_topk_local]
                    y_topk_cids = [int(node_cids_row[idx].item()) for idx in y_topk_local]
                    y_mass_d0 = y_mass_d1 = y_mass_d2 = y_mass_d3p = 0.0
                    if dist_to_pos is not None:
                        m0 = (dist_to_pos == 0)
                        m1 = (dist_to_pos == 1)
                        m2 = (dist_to_pos == 2)
                        m3p = (dist_to_pos >= 3)
                        if bool(m0.any()):
                            y_mass_d0 = float(y_row[m0].sum().item())
                        if bool(m1.any()):
                            y_mass_d1 = float(y_row[m1].sum().item())
                        if bool(m2.any()):
                            y_mass_d2 = float(y_row[m2].sum().item())
                        if bool(m3p.any()):
                            y_mass_d3p = float(y_row[m3p].sum().item())

                    rec["supervision_steps"][-1]["frontier_diag"] = {
                        "y_entropy": float(y_ent),
                        "y_gt_mass": float(y_gt_mass),
                        "raw_gt_mass": float(raw_gt_mass),
                        "raw_abs_mean": float(raw_abs),
                        "frontier_abs_mean": float(front_abs),
                        "y_mass_dist0": float(y_mass_d0),
                        "y_mass_dist1": float(y_mass_d1),
                        "y_mass_dist2": float(y_mass_d2),
                        "y_mass_dist3plus": float(y_mass_d3p),
                        "top1_cid_raw": int(raw_top_cid),
                        "top1_cid_total": int(total_top_cid),
                        "top1_cid_y": int(y_top_cid),
                        "top5_cids_raw": raw_topk_cids,
                        "top5_cids_total": total_topk_cids,
                        "top5_cids_y": y_topk_cids,
                    }

            f_hit1, f_f1, f_precision, f_recall = _sample_metrics_from_logits(
                logits_row=final_logits[i, :n],
                mask_row=batch_dev["node_mask"][i],
                node_cids_row=batch_dev["node_cids"][i],
                gold_answers=batch["gold_answers"][i],
                candidate_mask_row=batch_dev.get("candidate_mask", None)[i] if batch_dev.get("candidate_mask", None) is not None else None,
                pred_topk=1,
                threshold=0.5,
            )
            rec["final"] = {
                "hit1": float(0.0 if f_hit1 is None else f_hit1),
                "f1": float(0.0 if f_f1 is None else f_f1),
                "precision": float(0.0 if f_precision is None else f_precision),
                "recall": float(0.0 if f_recall is None else f_recall),
            }
            records.append(rec)
            step_summaries: List[str] = []
            for st in rec["supervision_steps"]:
                frontier_diag = st.get("frontier_diag", None)
                if isinstance(frontier_diag, dict):
                    step_summaries.append(
                        "s{step}:KL={kl:.3f},GT={gt:.4f},H1={h1:.2f},F1={f1:.2f},"
                        "yH={yh:.3f},yGT={ygt:.4f},rawGT={rgt:.4f},|raw|={ra:.3f},|fr|={fa:.3f},"
                        "m(d0/d1/d2/d3+)={d0:.3f}/{d1:.3f}/{d2:.3f}/{d3p:.3f},"
                        "top1(r/t/y)={tr}/{tt}/{ty}".format(
                            step=int(st.get("sup_step", 0)),
                            kl=float(st.get("kl", 0.0)),
                            gt=float(st.get("gt_mass", 0.0)),
                            h1=float(st.get("hit1", 0.0)),
                            f1=float(st.get("f1", 0.0)),
                            yh=float(frontier_diag.get("y_entropy", 0.0)),
                            ygt=float(frontier_diag.get("y_gt_mass", 0.0)),
                            rgt=float(frontier_diag.get("raw_gt_mass", 0.0)),
                            ra=float(frontier_diag.get("raw_abs_mean", 0.0)),
                            fa=float(frontier_diag.get("frontier_abs_mean", 0.0)),
                            d0=float(frontier_diag.get("y_mass_dist0", 0.0)),
                            d1=float(frontier_diag.get("y_mass_dist1", 0.0)),
                            d2=float(frontier_diag.get("y_mass_dist2", 0.0)),
                            d3p=float(frontier_diag.get("y_mass_dist3plus", 0.0)),
                            tr=int(frontier_diag.get("top1_cid_raw", -1)),
                            tt=int(frontier_diag.get("top1_cid_total", -1)),
                            ty=int(frontier_diag.get("top1_cid_y", -1)),
                        )
                    )
                else:
                    step_summaries.append(
                        "s{step}:KL={kl:.3f},GT={gt:.4f},H1={h1:.2f},F1={f1:.2f}".format(
                            step=int(st.get("sup_step", 0)),
                            kl=float(st.get("kl", 0.0)),
                            gt=float(st.get("gt_mass", 0.0)),
                            h1=float(st.get("hit1", 0.0)),
                            f1=float(st.get("f1", 0.0)),
                        )
                    )
            print(
                f"{log_prefix} ex={rec['example_index']} orig_id={rec['orig_id']} "
                f"steps={len(rec['supervision_steps'])} final_hit1={rec['final']['hit1']:.4f} "
                f"final_f1={rec['final']['f1']:.4f}"
            )
            if step_summaries:
                print(f"{log_prefix}   " + " | ".join(step_summaries))
        if len(records) >= max_examples:
            break

    if not records:
        print(f"{log_prefix} no records collected.")
        return records

    max_steps = max(len(r.get("supervision_steps", [])) for r in records)
    if max_steps > 0:
        for sup_step in range(1, max_steps + 1):
            vals_kl: List[float] = []
            vals_gt: List[float] = []
            vals_h1: List[float] = []
            vals_f1: List[float] = []
            vals_yh: List[float] = []
            vals_ygt: List[float] = []
            vals_rgt: List[float] = []
            vals_ra: List[float] = []
            vals_fa: List[float] = []
            vals_md0: List[float] = []
            vals_md1: List[float] = []
            vals_md2: List[float] = []
            vals_md3p: List[float] = []
            for rec in records:
                for st in rec.get("supervision_steps", []):
                    if int(st.get("sup_step", 0)) == sup_step:
                        vals_kl.append(float(st.get("kl", 0.0)))
                        vals_gt.append(float(st.get("gt_mass", 0.0)))
                        vals_h1.append(float(st.get("hit1", 0.0)))
                        vals_f1.append(float(st.get("f1", 0.0)))
                        frontier_diag = st.get("frontier_diag", None)
                        if isinstance(frontier_diag, dict):
                            vals_yh.append(float(frontier_diag.get("y_entropy", 0.0)))
                            vals_ygt.append(float(frontier_diag.get("y_gt_mass", 0.0)))
                            vals_rgt.append(float(frontier_diag.get("raw_gt_mass", 0.0)))
                            vals_ra.append(float(frontier_diag.get("raw_abs_mean", 0.0)))
                            vals_fa.append(float(frontier_diag.get("frontier_abs_mean", 0.0)))
                            vals_md0.append(float(frontier_diag.get("y_mass_dist0", 0.0)))
                            vals_md1.append(float(frontier_diag.get("y_mass_dist1", 0.0)))
                            vals_md2.append(float(frontier_diag.get("y_mass_dist2", 0.0)))
                            vals_md3p.append(float(frontier_diag.get("y_mass_dist3plus", 0.0)))
                        break
            if vals_kl:
                msg = (
                    f"{log_prefix} mean_s{sup_step}: "
                    f"KL={float(np.mean(vals_kl)):.4f} "
                    f"GT={float(np.mean(vals_gt)):.6f} "
                    f"H1={float(np.mean(vals_h1)):.4f} "
                    f"F1={float(np.mean(vals_f1)):.4f}"
                )
                if vals_yh:
                    msg += (
                        f" yH={float(np.mean(vals_yh)):.4f}"
                        f" yGT={float(np.mean(vals_ygt)):.6f}"
                        f" rawGT={float(np.mean(vals_rgt)):.6f}"
                        f" |raw|={float(np.mean(vals_ra)):.4f}"
                        f" |fr|={float(np.mean(vals_fa)):.4f}"
                        f" m(d0/d1/d2/d3+)={float(np.mean(vals_md0)):.4f}/"
                        f"{float(np.mean(vals_md1)):.4f}/"
                        f"{float(np.mean(vals_md2)):.4f}/"
                        f"{float(np.mean(vals_md3p)):.4f}"
                    )
                print(msg)

    dump_path = str(dump_jsonl or "").strip()
    if dump_path:
        dump_dir = os.path.dirname(dump_path)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{log_prefix} dumped {len(records)} examples to {dump_path}")

    plot_path = str(plot_png or "").strip()
    if not plot_path and dump_path:
        plot_path = os.path.splitext(dump_path)[0] + ".png"
    if not plot_path:
        return records

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"{log_prefix} matplotlib unavailable, skip plot: {e}")
        return records

    max_steps = max(len(r.get("supervision_steps", [])) for r in records)
    if max_steps <= 0:
        print(f"{log_prefix} no supervision steps to plot.")
        return records

    xs = list(range(1, max_steps + 1))
    kl_rows: List[List[float]] = []
    gt_rows: List[List[float]] = []
    hit_rows: List[List[float]] = []
    for rec in records:
        steps = rec.get("supervision_steps", [])
        kl_row = [float("nan")] * max_steps
        gt_row = [float("nan")] * max_steps
        hit_row = [float("nan")] * max_steps
        for i, st in enumerate(steps):
            if i >= max_steps:
                break
            kl_row[i] = float(st.get("kl", 0.0))
            gt_row[i] = float(st.get("gt_mass", 0.0))
            hit_row[i] = float(st.get("hit1", 0.0))
        kl_rows.append(kl_row)
        gt_rows.append(gt_row)
        hit_rows.append(hit_row)

    arr_kl = np.array(kl_rows, dtype=np.float64)
    arr_gt = np.array(gt_rows, dtype=np.float64)
    arr_hit = np.array(hit_rows, dtype=np.float64)
    mean_kl = np.nanmean(arr_kl, axis=0)
    mean_gt = np.nanmean(arr_gt, axis=0)
    mean_hit = np.nanmean(arr_hit, axis=0)

    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["KL(target||y_t)", "GT probability mass", "Top1 Hit@1"]
    ysets = [arr_kl, arr_gt, arr_hit]
    means = [mean_kl, mean_gt, mean_hit]
    for ax, title, ys, ymean in zip(axes, titles, ysets, means):
        for row in ys:
            ax.plot(xs, row, alpha=0.30, linewidth=1.0)
        ax.plot(xs, ymean, linewidth=2.2)
        ax.set_title(title)
        ax.set_xlabel("Supervision step")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"{log_prefix} plot saved to {plot_path}")
    return records


def _format_trace_output_path(path: str, ep: int) -> str:
    p = str(path or "").strip()
    if not p:
        return ""
    if "{ep}" in p:
        try:
            return p.format(ep=int(ep))
        except Exception:
            return p.replace("{ep}", str(int(ep)))
    root, ext = os.path.splitext(p)
    if ext:
        return f"{root}_ep{int(ep)}{ext}"
    return f"{p}_ep{int(ep)}"


def _safe_load_state_dict(model: nn.Module, sd: dict) -> Tuple[int, int]:
    if not isinstance(sd, dict):
        return 0, 0
    cur = model.state_dict()
    keep = {}
    skipped = 0
    for k, v in sd.items():
        if k not in cur:
            continue
        tgt = cur[k]
        try:
            same_shape = tuple(v.shape) == tuple(tgt.shape)
        except Exception:
            same_shape = False
        if same_shape:
            keep[k] = v
        else:
            skipped += 1
    missing, _ = model.load_state_dict(keep, strict=False)
    return int(skipped), int(len(missing))


def _infer_resume_epoch_from_ckpt(ckpt_path: str) -> int:
    if not ckpt_path:
        return 0
    base = os.path.basename(str(ckpt_path))
    m = re.search(r"model_ep(\d+)\.pt$", base)
    if not m:
        return 0
    try:
        return max(0, int(m.group(1)))
    except Exception:
        return 0


def _load_model_from_ckpt_or_init(
    ckpt_path: str,
    entity_dim: int,
    relation_dim: int,
    query_dim: int,
    hidden_size: int,
    recursion_steps: int,
    dropout: float,
    use_direction_embedding: bool = False,
    outer_reasoning_enabled: bool = False,
    outer_reasoning_steps: int = 3,
    gnn_variant: str = "rearev_bfs",
    rearev_num_instructions: int = 3,
    rearev_adapt_stages: int = 1,
    rearev_normalized_gnn: bool = False,
    rearev_latent_reasoning_enabled: bool = False,
    rearev_latent_residual_alpha: float = 0.25,
    rearev_latent_update_mode: str = "gru",
    rearev_global_gate_enabled: bool = False,
    rearev_logit_global_fusion_enabled: bool = False,
    rearev_dynamic_halting_enabled: bool = False,
    rearev_dynamic_halting_threshold: float = 0.9,
    rearev_dynamic_halting_min_steps: int = 1,
    rearev_trm_style_enabled: bool = False,
    rearev_trm_tminus1_no_grad: bool = True,
    rearev_trm_detach_carry: bool = True,
    rearev_trm_supervise_all_stages: bool = False,
    rearev_act_stop_in_train: bool = False,
    rearev_asymmetric_yz_enabled: bool = False,
    rearev_asym_inner_y_ema_enabled: bool = False,
    rearev_asym_inner_y_ema_alpha: float = 0.0,
    trm_rel_topk_relations: int = 0,
    trm_rel_score_alpha: float = 1.0,
    trm_rel_use_relid_policy: bool = True,
) -> Tuple[RecursiveSubgraphReader, Optional[dict]]:
    meta = None
    if ckpt_path and os.path.exists(ckpt_path):
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict):
            meta = obj
            model_cfg = obj.get("model_cfg", {})
            if isinstance(model_cfg, dict) and "recursion_steps" in model_cfg:
                recursion_steps = max(0, int(model_cfg.get("recursion_steps", recursion_steps)))
            if isinstance(model_cfg, dict) and "use_direction_embedding" in model_cfg:
                use_direction_embedding = _as_bool(model_cfg.get("use_direction_embedding", False))
            if isinstance(model_cfg, dict) and "outer_reasoning_enabled" in model_cfg:
                outer_reasoning_enabled = _as_bool(model_cfg.get("outer_reasoning_enabled", False))
            if isinstance(model_cfg, dict) and "outer_reasoning_steps" in model_cfg:
                outer_reasoning_steps = int(model_cfg.get("outer_reasoning_steps", 3))
            if isinstance(model_cfg, dict) and "gnn_variant" in model_cfg:
                gnn_variant = str(model_cfg.get("gnn_variant", "rearev_bfs"))
            if isinstance(model_cfg, dict) and "rearev_num_instructions" in model_cfg:
                rearev_num_instructions = int(model_cfg.get("rearev_num_instructions", 3))
            if isinstance(model_cfg, dict) and "rearev_adapt_stages" in model_cfg:
                rearev_adapt_stages = int(model_cfg.get("rearev_adapt_stages", 1))
            if isinstance(model_cfg, dict) and "rearev_normalized_gnn" in model_cfg:
                rearev_normalized_gnn = _as_bool(model_cfg.get("rearev_normalized_gnn", False))
            if isinstance(model_cfg, dict) and "rearev_latent_reasoning_enabled" in model_cfg:
                rearev_latent_reasoning_enabled = _as_bool(
                    model_cfg.get("rearev_latent_reasoning_enabled", False)
                )
            if isinstance(model_cfg, dict) and "rearev_latent_residual_alpha" in model_cfg:
                rearev_latent_residual_alpha = float(model_cfg.get("rearev_latent_residual_alpha", 0.25))
            if isinstance(model_cfg, dict) and "rearev_latent_update_mode" in model_cfg:
                rearev_latent_update_mode = str(model_cfg.get("rearev_latent_update_mode", "gru"))
            if isinstance(model_cfg, dict) and "rearev_global_gate_enabled" in model_cfg:
                rearev_global_gate_enabled = _as_bool(model_cfg.get("rearev_global_gate_enabled", False))
            if isinstance(model_cfg, dict) and "rearev_logit_global_fusion_enabled" in model_cfg:
                rearev_logit_global_fusion_enabled = _as_bool(
                    model_cfg.get("rearev_logit_global_fusion_enabled", False)
                )
            if isinstance(model_cfg, dict) and "rearev_dynamic_halting_enabled" in model_cfg:
                rearev_dynamic_halting_enabled = _as_bool(
                    model_cfg.get("rearev_dynamic_halting_enabled", False)
                )
            if isinstance(model_cfg, dict) and "rearev_dynamic_halting_threshold" in model_cfg:
                rearev_dynamic_halting_threshold = float(
                    model_cfg.get("rearev_dynamic_halting_threshold", 0.9)
                )
            if isinstance(model_cfg, dict) and "rearev_dynamic_halting_min_steps" in model_cfg:
                rearev_dynamic_halting_min_steps = int(
                    model_cfg.get("rearev_dynamic_halting_min_steps", 1)
                )
            if isinstance(model_cfg, dict) and "rearev_trm_style_enabled" in model_cfg:
                rearev_trm_style_enabled = _as_bool(model_cfg.get("rearev_trm_style_enabled", False))
            if isinstance(model_cfg, dict) and "rearev_trm_tminus1_no_grad" in model_cfg:
                rearev_trm_tminus1_no_grad = _as_bool(model_cfg.get("rearev_trm_tminus1_no_grad", True))
            if isinstance(model_cfg, dict) and "rearev_trm_detach_carry" in model_cfg:
                rearev_trm_detach_carry = _as_bool(model_cfg.get("rearev_trm_detach_carry", True))
            if isinstance(model_cfg, dict) and "rearev_trm_supervise_all_stages" in model_cfg:
                rearev_trm_supervise_all_stages = _as_bool(
                    model_cfg.get("rearev_trm_supervise_all_stages", False)
                )
            if isinstance(model_cfg, dict) and "rearev_act_stop_in_train" in model_cfg:
                rearev_act_stop_in_train = _as_bool(model_cfg.get("rearev_act_stop_in_train", False))
            if isinstance(model_cfg, dict) and "rearev_asymmetric_yz_enabled" in model_cfg:
                rearev_asymmetric_yz_enabled = _as_bool(
                    model_cfg.get("rearev_asymmetric_yz_enabled", False)
                )
            if isinstance(model_cfg, dict) and "rearev_asym_inner_y_ema_enabled" in model_cfg:
                rearev_asym_inner_y_ema_enabled = _as_bool(
                    model_cfg.get("rearev_asym_inner_y_ema_enabled", False)
                )
            if isinstance(model_cfg, dict) and "rearev_asym_inner_y_ema_alpha" in model_cfg:
                rearev_asym_inner_y_ema_alpha = float(
                    model_cfg.get("rearev_asym_inner_y_ema_alpha", 0.0)
                )
            if isinstance(model_cfg, dict) and "trm_rel_topk_relations" in model_cfg:
                trm_rel_topk_relations = int(model_cfg.get("trm_rel_topk_relations", 0))
            if isinstance(model_cfg, dict) and "trm_rel_score_alpha" in model_cfg:
                trm_rel_score_alpha = float(model_cfg.get("trm_rel_score_alpha", 1.0))
            if isinstance(model_cfg, dict) and "trm_rel_use_relid_policy" in model_cfg:
                trm_rel_use_relid_policy = _as_bool(
                    model_cfg.get("trm_rel_use_relid_policy", True), default=True
                )

    model = RecursiveSubgraphReader(
        entity_dim=entity_dim,
        relation_dim=relation_dim,
        query_dim=query_dim,
        hidden_size=hidden_size,
        recursion_steps=recursion_steps,
        dropout=dropout,
        use_direction_embedding=bool(use_direction_embedding),
        outer_reasoning_enabled=bool(outer_reasoning_enabled),
        outer_reasoning_steps=max(1, int(outer_reasoning_steps)),
        gnn_variant=str(gnn_variant),
        rearev_num_instructions=max(1, int(rearev_num_instructions)),
        rearev_adapt_stages=max(1, int(rearev_adapt_stages)),
        rearev_normalized_gnn=bool(rearev_normalized_gnn),
        rearev_latent_reasoning_enabled=bool(rearev_latent_reasoning_enabled),
        rearev_latent_residual_alpha=float(max(0.0, rearev_latent_residual_alpha)),
        rearev_latent_update_mode=str(rearev_latent_update_mode),
        rearev_global_gate_enabled=bool(rearev_global_gate_enabled),
        rearev_logit_global_fusion_enabled=bool(rearev_logit_global_fusion_enabled),
        rearev_dynamic_halting_enabled=bool(rearev_dynamic_halting_enabled),
        rearev_dynamic_halting_threshold=float(rearev_dynamic_halting_threshold),
        rearev_dynamic_halting_min_steps=max(1, int(rearev_dynamic_halting_min_steps)),
        rearev_trm_style_enabled=bool(rearev_trm_style_enabled),
        rearev_trm_tminus1_no_grad=bool(rearev_trm_tminus1_no_grad),
        rearev_trm_detach_carry=bool(rearev_trm_detach_carry),
        rearev_trm_supervise_all_stages=bool(rearev_trm_supervise_all_stages),
        rearev_act_stop_in_train=bool(rearev_act_stop_in_train),
        rearev_asymmetric_yz_enabled=bool(rearev_asymmetric_yz_enabled),
        rearev_asym_inner_y_ema_enabled=bool(rearev_asym_inner_y_ema_enabled),
        rearev_asym_inner_y_ema_alpha=float(rearev_asym_inner_y_ema_alpha),
        trm_rel_topk_relations=max(0, int(trm_rel_topk_relations)),
        trm_rel_score_alpha=float(max(0.0, trm_rel_score_alpha)),
        trm_rel_use_relid_policy=bool(trm_rel_use_relid_policy),
    )
    if ckpt_path and os.path.exists(ckpt_path):
        obj = meta if meta is not None else torch.load(ckpt_path, map_location="cpu")
        sd = obj.get("model_state", obj) if isinstance(obj, dict) else obj
        _safe_load_state_dict(model, sd)
    return model, meta


def train_subgraph_reader(
    args,
    *,
    is_ddp: bool,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    wb,
):
    is_main = rank == 0
    rel2idx = load_rel_map(args.relations_txt)

    hops = int(getattr(args, "subgraph_hops", 3))
    max_nodes = int(getattr(args, "subgraph_max_nodes", 256))
    max_edges = int(getattr(args, "subgraph_max_edges", 2048))
    add_reverse_edges = _as_bool(getattr(args, "subgraph_add_reverse_edges", False))
    recursion_steps = int(getattr(args, "subgraph_recursion_steps", 8))
    dropout = float(getattr(args, "subgraph_dropout", 0.1))
    threshold = float(getattr(args, "subgraph_pred_threshold", 0.5))
    loss_mode_raw = str(getattr(args, "subgraph_loss_mode", "rearev_kl")).strip().lower()
    if loss_mode_raw in {"rearev", "rearev_kl", "kl", "kl_div", "kld"}:
        loss_mode = "rearev_kl"
    elif loss_mode_raw in {"rearev_kl_rank", "kl_rank", "rearev_hybrid", "hybrid_kl_rank"}:
        loss_mode = "rearev_kl_rank"
    elif loss_mode_raw in {"rearev_kl_halt", "kl_halt", "hybrid_kl_halt"}:
        loss_mode = "rearev_kl_halt"
    elif loss_mode_raw in {"rearev_kl_trm", "kl_trm", "hybrid_kl_trm"}:
        loss_mode = "rearev_kl_trm"
    elif loss_mode_raw in {"rearev_trm", "trm", "trm_ce_halt", "trm_latent"}:
        loss_mode = "rearev_trm"
    elif loss_mode_raw in {"bce", "legacy_bce"}:
        loss_mode = "bce"
    else:
        loss_mode = "rearev_kl"
    pos_weight_mode = str(getattr(args, "subgraph_pos_weight_mode", "auto")).strip().lower()
    if pos_weight_mode not in {"auto", "fixed", "off", "none", "disabled"}:
        pos_weight_mode = "auto"
    fixed_pos_weight = float(getattr(args, "subgraph_pos_weight", 1.0))
    max_pos_weight = float(getattr(args, "subgraph_pos_weight_max", 256.0))
    max_pos_weight = max(1.0, max_pos_weight)
    split_reverse_relations = _as_bool(getattr(args, "subgraph_split_reverse_relations", False))
    direction_embedding_enabled = _as_bool(
        getattr(args, "subgraph_direction_embedding_enabled", split_reverse_relations),
        default=split_reverse_relations,
    )
    outer_reasoning_enabled = _as_bool(getattr(args, "subgraph_outer_reasoning_enabled", False))
    outer_reasoning_steps = max(1, int(getattr(args, "subgraph_outer_reasoning_steps", 3)))
    gnn_variant = str(getattr(args, "subgraph_gnn_variant", "rearev_bfs")).strip().lower()
    rearev_num_instructions = max(1, int(getattr(args, "subgraph_rearev_num_ins", 3)))
    rearev_adapt_stages = max(1, int(getattr(args, "subgraph_rearev_adapt_stages", 1)))
    rearev_normalized_gnn = _as_bool(getattr(args, "subgraph_rearev_normalized_gnn", False))
    rearev_latent_reasoning_enabled = _as_bool(
        getattr(args, "subgraph_rearev_latent_reasoning_enabled", False)
    )
    rearev_latent_residual_alpha = max(
        0.0, float(getattr(args, "subgraph_rearev_latent_residual_alpha", 0.25))
    )
    rearev_latent_update_mode = str(
        getattr(args, "subgraph_rearev_latent_update_mode", "gru")
    ).strip().lower()
    rearev_global_gate_enabled = _as_bool(
        getattr(args, "subgraph_rearev_global_gate_enabled", False)
    )
    rearev_logit_global_fusion_enabled = _as_bool(
        getattr(args, "subgraph_rearev_logit_global_fusion_enabled", False)
    )
    rearev_dynamic_halting_enabled = _as_bool(
        getattr(args, "subgraph_rearev_dynamic_halting_enabled", False)
    )
    rearev_dynamic_halting_threshold = float(
        getattr(args, "subgraph_rearev_dynamic_halting_threshold", 0.9)
    )
    rearev_dynamic_halting_min_steps = max(
        1, int(getattr(args, "subgraph_rearev_dynamic_halting_min_steps", 1))
    )
    rearev_trm_style_enabled = _as_bool(
        getattr(args, "subgraph_rearev_trm_style_enabled", False)
    )
    rearev_trm_tminus1_no_grad = _as_bool(
        getattr(args, "subgraph_rearev_trm_tminus1_no_grad", True), default=True
    )
    rearev_trm_detach_carry = _as_bool(
        getattr(args, "subgraph_rearev_trm_detach_carry", True), default=True
    )
    rearev_trm_supervise_all_stages = _as_bool(
        getattr(args, "subgraph_rearev_trm_supervise_all_stages", False)
    )
    rearev_act_stop_in_train = _as_bool(
        getattr(args, "subgraph_rearev_act_stop_in_train", False)
    )
    rearev_asymmetric_yz_enabled = _as_bool(
        getattr(args, "subgraph_rearev_asymmetric_yz_enabled", False)
    )
    rearev_asym_inner_y_ema_enabled = _as_bool(
        getattr(args, "subgraph_rearev_asym_inner_y_ema_enabled", False)
    )
    rearev_asym_inner_y_ema_alpha = float(
        min(1.0, max(0.0, float(getattr(args, "subgraph_rearev_asym_inner_y_ema_alpha", 0.0))))
    )
    trm_rel_topk_relations = max(
        0, int(getattr(args, "subgraph_trm_rel_topk_relations", 0))
    )
    trm_rel_score_alpha = float(
        max(0.0, float(getattr(args, "subgraph_trm_rel_score_alpha", 1.0)))
    )
    trm_rel_use_relid_policy = _as_bool(
        getattr(args, "subgraph_trm_rel_use_relid_policy", True), default=True
    )
    rearev_trm_halt_bce_weight = max(
        0.0, float(getattr(args, "subgraph_rearev_trm_halt_bce_weight", 1.0))
    )
    rearev_trm_ce_weight = max(
        0.0, float(getattr(args, "subgraph_rearev_trm_ce_weight", 1.0))
    )
    rearev_trm_weight = max(
        0.0, float(getattr(args, "subgraph_rearev_trm_weight", 1.0))
    )
    deep_supervision_enabled = _as_bool(
        getattr(args, "subgraph_deep_supervision_enabled", False)
    )
    deep_supervision_weight = max(
        0.0, float(getattr(args, "subgraph_deep_supervision_weight", 0.0))
    )
    deep_supervision_ce_weight = max(
        0.0, float(getattr(args, "subgraph_deep_supervision_ce_weight", 1.0))
    )
    deep_supervision_halt_weight = max(
        0.0, float(getattr(args, "subgraph_deep_supervision_halt_weight", 1.0))
    )
    ranking_enabled = _as_bool(getattr(args, "subgraph_ranking_enabled", False))
    ranking_weight = max(0.0, float(getattr(args, "subgraph_ranking_weight", 0.0)))
    ranking_margin = float(getattr(args, "subgraph_ranking_margin", 0.2))
    hard_negative_topk = max(1, int(getattr(args, "subgraph_hard_negative_topk", 16)))
    bce_hard_negative_enabled = _as_bool(getattr(args, "subgraph_bce_hard_negative_enabled", False))
    bce_hard_negative_topk = max(1, int(getattr(args, "subgraph_bce_hard_negative_topk", 64)))
    kl_no_positive_mode = str(getattr(args, "subgraph_kl_no_positive_mode", "uniform")).strip().lower()
    if kl_no_positive_mode not in {"uniform", "skip", "mask", "drop"}:
        kl_no_positive_mode = "uniform"
    kl_supervision_mode = str(getattr(args, "subgraph_kl_supervision_mode", "final")).strip().lower()
    if kl_supervision_mode not in {"final", "step_uniform"}:
        kl_supervision_mode = "final"
    uses_kl_objective = loss_mode in {"rearev_kl", "rearev_kl_rank"}
    uses_kl_halt_objective = loss_mode == "rearev_kl_halt"
    uses_kl_trm_objective = loss_mode == "rearev_kl_trm"
    uses_trm_objective = loss_mode == "rearev_trm"
    deep_supervision_active = bool(deep_supervision_enabled) and float(deep_supervision_weight) > 0.0
    uses_kl_deep_supervision = (
        deep_supervision_active and loss_mode in {"rearev_kl", "rearev_kl_rank"}
    )
    uses_kl_step_uniform = (
        (uses_kl_objective or uses_kl_halt_objective)
        and kl_supervision_mode == "step_uniform"
    )
    if uses_trm_objective:
        rearev_trm_style_enabled = True
        ranking_enabled = False
        ranking_weight = 0.0
        bce_hard_negative_enabled = False
        pos_weight_mode = "off"
        fixed_pos_weight = 1.0
        max_pos_weight = 1.0
    elif uses_kl_halt_objective:
        rearev_trm_style_enabled = True
        ranking_enabled = False
        ranking_weight = 0.0
        bce_hard_negative_enabled = False
        pos_weight_mode = "off"
        fixed_pos_weight = 1.0
        max_pos_weight = 1.0
    elif uses_kl_trm_objective:
        rearev_trm_style_enabled = True
        ranking_enabled = False
        ranking_weight = 0.0
        bce_hard_negative_enabled = False
        pos_weight_mode = "off"
        fixed_pos_weight = 1.0
        max_pos_weight = 1.0
    else:
        # Legacy KL can optionally add TRM-style deep supervision as an auxiliary term.
        rearev_trm_style_enabled = bool(uses_kl_deep_supervision or uses_kl_step_uniform)
    if uses_kl_step_uniform or uses_kl_halt_objective:
        # Step-wise KL needs per-step logits; force full stage supervision rows.
        rearev_trm_supervise_all_stages = True
    if uses_kl_objective:
        # Keep training behavior aligned with ReaRev objective.
        pos_weight_mode = "off"
        fixed_pos_weight = 1.0
        max_pos_weight = 1.0
        if loss_mode == "rearev_kl":
            ranking_enabled = False
            ranking_weight = 0.0
        bce_hard_negative_enabled = False
    lr_scheduler_mode = str(getattr(args, "subgraph_lr_scheduler", "none")).strip().lower()
    if lr_scheduler_mode not in {"none", "off", "disabled", "cosine", "step", "plateau"}:
        lr_scheduler_mode = "none"
    lr_min = max(0.0, float(getattr(args, "subgraph_lr_min", 0.0)))
    lr_step_size = max(1, int(getattr(args, "subgraph_lr_step_size", 5)))
    lr_gamma = float(getattr(args, "subgraph_lr_gamma", 0.5))
    lr_plateau_factor = float(getattr(args, "subgraph_lr_plateau_factor", 0.5))
    lr_plateau_patience = max(0, int(getattr(args, "subgraph_lr_plateau_patience", 2)))
    lr_plateau_threshold = float(getattr(args, "subgraph_lr_plateau_threshold", 1e-4))
    lr_plateau_metric = str(getattr(args, "subgraph_lr_plateau_metric", "train_loss")).strip().lower()
    if lr_plateau_metric not in {"train_loss", "dev_hit1", "dev_f1"}:
        lr_plateau_metric = "train_loss"
    early_stop_enabled = _as_bool(getattr(args, "subgraph_early_stop_enabled", False))
    early_stop_metric = str(getattr(args, "subgraph_early_stop_metric", "dev_hit1")).strip().lower()
    if early_stop_metric not in {"train_loss", "dev_hit1", "dev_f1"}:
        early_stop_metric = "dev_hit1"
    early_stop_patience = max(0, int(getattr(args, "subgraph_early_stop_patience", 0)))
    early_stop_min_delta = float(getattr(args, "subgraph_early_stop_min_delta", 1e-4))
    early_stop_min_epochs = max(1, int(getattr(args, "subgraph_early_stop_min_epochs", 1)))
    trace_supervision_enabled = _as_bool(
        getattr(args, "subgraph_trace_supervision_enabled", False)
    )
    trace_supervision_examples = max(
        1, int(getattr(args, "subgraph_trace_supervision_examples", 5))
    )
    trace_supervision_dump_jsonl = str(
        getattr(args, "subgraph_trace_supervision_dump_jsonl", "")
    )
    trace_supervision_plot_png = str(
        getattr(args, "subgraph_trace_supervision_plot_png", "")
    )
    grad_accum_steps = max(1, int(getattr(args, "subgraph_grad_accum_steps", 1)))
    resume_epoch_cfg = int(getattr(args, "subgraph_resume_epoch", -1))
    if resume_epoch_cfg >= 0:
        resume_epoch = int(resume_epoch_cfg)
    else:
        resume_epoch = _infer_resume_epoch_from_ckpt(getattr(args, "ckpt", ""))

    train_ds = SubgraphExampleDataset(args.train_json)
    if len(train_ds) <= 0:
        raise RuntimeError(f"Empty subgraph reader train dataset: {args.train_json}")

    train_collate = SubgraphCollator(
        entity_emb_npy=args.entity_emb_npy,
        relation_emb_npy=args.relation_emb_npy,
        query_emb_npy=args.query_emb_train_npy,
        rel2idx=rel2idx,
        hops=hops,
        max_nodes=max_nodes,
        max_edges=max_edges,
        add_reverse_edges=add_reverse_edges,
        split_reverse_relations=split_reverse_relations,
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None

    # Subgraph collate uses large mmap arrays; keep workers=0 for stability.
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=train_collate,
        pin_memory=torch.cuda.is_available(),
    )
    if len(loader) <= 0:
        raise RuntimeError("Empty subgraph reader train loader.")

    model, _ = _load_model_from_ckpt_or_init(
        ckpt_path=getattr(args, "ckpt", ""),
        entity_dim=train_collate.entity_dim,
        relation_dim=train_collate.relation_dim,
        query_dim=train_collate.query_dim,
        hidden_size=int(args.hidden_size),
        recursion_steps=recursion_steps,
        dropout=dropout,
        use_direction_embedding=direction_embedding_enabled,
        outer_reasoning_enabled=outer_reasoning_enabled,
        outer_reasoning_steps=outer_reasoning_steps,
        gnn_variant=gnn_variant,
        rearev_num_instructions=rearev_num_instructions,
        rearev_adapt_stages=rearev_adapt_stages,
        rearev_normalized_gnn=rearev_normalized_gnn,
        rearev_latent_reasoning_enabled=rearev_latent_reasoning_enabled,
        rearev_latent_residual_alpha=rearev_latent_residual_alpha,
        rearev_latent_update_mode=rearev_latent_update_mode,
        rearev_global_gate_enabled=rearev_global_gate_enabled,
        rearev_logit_global_fusion_enabled=rearev_logit_global_fusion_enabled,
        rearev_dynamic_halting_enabled=rearev_dynamic_halting_enabled,
        rearev_dynamic_halting_threshold=rearev_dynamic_halting_threshold,
        rearev_dynamic_halting_min_steps=rearev_dynamic_halting_min_steps,
        rearev_trm_style_enabled=rearev_trm_style_enabled,
        rearev_trm_tminus1_no_grad=rearev_trm_tminus1_no_grad,
        rearev_trm_detach_carry=rearev_trm_detach_carry,
        rearev_trm_supervise_all_stages=rearev_trm_supervise_all_stages,
        rearev_act_stop_in_train=rearev_act_stop_in_train,
        rearev_asymmetric_yz_enabled=rearev_asymmetric_yz_enabled,
        rearev_asym_inner_y_ema_enabled=rearev_asym_inner_y_ema_enabled,
        rearev_asym_inner_y_ema_alpha=rearev_asym_inner_y_ema_alpha,
        trm_rel_topk_relations=trm_rel_topk_relations,
        trm_rel_score_alpha=trm_rel_score_alpha,
        trm_rel_use_relid_policy=trm_rel_use_relid_policy,
    )
    model.to(device)
    ddp_find_unused_default = bool(
        (
            rearev_trm_style_enabled
            and rearev_trm_tminus1_no_grad
            and int(recursion_steps) > 1
        )
        # Frontier/TRM-recursive variants keep optional heads that may not
        # participate in the active loss path (e.g., final-only KL), so DDP
        # must track unused parameters by default.
        or gnn_variant in {"trm_frontier_recursive", "trm_frontier_rearev1", "trm_rel_recursive"}
    )
    ddp_find_unused = _as_bool(
        getattr(args, "subgraph_ddp_find_unused_parameters", ddp_find_unused_default),
        default=ddp_find_unused_default,
    )
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=bool(ddp_find_unused),
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=float(args.lr))
    scheduler = None
    if lr_scheduler_mode == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, int(args.epochs)),
            eta_min=float(lr_min),
        )
    elif lr_scheduler_mode == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=int(lr_step_size),
            gamma=float(lr_gamma),
        )
    elif lr_scheduler_mode == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max" if lr_plateau_metric in {"dev_hit1", "dev_f1"} else "min",
            factor=float(lr_plateau_factor),
            patience=int(lr_plateau_patience),
            threshold=float(lr_plateau_threshold),
        )

    has_dev = bool(getattr(args, "dev_json", "")) and os.path.exists(getattr(args, "dev_json", ""))
    dev_loader = None
    if has_dev:
        dev_ds = SubgraphExampleDataset(args.dev_json)
        eval_limit = int(getattr(args, "eval_limit", -1))
        if eval_limit > 0 and len(dev_ds) > eval_limit:
            dev_ds = Subset(dev_ds, list(range(eval_limit)))
        dev_collate = SubgraphCollator(
            entity_emb_npy=args.entity_emb_npy,
            relation_emb_npy=args.relation_emb_npy,
            query_emb_npy=getattr(args, "query_emb_dev_npy", ""),
            rel2idx=rel2idx,
            hops=hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            add_reverse_edges=add_reverse_edges,
            split_reverse_relations=split_reverse_relations,
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=dev_collate,
            pin_memory=torch.cuda.is_available(),
        )

    if is_main:
        print(
            "[SubgraphReader] "
            f"hops={hops} recursion_steps={recursion_steps} max_nodes={max_nodes} max_edges={max_edges} "
            f"reverse_edges={add_reverse_edges} "
            f"split_reverse_relations={split_reverse_relations} "
            f"direction_embedding={direction_embedding_enabled} "
            f"outer_reasoning={outer_reasoning_enabled} "
            f"outer_steps={outer_reasoning_steps} "
            f"gnn_variant={gnn_variant} "
            f"trm_rel_topk={trm_rel_topk_relations} "
            f"trm_rel_alpha={trm_rel_score_alpha:.3f} "
            f"trm_rel_relid_policy={trm_rel_use_relid_policy} "
            f"loss_mode={loss_mode} "
            f"rearev_num_ins={rearev_num_instructions} "
            f"rearev_adapt_stages={rearev_adapt_stages} "
            f"rearev_normalized_gnn={rearev_normalized_gnn} "
            f"rearev_latent={rearev_latent_reasoning_enabled} "
            f"rearev_latent_alpha={rearev_latent_residual_alpha:.3f} "
            f"rearev_latent_update={rearev_latent_update_mode} "
            f"rearev_gate={rearev_global_gate_enabled} "
            f"rearev_logit_fusion={rearev_logit_global_fusion_enabled} "
            f"rearev_dyn_halt={rearev_dynamic_halting_enabled} "
            f"rearev_halt_thr={rearev_dynamic_halting_threshold:.3f} "
            f"rearev_halt_min={rearev_dynamic_halting_min_steps} "
            f"rearev_trm_style={rearev_trm_style_enabled} "
            f"rearev_trm_tminus1_nograd={rearev_trm_tminus1_no_grad} "
            f"rearev_trm_detach_carry={rearev_trm_detach_carry} "
            f"rearev_trm_all_stages={rearev_trm_supervise_all_stages} "
            f"rearev_act_stop_train={rearev_act_stop_in_train} "
            f"rearev_asym_yz={rearev_asymmetric_yz_enabled} "
            f"rearev_asym_y_ema={rearev_asym_inner_y_ema_enabled} "
            f"rearev_asym_y_ema_alpha={rearev_asym_inner_y_ema_alpha:.3f} "
            f"rearev_trm_halt_w={rearev_trm_halt_bce_weight:.3f} "
            f"rearev_trm_ce_w={rearev_trm_ce_weight:.3f} "
            f"rearev_trm_w={rearev_trm_weight:.3f} "
            f"deep_sup={deep_supervision_enabled} "
            f"deep_sup_w={deep_supervision_weight:.3f} "
            f"deep_sup_ce_w={deep_supervision_ce_weight:.3f} "
            f"deep_sup_halt_w={deep_supervision_halt_weight:.3f} "
            f"kl_no_pos={kl_no_positive_mode} "
            f"kl_sup={kl_supervision_mode} "
            f"ddp_find_unused={ddp_find_unused} "
            f"early_stop={early_stop_enabled} "
            f"early_stop_metric={early_stop_metric} "
            f"early_stop_patience={early_stop_patience} "
            f"early_stop_min_delta={early_stop_min_delta:.6g} "
            f"early_stop_min_epochs={early_stop_min_epochs} "
            f"trace_sup={trace_supervision_enabled} "
            f"trace_sup_examples={trace_supervision_examples} "
            f"pos_weight_mode={pos_weight_mode} "
            f"ranking_enabled={ranking_enabled} "
            f"bce_hardneg={bce_hard_negative_enabled} "
            f"lr={float(args.lr):.3e} scheduler={lr_scheduler_mode} "
            f"grad_accum={grad_accum_steps}"
        )
        if resume_epoch > 0:
            print(f"[SubgraphReader-Resume] start_from_ep={resume_epoch}")

    early_best_metric = None
    early_bad_epochs = 0

    progress_single_line = _as_bool(getattr(args, "progress_single_line", False), default=False)
    progress_log_every = max(1, int(getattr(args, "progress_log_every", 50)))
    progress_mininterval = max(0.5, float(getattr(args, "progress_mininterval", 1.0)))

    for local_ep in range(1, int(args.epochs) + 1):
        ep = int(resume_epoch + local_ep)
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        pbar = loader if progress_single_line else tqdm(
            loader,
            disable=not is_main,
            desc=f"Ep {ep} [Subgraph]",
            dynamic_ncols=True,
            mininterval=progress_mininterval,
        )
        tot_loss = 0.0
        tot_obj_loss = 0.0
        tot_halt_loss = 0.0
        tot_rank_loss = 0.0
        steps = 0
        rank_pairs_sum = 0
        obj_aux_sum = 0
        halt_aux_sum = 0
        optimizer_steps = 0
        opt.zero_grad(set_to_none=True)
        last_grad_norm = 0.0
        num_batches = len(loader)
        last_progress_chars = 0

        for batch_idx, batch in enumerate(pbar, start=1):
            batch_dev = _move_batch_to_device(batch, device)
            model_out = model(
                node_emb=batch_dev["node_emb"],
                node_mask=batch_dev["node_mask"],
                seed_mask=batch_dev.get("seed_mask", None),
                edge_src=batch_dev["edge_src"],
                edge_dst=batch_dev["edge_dst"],
                edge_rel_emb=batch_dev["edge_rel_emb"],
                edge_rel_ids=batch_dev.get("edge_rel_ids", None),
                edge_dir=batch_dev.get("edge_dir", None),
                edge_mask=batch_dev["edge_mask"],
                q_emb=batch_dev["q_emb"],
                return_aux=(
                    uses_trm_objective
                    or uses_kl_halt_objective
                    or uses_kl_trm_objective
                    or uses_kl_deep_supervision
                    or uses_kl_step_uniform
                ),
            )
            if (
                uses_trm_objective
                or uses_kl_halt_objective
                or uses_kl_trm_objective
                or uses_kl_deep_supervision
                or uses_kl_step_uniform
            ):
                logits = model_out["logits"]
                step_logits = model_out["step_logits"]
                step_halt_logits = model_out["step_halt_logits"]
                step_valid_mask = model_out["step_valid_mask"]
            else:
                logits = model_out
                step_logits = None
                step_halt_logits = None
                step_valid_mask = None

            halt_loss = logits.new_tensor(0.0)
            halt_aux = 0
            kl_component = logits.new_tensor(0.0)
            trm_ce_component = logits.new_tensor(0.0)
            if uses_trm_objective:
                ce_loss, obj_aux = _masked_trm_step_ce_loss(
                    step_logits=step_logits,
                    targets=batch_dev["node_labels"],
                    mask=batch_dev["node_mask"],
                    step_valid_mask=step_valid_mask,
                )
                halt_loss, halt_aux = _trm_halt_bce_loss(
                    step_halt_logits=step_halt_logits,
                    step_valid_mask=step_valid_mask,
                )
                trm_ce_component = ce_loss
                obj_loss = (
                    float(rearev_trm_ce_weight) * trm_ce_component
                    + float(rearev_trm_halt_bce_weight) * halt_loss
                )
                step_pos_weight = 1.0
            elif uses_kl_halt_objective:
                if uses_kl_step_uniform:
                    kl_component, kl_aux = _masked_rearev_step_kl_loss(
                        step_logits=step_logits,
                        targets=batch_dev["node_labels"],
                        mask=batch_dev["node_mask"],
                        step_valid_mask=step_valid_mask,
                        no_positive_mode=kl_no_positive_mode,
                    )
                else:
                    kl_component, kl_aux = _masked_rearev_kl_loss(
                        logits=logits,
                        targets=batch_dev["node_labels"],
                        mask=batch_dev["node_mask"],
                        no_positive_mode=kl_no_positive_mode,
                    )
                halt_loss, halt_aux = _trm_halt_bce_loss(
                    step_halt_logits=step_halt_logits,
                    step_valid_mask=step_valid_mask,
                )
                obj_loss = kl_component + float(rearev_trm_halt_bce_weight) * halt_loss
                obj_aux = int(kl_aux)
                step_pos_weight = 1.0
            elif uses_kl_trm_objective:
                kl_component, kl_aux = _masked_rearev_kl_loss(
                    logits=logits,
                    targets=batch_dev["node_labels"],
                    mask=batch_dev["node_mask"],
                    no_positive_mode=kl_no_positive_mode,
                )
                trm_ce_component, trm_aux = _masked_trm_step_ce_loss(
                    step_logits=step_logits,
                    targets=batch_dev["node_labels"],
                    mask=batch_dev["node_mask"],
                    step_valid_mask=step_valid_mask,
                )
                halt_loss, halt_aux = _trm_halt_bce_loss(
                    step_halt_logits=step_halt_logits,
                    step_valid_mask=step_valid_mask,
                )
                trm_term = (
                    float(rearev_trm_ce_weight) * trm_ce_component
                    + float(rearev_trm_halt_bce_weight) * halt_loss
                )
                obj_loss = kl_component + float(rearev_trm_weight) * trm_term
                # Keep existing aggregation field semantics as "primary objective valid rows".
                obj_aux = int(kl_aux)
                halt_aux += int(trm_aux)
                step_pos_weight = 1.0
            elif uses_kl_objective:
                if uses_kl_step_uniform:
                    obj_loss, obj_aux = _masked_rearev_step_kl_loss(
                        step_logits=step_logits,
                        targets=batch_dev["node_labels"],
                        mask=batch_dev["node_mask"],
                        step_valid_mask=step_valid_mask,
                        no_positive_mode=kl_no_positive_mode,
                    )
                else:
                    obj_loss, obj_aux = _masked_rearev_kl_loss(
                        logits=logits,
                        targets=batch_dev["node_labels"],
                        mask=batch_dev["node_mask"],
                        no_positive_mode=kl_no_positive_mode,
                    )
                kl_component = obj_loss
                if uses_kl_deep_supervision:
                    trm_ce_component, trm_aux = _masked_trm_step_ce_loss(
                        step_logits=step_logits,
                        targets=batch_dev["node_labels"],
                        mask=batch_dev["node_mask"],
                        step_valid_mask=step_valid_mask,
                    )
                    halt_loss, halt_aux = _trm_halt_bce_loss(
                        step_halt_logits=step_halt_logits,
                        step_valid_mask=step_valid_mask,
                    )
                    trm_term = (
                        float(deep_supervision_ce_weight) * trm_ce_component
                        + float(deep_supervision_halt_weight) * halt_loss
                    )
                    obj_loss = kl_component + (float(deep_supervision_weight) * trm_term)
                    halt_aux += int(trm_aux)
                step_pos_weight = 1.0
            else:
                pos_weight_t = None
                step_pos_weight = 1.0
                if pos_weight_mode == "fixed":
                    step_pos_weight = max(1.0, float(fixed_pos_weight))
                    pos_weight_t = torch.tensor(step_pos_weight, dtype=logits.dtype, device=logits.device)
                elif pos_weight_mode in {"off", "none", "disabled"}:
                    pos_weight_t = None
                    step_pos_weight = 1.0
                else:
                    # auto mode: rebalance BCE by current masked positive/negative ratio.
                    # This prevents easy all-negative minima when positive labels are sparse.
                    with torch.no_grad():
                        m = batch_dev["node_mask"].to(torch.float32)
                        y = batch_dev["node_labels"].to(torch.float32)
                        pos = (y * m).sum()
                        tot = m.sum()
                        neg = (tot - pos).clamp(min=0.0)
                        if float(pos.item()) > 0.0:
                            step_pos_weight = float((neg / pos).item())
                        else:
                            step_pos_weight = 1.0
                    step_pos_weight = float(max(1.0, min(max_pos_weight, step_pos_weight)))
                    pos_weight_t = torch.tensor(step_pos_weight, dtype=logits.dtype, device=logits.device)

                obj_loss, obj_aux = _masked_bce_hard_negative_loss(
                    logits,
                    batch_dev["node_labels"],
                    batch_dev["node_mask"],
                    pos_weight=pos_weight_t,
                    hard_negative_enabled=bce_hard_negative_enabled,
                    hard_negative_topk=bce_hard_negative_topk,
                )
            rank_loss = logits.new_tensor(0.0)
            rank_pairs = 0
            if ranking_enabled and ranking_weight > 0.0:
                rank_loss, rank_pairs = _ranking_hard_negative_loss(
                    logits=logits,
                    targets=batch_dev["node_labels"],
                    mask=batch_dev["node_mask"],
                    margin=ranking_margin,
                    hard_negative_topk=hard_negative_topk,
                )
            loss = obj_loss + (float(ranking_weight) * rank_loss)

            if not torch.isfinite(loss):
                if is_ddp:
                    raise RuntimeError("non-finite subgraph reader loss in DDP")
                continue
            do_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == num_batches)
            scaled_loss = loss / float(grad_accum_steps)
            if not bool(scaled_loss.requires_grad):
                # Can happen with KL no-positive skip mode when a full mini-batch has no positives.
                if is_main and (batch_idx <= 3 or (batch_idx % 1000 == 0)):
                    print(
                        "[warn] skip backward for no-grad loss "
                        f"(batch_idx={batch_idx}, loss_mode={loss_mode}, kl_no_pos={kl_no_positive_mode})"
                    )
                continue
            if is_ddp and (not do_step):
                with model.no_sync():
                    scaled_loss.backward()
                grad_norm_val = float(last_grad_norm)
            else:
                scaled_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                grad_norm_val = float(grad_norm)
                last_grad_norm = grad_norm_val
                optimizer_steps += 1

            steps += 1
            tot_loss += float(loss.item())
            tot_obj_loss += float(obj_loss.item())
            tot_halt_loss += float(halt_loss.item())
            tot_rank_loss += float(rank_loss.item())
            rank_pairs_sum += int(rank_pairs)
            obj_aux_sum += int(obj_aux)
            halt_aux_sum += int(halt_aux)

            should_update_progress = (
                steps == 1
                or steps == num_batches
                or (progress_log_every > 0 and (steps % progress_log_every) == 0)
            )

            if is_main:
                if uses_trm_objective:
                    if should_update_progress and not progress_single_line:
                        pbar.set_postfix_str(
                            f"loss={loss.item():.4f} ce+halt={obj_loss.item():.4f} "
                            f"halt={halt_loss.item():.4f} avg={tot_loss/max(1,steps):.4f} "
                            f"grad={grad_norm_val:.2e}"
                        )
                elif uses_kl_halt_objective:
                    if should_update_progress and not progress_single_line:
                        pbar.set_postfix_str(
                            f"loss={loss.item():.4f} kl={kl_component.item():.4f} "
                            f"halt={halt_loss.item():.4f} avg={tot_loss/max(1,steps):.4f} "
                            f"grad={grad_norm_val:.2e}"
                        )
                elif uses_kl_trm_objective:
                    if should_update_progress and not progress_single_line:
                        pbar.set_postfix_str(
                            f"loss={loss.item():.4f} kl={kl_component.item():.4f} "
                            f"trm_ce={trm_ce_component.item():.4f} halt={halt_loss.item():.4f} "
                            f"avg={tot_loss/max(1,steps):.4f} grad={grad_norm_val:.2e}"
                        )
                elif uses_kl_objective and uses_kl_deep_supervision:
                    if should_update_progress and not progress_single_line:
                        pbar.set_postfix_str(
                            f"loss={loss.item():.4f} kl={kl_component.item():.4f} "
                            f"trm_ce={trm_ce_component.item():.4f} halt={halt_loss.item():.4f} "
                            f"rank={rank_loss.item():.4f} avg={tot_loss/max(1,steps):.4f} "
                            f"grad={grad_norm_val:.2e}"
                        )
                elif uses_kl_objective:
                    if should_update_progress and not progress_single_line:
                        pbar.set_postfix_str(
                            f"loss={loss.item():.4f} kl={obj_loss.item():.4f} rank={rank_loss.item():.4f} "
                            f"avg={tot_loss/max(1,steps):.4f} grad={grad_norm_val:.2e}"
                        )
                else:
                    if should_update_progress and not progress_single_line:
                        pbar.set_postfix_str(
                            f"loss={loss.item():.4f} bce={obj_loss.item():.4f} rank={rank_loss.item():.4f} "
                            f"avg={tot_loss/max(1,steps):.4f} pw={step_pos_weight:.2f} grad={grad_norm_val:.2e}"
                        )
                if should_update_progress and progress_single_line:
                    if uses_trm_objective:
                        msg = (
                            f"Ep {ep} [Subgraph] {steps}/{num_batches} ({(100.0 * steps) / max(1, num_batches):.1f}%) "
                            f"loss={loss.item():.4f} ce+halt={obj_loss.item():.4f} "
                            f"halt={halt_loss.item():.4f} avg={tot_loss/max(1,steps):.4f} "
                            f"grad={grad_norm_val:.2e}"
                        )
                    elif uses_kl_halt_objective:
                        msg = (
                            f"Ep {ep} [Subgraph] {steps}/{num_batches} ({(100.0 * steps) / max(1, num_batches):.1f}%) "
                            f"loss={loss.item():.4f} kl={kl_component.item():.4f} "
                            f"halt={halt_loss.item():.4f} avg={tot_loss/max(1,steps):.4f} "
                            f"grad={grad_norm_val:.2e}"
                        )
                    elif uses_kl_trm_objective:
                        msg = (
                            f"Ep {ep} [Subgraph] {steps}/{num_batches} ({(100.0 * steps) / max(1, num_batches):.1f}%) "
                            f"loss={loss.item():.4f} kl={kl_component.item():.4f} "
                            f"trm_ce={trm_ce_component.item():.4f} halt={halt_loss.item():.4f} "
                            f"avg={tot_loss/max(1,steps):.4f} grad={grad_norm_val:.2e}"
                        )
                    elif uses_kl_objective and uses_kl_deep_supervision:
                        msg = (
                            f"Ep {ep} [Subgraph] {steps}/{num_batches} ({(100.0 * steps) / max(1, num_batches):.1f}%) "
                            f"loss={loss.item():.4f} kl={kl_component.item():.4f} "
                            f"trm_ce={trm_ce_component.item():.4f} halt={halt_loss.item():.4f} "
                            f"rank={rank_loss.item():.4f} avg={tot_loss/max(1,steps):.4f} "
                            f"grad={grad_norm_val:.2e}"
                        )
                    elif uses_kl_objective:
                        msg = (
                            f"Ep {ep} [Subgraph] {steps}/{num_batches} ({(100.0 * steps) / max(1, num_batches):.1f}%) "
                            f"loss={loss.item():.4f} kl={obj_loss.item():.4f} rank={rank_loss.item():.4f} "
                            f"avg={tot_loss/max(1,steps):.4f} grad={grad_norm_val:.2e}"
                        )
                    else:
                        msg = (
                            f"Ep {ep} [Subgraph] {steps}/{num_batches} ({(100.0 * steps) / max(1, num_batches):.1f}%) "
                            f"loss={loss.item():.4f} bce={obj_loss.item():.4f} rank={rank_loss.item():.4f} "
                            f"avg={tot_loss/max(1,steps):.4f} pw={step_pos_weight:.2f} grad={grad_norm_val:.2e}"
                        )
                    last_progress_chars = _progress_write_line(msg, last_progress_chars)
                if wb is not None:
                    step_log = {
                        "train/step_loss": float(loss.item()),
                        "train/step_rank_loss": float(rank_loss.item()),
                        "train/step_avg_loss": float(tot_loss / max(1, steps)),
                        "train/step_rank_pairs": int(rank_pairs),
                        "train/grad_norm": float(grad_norm_val),
                        "train/step_optimizer_steps": int(optimizer_steps),
                        "train/epoch": int(ep),
                        "train/step": int(steps),
                    }
                    if uses_trm_objective:
                        step_log["train/step_trm_ce_halt_loss"] = float(obj_loss.item())
                        step_log["train/step_trm_halt_loss"] = float(halt_loss.item())
                        step_log["train/step_trm_halt_valid_steps"] = int(halt_aux)
                        step_log["train/step_trm_ce_valid_rows"] = int(obj_aux)
                    elif uses_kl_halt_objective:
                        step_log["train/step_kl_halt_loss"] = float(obj_loss.item())
                        step_log["train/step_kl_component"] = float(kl_component.item())
                        step_log["train/step_halt_component"] = float(halt_loss.item())
                        step_log["train/step_kl_valid_rows"] = int(obj_aux)
                        step_log["train/step_halt_valid_steps"] = int(halt_aux)
                    elif uses_kl_trm_objective:
                        step_log["train/step_kl_trm_loss"] = float(obj_loss.item())
                        step_log["train/step_kl_component"] = float(kl_component.item())
                        step_log["train/step_trm_ce_component"] = float(trm_ce_component.item())
                        step_log["train/step_trm_halt_component"] = float(halt_loss.item())
                        step_log["train/step_kl_valid_rows"] = int(obj_aux)
                        step_log["train/step_trm_halt_valid_steps"] = int(halt_aux)
                    elif uses_kl_objective and uses_kl_deep_supervision:
                        step_log["train/step_kl_ds_loss"] = float(obj_loss.item())
                        step_log["train/step_kl_component"] = float(kl_component.item())
                        step_log["train/step_ds_trm_ce_component"] = float(trm_ce_component.item())
                        step_log["train/step_ds_halt_component"] = float(halt_loss.item())
                        step_log["train/step_kl_valid_rows"] = int(obj_aux)
                        step_log["train/step_ds_halt_valid_steps"] = int(halt_aux)
                    elif uses_kl_objective:
                        step_log["train/step_kl_loss"] = float(obj_loss.item())
                        step_log["train/step_kl_valid_rows"] = int(obj_aux)
                    else:
                        step_log["train/step_bce_loss"] = float(obj_loss.item())
                        step_log["train/step_pos_weight"] = float(step_pos_weight)
                        step_log["train/step_bce_kept_nodes"] = int(obj_aux)
                    wb.log(step_log, step=(ep - 1) * max(1, len(loader)) + steps)

        epoch_loss = float(tot_loss)
        epoch_obj_loss = float(tot_obj_loss)
        epoch_halt_loss = float(tot_halt_loss)
        epoch_rank_loss = float(tot_rank_loss)
        epoch_steps = int(steps)
        epoch_rank_pairs = int(rank_pairs_sum)
        epoch_obj_aux = int(obj_aux_sum)
        epoch_halt_aux = int(halt_aux_sum)
        if is_ddp:
            agg = torch.tensor(
                [
                    epoch_loss,
                    epoch_obj_loss,
                    epoch_halt_loss,
                    epoch_rank_loss,
                    float(epoch_steps),
                    float(epoch_rank_pairs),
                    float(epoch_obj_aux),
                    float(epoch_halt_aux),
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
            epoch_loss = float(agg[0].item())
            epoch_obj_loss = float(agg[1].item())
            epoch_halt_loss = float(agg[2].item())
            epoch_rank_loss = float(agg[3].item())
            epoch_steps = int(round(float(agg[4].item())))
            epoch_rank_pairs = int(round(float(agg[5].item())))
            epoch_obj_aux = int(round(float(agg[6].item())))
            epoch_halt_aux = int(round(float(agg[7].item())))
            dist.barrier()

        mean_loss = epoch_loss / max(1, epoch_steps)
        mean_obj = epoch_obj_loss / max(1, epoch_steps)
        mean_halt = epoch_halt_loss / max(1, epoch_steps)
        mean_rank = epoch_rank_loss / max(1, epoch_steps)
        mean_rank_pairs = float(epoch_rank_pairs) / max(1, epoch_steps)
        mean_obj_aux = float(epoch_obj_aux) / max(1, epoch_steps)
        mean_halt_aux = float(epoch_halt_aux) / max(1, epoch_steps)
        dev_hit = None
        dev_f1 = None
        dev_precision = None
        dev_recall = None

        if is_main and progress_single_line:
            _progress_finish_line(last_progress_chars)

        if is_main:
            os.makedirs(args.out_dir, exist_ok=True)
            save_obj = model.module if hasattr(model, "module") else model
            ckpt = os.path.join(args.out_dir, f"model_ep{ep}.pt")
            payload = {
                "subgraph_reader": True,
                "epoch": int(ep),
                "model_state": save_obj.state_dict(),
                "model_cfg": {
                    "entity_dim": int(train_collate.entity_dim),
                    "relation_dim": int(train_collate.relation_dim),
                    "query_dim": int(train_collate.query_dim),
                    "hidden_size": int(save_obj.hidden_size),
                    "recursion_steps": int(save_obj.recursion_steps),
                    "dropout": float(dropout),
                    "use_direction_embedding": bool(direction_embedding_enabled),
                    "outer_reasoning_enabled": bool(outer_reasoning_enabled),
                    "outer_reasoning_steps": int(outer_reasoning_steps),
                    "gnn_variant": str(gnn_variant),
                    "trm_rel_topk_relations": int(trm_rel_topk_relations),
                    "trm_rel_score_alpha": float(trm_rel_score_alpha),
                    "trm_rel_use_relid_policy": bool(trm_rel_use_relid_policy),
                    "rearev_num_instructions": int(rearev_num_instructions),
                    "rearev_adapt_stages": int(rearev_adapt_stages),
                    "rearev_normalized_gnn": bool(rearev_normalized_gnn),
                    "rearev_latent_reasoning_enabled": bool(rearev_latent_reasoning_enabled),
                    "rearev_latent_residual_alpha": float(rearev_latent_residual_alpha),
                    "rearev_latent_update_mode": str(rearev_latent_update_mode),
                    "rearev_global_gate_enabled": bool(rearev_global_gate_enabled),
                    "rearev_logit_global_fusion_enabled": bool(rearev_logit_global_fusion_enabled),
                    "rearev_dynamic_halting_enabled": bool(rearev_dynamic_halting_enabled),
                    "rearev_dynamic_halting_threshold": float(rearev_dynamic_halting_threshold),
                    "rearev_dynamic_halting_min_steps": int(rearev_dynamic_halting_min_steps),
                    "rearev_trm_style_enabled": bool(rearev_trm_style_enabled),
                    "rearev_trm_tminus1_no_grad": bool(rearev_trm_tminus1_no_grad),
                    "rearev_trm_detach_carry": bool(rearev_trm_detach_carry),
                    "rearev_trm_supervise_all_stages": bool(rearev_trm_supervise_all_stages),
                    "rearev_act_stop_in_train": bool(rearev_act_stop_in_train),
                    "rearev_asymmetric_yz_enabled": bool(rearev_asymmetric_yz_enabled),
                    "rearev_asym_inner_y_ema_enabled": bool(rearev_asym_inner_y_ema_enabled),
                    "rearev_asym_inner_y_ema_alpha": float(rearev_asym_inner_y_ema_alpha),
                },
                "subgraph_cfg": {
                    "hops": int(hops),
                    "max_nodes": int(max_nodes),
                    "max_edges": int(max_edges),
                    "add_reverse_edges": bool(add_reverse_edges),
                    "split_reverse_relations": bool(split_reverse_relations),
                    "pred_threshold": float(threshold),
                    "loss_mode": str(loss_mode),
                    "outer_reasoning_enabled": bool(outer_reasoning_enabled),
                    "outer_reasoning_steps": int(outer_reasoning_steps),
                    "gnn_variant": str(gnn_variant),
                    "trm_rel_topk_relations": int(trm_rel_topk_relations),
                    "trm_rel_score_alpha": float(trm_rel_score_alpha),
                    "trm_rel_use_relid_policy": bool(trm_rel_use_relid_policy),
                    "rearev_num_instructions": int(rearev_num_instructions),
                    "rearev_adapt_stages": int(rearev_adapt_stages),
                    "rearev_normalized_gnn": bool(rearev_normalized_gnn),
                    "rearev_latent_reasoning_enabled": bool(rearev_latent_reasoning_enabled),
                    "rearev_latent_residual_alpha": float(rearev_latent_residual_alpha),
                    "rearev_latent_update_mode": str(rearev_latent_update_mode),
                    "rearev_global_gate_enabled": bool(rearev_global_gate_enabled),
                    "rearev_logit_global_fusion_enabled": bool(rearev_logit_global_fusion_enabled),
                    "rearev_dynamic_halting_enabled": bool(rearev_dynamic_halting_enabled),
                    "rearev_dynamic_halting_threshold": float(rearev_dynamic_halting_threshold),
                    "rearev_dynamic_halting_min_steps": int(rearev_dynamic_halting_min_steps),
                    "rearev_trm_style_enabled": bool(rearev_trm_style_enabled),
                    "rearev_trm_tminus1_no_grad": bool(rearev_trm_tminus1_no_grad),
                    "rearev_trm_detach_carry": bool(rearev_trm_detach_carry),
                    "rearev_trm_supervise_all_stages": bool(rearev_trm_supervise_all_stages),
                    "rearev_act_stop_in_train": bool(rearev_act_stop_in_train),
                    "rearev_asymmetric_yz_enabled": bool(rearev_asymmetric_yz_enabled),
                    "rearev_asym_inner_y_ema_enabled": bool(rearev_asym_inner_y_ema_enabled),
                    "rearev_asym_inner_y_ema_alpha": float(rearev_asym_inner_y_ema_alpha),
                    "rearev_trm_halt_bce_weight": float(rearev_trm_halt_bce_weight),
                    "rearev_trm_ce_weight": float(rearev_trm_ce_weight),
                    "rearev_trm_weight": float(rearev_trm_weight),
                    "deep_supervision_enabled": bool(deep_supervision_enabled),
                    "deep_supervision_weight": float(deep_supervision_weight),
                    "deep_supervision_ce_weight": float(deep_supervision_ce_weight),
                    "deep_supervision_halt_weight": float(deep_supervision_halt_weight),
                    "kl_no_positive_mode": str(kl_no_positive_mode),
                    "kl_supervision_mode": str(kl_supervision_mode),
                    "grad_accum_steps": int(grad_accum_steps),
                },
            }
            torch.save(payload, ckpt)
            print(f"Saved {ckpt}")

            if uses_trm_objective:
                print(
                    f"[Train-Subgraph] ep={ep} loss={mean_loss:.4f} "
                    f"trm_ce_halt={mean_obj:.4f} halt={mean_halt:.4f} "
                    f"trm_ce_rows/step={mean_obj_aux:.1f} halt_steps/step={mean_halt_aux:.1f} "
                    f"opt_steps={optimizer_steps}"
                )
            elif uses_kl_halt_objective:
                print(
                    f"[Train-Subgraph] ep={ep} loss={mean_loss:.4f} "
                    f"kl+halt={mean_obj:.4f} halt={mean_halt:.4f} "
                    f"kl_rows/step={mean_obj_aux:.1f} halt_steps/step={mean_halt_aux:.1f} "
                    f"opt_steps={optimizer_steps}"
                )
            elif uses_kl_trm_objective:
                print(
                    f"[Train-Subgraph] ep={ep} loss={mean_loss:.4f} "
                    f"kl+trm={mean_obj:.4f} halt={mean_halt:.4f} "
                    f"kl_rows/step={mean_obj_aux:.1f} trm_steps/step={mean_halt_aux:.1f} "
                    f"opt_steps={optimizer_steps}"
                )
            elif uses_kl_objective and uses_kl_deep_supervision:
                print(
                    f"[Train-Subgraph] ep={ep} loss={mean_loss:.4f} "
                    f"kl+ds={mean_obj:.4f} halt={mean_halt:.4f} rank={mean_rank:.4f} "
                    f"rank_pairs/step={mean_rank_pairs:.2f} kl_valid_rows/step={mean_obj_aux:.1f} "
                    f"ds_steps/step={mean_halt_aux:.1f} opt_steps={optimizer_steps}"
                )
            elif uses_kl_objective:
                print(
                    f"[Train-Subgraph] ep={ep} loss={mean_loss:.4f} "
                    f"kl={mean_obj:.4f} rank={mean_rank:.4f} "
                    f"rank_pairs/step={mean_rank_pairs:.2f} kl_valid_rows/step={mean_obj_aux:.1f} "
                    f"opt_steps={optimizer_steps}"
                )
            else:
                print(
                    f"[Train-Subgraph] ep={ep} loss={mean_loss:.4f} "
                    f"bce={mean_obj:.4f} rank={mean_rank:.4f} "
                    f"rank_pairs/step={mean_rank_pairs:.2f} bce_kept/step={mean_obj_aux:.1f} "
                    f"opt_steps={optimizer_steps}"
                )
            if wb is not None:
                epoch_log = {
                    "train/epoch_avg_loss": float(mean_loss),
                    "train/epoch_avg_rank_loss": float(mean_rank),
                    "train/epoch_avg_rank_pairs": float(mean_rank_pairs),
                    "train/epoch_optimizer_steps": int(optimizer_steps),
                    "train/epoch": int(ep),
                }
                if uses_trm_objective:
                    epoch_log["train/epoch_avg_trm_ce_halt_loss"] = float(mean_obj)
                    epoch_log["train/epoch_avg_trm_halt_loss"] = float(mean_halt)
                    epoch_log["train/epoch_avg_trm_ce_valid_rows"] = float(mean_obj_aux)
                    epoch_log["train/epoch_avg_trm_halt_valid_steps"] = float(mean_halt_aux)
                elif uses_kl_halt_objective:
                    epoch_log["train/epoch_avg_kl_halt_loss"] = float(mean_obj)
                    epoch_log["train/epoch_avg_halt_loss"] = float(mean_halt)
                    epoch_log["train/epoch_avg_kl_valid_rows"] = float(mean_obj_aux)
                    epoch_log["train/epoch_avg_halt_valid_steps"] = float(mean_halt_aux)
                elif uses_kl_trm_objective:
                    epoch_log["train/epoch_avg_kl_trm_loss"] = float(mean_obj)
                    epoch_log["train/epoch_avg_trm_halt_loss"] = float(mean_halt)
                    epoch_log["train/epoch_avg_kl_valid_rows"] = float(mean_obj_aux)
                    epoch_log["train/epoch_avg_trm_valid_steps"] = float(mean_halt_aux)
                elif uses_kl_objective and uses_kl_deep_supervision:
                    epoch_log["train/epoch_avg_kl_ds_loss"] = float(mean_obj)
                    epoch_log["train/epoch_avg_ds_halt_loss"] = float(mean_halt)
                    epoch_log["train/epoch_avg_kl_valid_rows"] = float(mean_obj_aux)
                    epoch_log["train/epoch_avg_ds_valid_steps"] = float(mean_halt_aux)
                elif uses_kl_objective:
                    epoch_log["train/epoch_avg_kl_loss"] = float(mean_obj)
                    epoch_log["train/epoch_avg_kl_valid_rows"] = float(mean_obj_aux)
                else:
                    epoch_log["train/epoch_avg_bce_loss"] = float(mean_obj)
                    epoch_log["train/epoch_avg_bce_kept_nodes"] = float(mean_obj_aux)
                wb.log(epoch_log, step=ep * max(1, len(loader)))

            eval_every = max(1, int(getattr(args, "eval_every_epochs", 1)))
            eval_start = max(1, int(getattr(args, "eval_start_epoch", 1)))
            should_eval = bool(dev_loader is not None) and ep >= eval_start and ((ep - eval_start) % eval_every == 0)
            if should_eval:
                dev_hit, dev_f1, dev_precision, dev_recall, dev_skip = evaluate_subgraph_reader(
                    model=save_obj,
                    loader=dev_loader,
                    device=device,
                    pred_topk=max(1, int(getattr(args, "eval_pred_topk", 5))),
                    threshold=threshold,
                    is_main=True,
                    desc=f"Dev ep{ep} [Subgraph]",
                )
                print(
                    f"[Dev-Subgraph] Hit@1={dev_hit:.4f} F1={dev_f1:.4f} "
                    f"Precision={dev_precision:.4f} Recall={dev_recall:.4f} Skip={dev_skip}"
                )
                if wb is not None:
                    wb.log(
                        {
                            "dev/hit1": float(dev_hit),
                            "dev/f1": float(dev_f1),
                            "dev/precision": float(dev_precision),
                            "dev/recall": float(dev_recall),
                            "dev/skip": int(dev_skip),
                            "train/epoch": int(ep),
                        },
                        step=ep * max(1, len(loader)),
                )
                if trace_supervision_enabled:
                    debug_supervision_step_trace(
                        model=save_obj,
                        loader=dev_loader,
                        device=device,
                        is_main=True,
                        examples=trace_supervision_examples,
                        dump_jsonl=_format_trace_output_path(trace_supervision_dump_jsonl, ep),
                        plot_png=_format_trace_output_path(trace_supervision_plot_png, ep),
                        log_prefix=f"[Dev-SupTrace][ep{ep}]",
                    )
            elif bool(getattr(args, "dev_json", "")):
                print(f"[Dev-Subgraph] skip eval at ep{ep} (start={eval_start}, every={eval_every})")

        if scheduler is not None:
            if lr_scheduler_mode == "plateau":
                if lr_plateau_metric == "dev_hit1":
                    sched_metric = float(dev_hit) if dev_hit is not None else float(-mean_loss)
                elif lr_plateau_metric == "dev_f1":
                    sched_metric = float(dev_f1) if dev_f1 is not None else float(-mean_loss)
                else:
                    sched_metric = float(mean_loss)

                if is_ddp and lr_plateau_metric in {"dev_hit1", "dev_f1"}:
                    metric_buf = torch.tensor(
                        [float(sched_metric) if is_main else 0.0],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.broadcast(metric_buf, src=0)
                    sched_metric = float(metric_buf.item())
                scheduler.step(float(sched_metric))
            else:
                scheduler.step()

            current_lr = float(opt.param_groups[0]["lr"])
            if is_main:
                print(f"[LR-Subgraph] ep={ep} lr={current_lr:.6g}")
                if wb is not None:
                    wb.log(
                        {
                            "train/lr": float(current_lr),
                            "train/epoch": int(ep),
                        },
                        step=ep * max(1, len(loader)),
                    )

        should_stop = False
        if early_stop_enabled and is_main:
            cur_metric = None
            if early_stop_metric == "train_loss":
                cur_metric = float(mean_loss)
            elif early_stop_metric == "dev_hit1":
                cur_metric = None if dev_hit is None else float(dev_hit)
            elif early_stop_metric == "dev_f1":
                cur_metric = None if dev_f1 is None else float(dev_f1)

            if cur_metric is None:
                if local_ep >= early_stop_min_epochs:
                    print(
                        f"[EarlyStop] metric={early_stop_metric} unavailable at ep={ep} "
                        f"(eval cadence). skip this epoch."
                    )
            else:
                improved = False
                if early_best_metric is None:
                    improved = True
                elif early_stop_metric == "train_loss":
                    improved = (early_best_metric - cur_metric) > early_stop_min_delta
                else:
                    improved = (cur_metric - early_best_metric) > early_stop_min_delta

                if improved:
                    early_best_metric = float(cur_metric)
                    early_bad_epochs = 0
                elif local_ep >= early_stop_min_epochs:
                    early_bad_epochs += 1

                if wb is not None:
                    wb.log(
                        {
                            "train/early_stop_metric_value": float(cur_metric),
                            "train/early_stop_best_metric": float(early_best_metric),
                            "train/early_stop_bad_epochs": int(early_bad_epochs),
                            "train/epoch": int(ep),
                        },
                        step=ep * max(1, len(loader)),
                    )

                if local_ep >= early_stop_min_epochs and early_bad_epochs >= early_stop_patience:
                    should_stop = True
                    print(
                        f"[EarlyStop] triggered at ep={ep} metric={early_stop_metric} "
                        f"best={early_best_metric:.6f} current={cur_metric:.6f} "
                        f"bad_epochs={early_bad_epochs} patience={early_stop_patience}"
                    )

        if is_ddp:
            stop_buf = torch.tensor([1 if should_stop else 0], dtype=torch.int32, device=device)
            dist.broadcast(stop_buf, src=0)
            should_stop = bool(int(stop_buf.item()) == 1)

        if is_ddp:
            dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if should_stop:
            break


def test_subgraph_reader(args):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = world_size > 1
    is_main = rank == 0

    if is_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        if is_ddp:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rel2idx = load_rel_map(args.relations_txt)
    if not getattr(args, "ckpt", ""):
        raise RuntimeError("subgraph reader test requires --ckpt")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")

    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    model_cfg = ckpt_obj.get("model_cfg", {}) if isinstance(ckpt_obj, dict) else {}
    sub_cfg = ckpt_obj.get("subgraph_cfg", {}) if isinstance(ckpt_obj, dict) else {}

    ent_dim = int(model_cfg.get("entity_dim", np.load(args.entity_emb_npy, mmap_mode="r").shape[1]))
    rel_dim = int(model_cfg.get("relation_dim", np.load(args.relation_emb_npy, mmap_mode="r").shape[1]))
    q_dim = int(model_cfg.get("query_dim", np.load(args.query_emb_eval_npy, mmap_mode="r").shape[1]))
    hidden_size = int(model_cfg.get("hidden_size", getattr(args, "hidden_size", 512)))
    recursion_steps = int(model_cfg.get("recursion_steps", getattr(args, "subgraph_recursion_steps", 8)))
    dropout = float(model_cfg.get("dropout", getattr(args, "subgraph_dropout", 0.1)))
    direction_embedding_enabled = _as_bool(
        getattr(args, "subgraph_direction_embedding_enabled", model_cfg.get("use_direction_embedding", False))
    )
    outer_reasoning_enabled = _as_bool(
        getattr(args, "subgraph_outer_reasoning_enabled", model_cfg.get("outer_reasoning_enabled", False))
    )
    outer_reasoning_steps = max(
        1, int(getattr(args, "subgraph_outer_reasoning_steps", model_cfg.get("outer_reasoning_steps", 3)))
    )
    gnn_variant = str(
        getattr(args, "subgraph_gnn_variant", model_cfg.get("gnn_variant", "rearev_bfs"))
    ).strip().lower()
    rearev_num_instructions = max(
        1, int(getattr(args, "subgraph_rearev_num_ins", model_cfg.get("rearev_num_instructions", 3)))
    )
    rearev_adapt_stages = max(
        1, int(getattr(args, "subgraph_rearev_adapt_stages", model_cfg.get("rearev_adapt_stages", 1)))
    )
    rearev_normalized_gnn = _as_bool(
        getattr(args, "subgraph_rearev_normalized_gnn", model_cfg.get("rearev_normalized_gnn", False))
    )
    rearev_latent_reasoning_enabled = _as_bool(
        getattr(
            args,
            "subgraph_rearev_latent_reasoning_enabled",
            model_cfg.get("rearev_latent_reasoning_enabled", False),
        )
    )
    rearev_latent_residual_alpha = max(
        0.0,
        float(
            getattr(
                args,
                "subgraph_rearev_latent_residual_alpha",
                model_cfg.get("rearev_latent_residual_alpha", 0.25),
            )
        ),
    )
    rearev_latent_update_mode = str(
        getattr(
            args,
            "subgraph_rearev_latent_update_mode",
            model_cfg.get("rearev_latent_update_mode", "gru"),
        )
    ).strip().lower()
    rearev_global_gate_enabled = _as_bool(
        getattr(
            args,
            "subgraph_rearev_global_gate_enabled",
            model_cfg.get("rearev_global_gate_enabled", False),
        )
    )
    rearev_logit_global_fusion_enabled = _as_bool(
        getattr(
            args,
            "subgraph_rearev_logit_global_fusion_enabled",
            model_cfg.get("rearev_logit_global_fusion_enabled", False),
        )
    )
    rearev_dynamic_halting_enabled = _as_bool(
        getattr(
            args,
            "subgraph_rearev_dynamic_halting_enabled",
            model_cfg.get("rearev_dynamic_halting_enabled", False),
        )
    )
    rearev_dynamic_halting_threshold = float(
        getattr(
            args,
            "subgraph_rearev_dynamic_halting_threshold",
            model_cfg.get("rearev_dynamic_halting_threshold", 0.9),
        )
    )
    rearev_dynamic_halting_min_steps = max(
        1,
        int(
            getattr(
                args,
                "subgraph_rearev_dynamic_halting_min_steps",
                model_cfg.get("rearev_dynamic_halting_min_steps", 1),
            )
        ),
    )
    rearev_trm_style_enabled = _as_bool(
        getattr(
            args,
            "subgraph_rearev_trm_style_enabled",
            model_cfg.get("rearev_trm_style_enabled", False),
        )
    )
    rearev_trm_tminus1_no_grad = _as_bool(
        getattr(
            args,
            "subgraph_rearev_trm_tminus1_no_grad",
            model_cfg.get("rearev_trm_tminus1_no_grad", True),
        ),
        default=True,
    )
    rearev_trm_detach_carry = _as_bool(
        getattr(
            args,
            "subgraph_rearev_trm_detach_carry",
            model_cfg.get("rearev_trm_detach_carry", True),
        ),
        default=True,
    )
    rearev_trm_supervise_all_stages = _as_bool(
        getattr(
            args,
            "subgraph_rearev_trm_supervise_all_stages",
            model_cfg.get("rearev_trm_supervise_all_stages", False),
        )
    )
    rearev_act_stop_in_train = _as_bool(
        getattr(
            args,
            "subgraph_rearev_act_stop_in_train",
            model_cfg.get("rearev_act_stop_in_train", False),
        )
    )
    rearev_asymmetric_yz_enabled = _as_bool(
        getattr(
            args,
            "subgraph_rearev_asymmetric_yz_enabled",
            model_cfg.get("rearev_asymmetric_yz_enabled", False),
        )
    )
    rearev_asym_inner_y_ema_enabled = _as_bool(
        getattr(
            args,
            "subgraph_rearev_asym_inner_y_ema_enabled",
            model_cfg.get("rearev_asym_inner_y_ema_enabled", False),
        )
    )
    rearev_asym_inner_y_ema_alpha = float(
        min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        args,
                        "subgraph_rearev_asym_inner_y_ema_alpha",
                        model_cfg.get("rearev_asym_inner_y_ema_alpha", 0.0),
                    )
                ),
            ),
        )
    )
    trm_rel_topk_relations = max(
        0,
        int(
            getattr(
                args,
                "subgraph_trm_rel_topk_relations",
                model_cfg.get("trm_rel_topk_relations", 0),
            )
        ),
    )
    trm_rel_score_alpha = float(
        max(
            0.0,
            float(
                getattr(
                    args,
                    "subgraph_trm_rel_score_alpha",
                    model_cfg.get("trm_rel_score_alpha", 1.0),
                )
            ),
        )
    )
    trm_rel_use_relid_policy = _as_bool(
        getattr(
            args,
            "subgraph_trm_rel_use_relid_policy",
            model_cfg.get("trm_rel_use_relid_policy", True),
        ),
        default=True,
    )

    model = RecursiveSubgraphReader(
        entity_dim=ent_dim,
        relation_dim=rel_dim,
        query_dim=q_dim,
        hidden_size=hidden_size,
        recursion_steps=recursion_steps,
        dropout=dropout,
        use_direction_embedding=direction_embedding_enabled,
        outer_reasoning_enabled=outer_reasoning_enabled,
        outer_reasoning_steps=outer_reasoning_steps,
        gnn_variant=gnn_variant,
        rearev_num_instructions=rearev_num_instructions,
        rearev_adapt_stages=rearev_adapt_stages,
        rearev_normalized_gnn=rearev_normalized_gnn,
        rearev_latent_reasoning_enabled=rearev_latent_reasoning_enabled,
        rearev_latent_residual_alpha=rearev_latent_residual_alpha,
        rearev_latent_update_mode=rearev_latent_update_mode,
        rearev_global_gate_enabled=rearev_global_gate_enabled,
        rearev_logit_global_fusion_enabled=rearev_logit_global_fusion_enabled,
        rearev_dynamic_halting_enabled=rearev_dynamic_halting_enabled,
        rearev_dynamic_halting_threshold=rearev_dynamic_halting_threshold,
        rearev_dynamic_halting_min_steps=rearev_dynamic_halting_min_steps,
        rearev_trm_style_enabled=rearev_trm_style_enabled,
        rearev_trm_tminus1_no_grad=rearev_trm_tminus1_no_grad,
        rearev_trm_detach_carry=rearev_trm_detach_carry,
        rearev_trm_supervise_all_stages=rearev_trm_supervise_all_stages,
        rearev_act_stop_in_train=rearev_act_stop_in_train,
        rearev_asymmetric_yz_enabled=rearev_asymmetric_yz_enabled,
        rearev_asym_inner_y_ema_enabled=rearev_asym_inner_y_ema_enabled,
        rearev_asym_inner_y_ema_alpha=rearev_asym_inner_y_ema_alpha,
        trm_rel_topk_relations=trm_rel_topk_relations,
        trm_rel_score_alpha=trm_rel_score_alpha,
        trm_rel_use_relid_policy=trm_rel_use_relid_policy,
    )
    sd = ckpt_obj.get("model_state", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj
    skipped, missing = _safe_load_state_dict(model, sd)
    if skipped > 0:
        print(f"[warn] subgraph reader checkpoint shape-mismatch keys skipped: {skipped}")
    if missing > 0:
        print(f"[warn] subgraph reader checkpoint missing keys after load: {missing}")
    model.to(device)

    hops = int(getattr(args, "subgraph_hops", sub_cfg.get("hops", 3)))
    max_nodes = int(getattr(args, "subgraph_max_nodes", sub_cfg.get("max_nodes", 256)))
    max_edges = int(getattr(args, "subgraph_max_edges", sub_cfg.get("max_edges", 2048)))
    add_reverse_edges = _as_bool(getattr(args, "subgraph_add_reverse_edges", sub_cfg.get("add_reverse_edges", False)))
    split_reverse_relations = _as_bool(
        getattr(args, "subgraph_split_reverse_relations", sub_cfg.get("split_reverse_relations", False))
    )
    threshold = float(getattr(args, "subgraph_pred_threshold", sub_cfg.get("pred_threshold", 0.5)))
    trace_rel_topk_enabled = _as_bool(
        getattr(args, "subgraph_trace_relation_topk_enabled", False)
    )
    trace_rel_topk = max(1, int(getattr(args, "subgraph_trace_relation_topk", 5)))
    trace_rel_log_examples = max(0, int(getattr(args, "subgraph_trace_log_examples", 5)))
    trace_rel_dump_max_examples = max(
        0, int(getattr(args, "subgraph_trace_dump_max_examples", 1000))
    )
    # Backward compatibility for older one-knob option.
    legacy_trace_max = int(getattr(args, "subgraph_trace_max_examples", -1))
    if (
        legacy_trace_max >= 0
        and trace_rel_log_examples == 5
        and trace_rel_dump_max_examples == 1000
    ):
        trace_rel_log_examples = int(legacy_trace_max)
        trace_rel_dump_max_examples = int(legacy_trace_max)
    trace_path_dump_jsonl = str(getattr(args, "subgraph_trace_path_dump_jsonl", "")).strip()
    trace_supervision_enabled = _as_bool(
        getattr(args, "subgraph_trace_supervision_enabled", False)
    )
    trace_supervision_examples = max(
        1, int(getattr(args, "subgraph_trace_supervision_examples", 5))
    )
    trace_supervision_dump_jsonl = str(
        getattr(args, "subgraph_trace_supervision_dump_jsonl", "")
    ).strip()
    trace_supervision_plot_png = str(
        getattr(args, "subgraph_trace_supervision_plot_png", "")
    ).strip()
    relation_ids, relation_labels = _load_relation_text_labels(args.relations_txt)
    entity_labels = _load_entity_text_labels(args.entities_txt)

    eval_ds = SubgraphExampleDataset(args.eval_json)
    eval_limit = int(getattr(args, "eval_limit", -1))
    if eval_limit > 0 and len(eval_ds) > eval_limit:
        eval_ds = Subset(eval_ds, list(range(eval_limit)))
    eval_collate = SubgraphCollator(
        entity_emb_npy=args.entity_emb_npy,
        relation_emb_npy=args.relation_emb_npy,
        query_emb_npy=getattr(args, "query_emb_eval_npy", ""),
        rel2idx=rel2idx,
        hops=hops,
        max_nodes=max_nodes,
        max_edges=max_edges,
        add_reverse_edges=add_reverse_edges,
        split_reverse_relations=split_reverse_relations,
    )
    eval_sampler = (
        DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        if is_ddp
        else None
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(getattr(args, "batch_size", 8)),
        shuffle=False if eval_sampler is not None else False,
        sampler=eval_sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=eval_collate,
        pin_memory=torch.cuda.is_available(),
    )

    if is_main:
        print("[ok] subgraph-reader checkpoint loaded:", args.ckpt)
    out = evaluate_subgraph_reader(
        model=model,
        loader=eval_loader,
        device=device,
        pred_topk=max(1, int(getattr(args, "eval_pred_topk", 5))),
        threshold=threshold,
        is_main=is_main,
        desc="Test-Subgraph",
        return_counts=True,
    )
    hit, f1, precision, recall, skip, n_valid = out

    if is_ddp:
        # Aggregate sums across ranks for exact global metrics.
        agg = torch.tensor(
            [
                float(hit) * float(n_valid),
                float(f1) * float(n_valid),
                float(precision) * float(n_valid),
                float(recall) * float(n_valid),
                float(skip),
                float(n_valid),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        total_valid = int(round(float(agg[5].item())))
        total_skip = int(round(float(agg[4].item())))
        hit = float(agg[0].item() / max(1, total_valid))
        f1 = float(agg[1].item() / max(1, total_valid))
        precision = float(agg[2].item() / max(1, total_valid))
        recall = float(agg[3].item() / max(1, total_valid))
        skip = total_skip

    if is_main:
        print(
            f"[Test-Subgraph] Hit@1={hit:.4f} F1={f1:.4f} "
            f"Precision={precision:.4f} Recall={recall:.4f} Skip={skip}"
        )
        trace_need_log = trace_rel_log_examples > 0
        trace_need_dump = bool(trace_path_dump_jsonl) and trace_rel_dump_max_examples > 0
        if trace_rel_topk_enabled and (trace_need_log or trace_need_dump):
            if is_ddp:
                print(
                    "[Trace-RelTopK] DDP enabled: rank0 shard only. "
                    "Run single-process test for full deterministic traces."
                )
            debug_relation_topk_trace(
                model=model,
                loader=eval_loader,
                device=device,
                relation_ids=relation_ids,
                relation_labels=relation_labels,
                entity_labels=entity_labels,
                topk=trace_rel_topk,
                log_examples=trace_rel_log_examples,
                is_main=is_main,
                path_dump_jsonl=trace_path_dump_jsonl,
                dump_max_examples=trace_rel_dump_max_examples,
            )
        if trace_supervision_enabled:
            if is_ddp:
                print(
                    "[Trace-Supervision] DDP enabled: rank0 shard only. "
                    "Run single-process test for full deterministic traces."
                )
            debug_supervision_step_trace(
                model=model,
                loader=eval_loader,
                device=device,
                is_main=is_main,
                examples=trace_supervision_examples,
                dump_jsonl=trace_supervision_dump_jsonl,
                plot_png=trace_supervision_plot_png,
            )

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
