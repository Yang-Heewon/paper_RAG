#!/usr/bin/env python3
import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = sys.executable
OOM_PAT = re.compile(r"outofmemoryerror|cuda out of memory|cudnn_status_alloc_failed", re.IGNORECASE)
PHASE1_EP_PAT = re.compile(r"Dev ep(\d+) \[Subgraph\]")
PHASE1_HIT_PAT = re.compile(r"\[Dev-Subgraph\] Hit@1=([0-9.]+)")
PHASE2_HIT_PAT = re.compile(r"Hit@1=([0-9.]+)")
PHASE2_F1_PAT = re.compile(r"F1=([0-9.]+)")
TEST_SUBGRAPH_PAT = re.compile(
    r"\[Test-Subgraph\]\s+Hit@1=([0-9.]+)\s+F1=([0-9.]+)\s+Precision=([0-9.]+)\s+Recall=([0-9.]+)\s+Skip=(\d+)"
)
DEFAULT_WANDB_ENTITY = "heewon6205-chung-ang-university"
DEFAULT_WANDB_PROJECT = "paper_final"
BUILTIN_HF_DATASETS = {"cwq", "webqsp"}
SUPPORTED_LAUNCHER_DATASETS = ("cwq", "webqsp")
DEFAULT_MEDICAL_EMBEDDING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _sanitize_tag(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "e5"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(_quote_arg(part) for part in cmd)


def _quote_arg(raw: str) -> str:
    if raw == "":
        return '""'
    if re.search(r"\s", raw):
        return f'"{raw}"'
    return raw


def _detect_gpu_count() -> int:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _default_embed_device() -> str:
    return "cuda" if _detect_gpu_count() > 0 else "cpu"


def _default_embedding_model(dataset: str) -> str:
    if dataset in {"primekgqa", "primekgqa_openkg", "biohopr", "medhopqa"}:
        return DEFAULT_MEDICAL_EMBEDDING_MODEL
    return "intfloat/multilingual-e5-large"


def _default_embed_style(dataset: str) -> str:
    if dataset in {"primekgqa", "primekgqa_openkg", "biohopr", "medhopqa"}:
        return "default"
    return "gnn_rag_gnn_exact"


def _default_embed_backend(dataset: str) -> str:
    if dataset in {"primekgqa", "primekgqa_openkg", "biohopr", "medhopqa"}:
        return "transformers"
    return "sentence_transformers"


def _default_entity_names_json(dataset: str) -> str:
    if dataset == "medhopqa":
        return "data/medhopqa/entity_names.json"
    if dataset in {"cwq", "webqsp"}:
        return "data/data/entities_names.json"
    return ""


def _default_emb_dir(dataset: str, emb_tag: str) -> Path:
    emb_dataset = "primekgqa" if dataset == "primekgqa_openkg" else dataset
    return REPO_ROOT / "trm_agent" / "emb" / f"{emb_dataset}_{emb_tag}"


def _default_nproc() -> int:
    if os.name == "nt":
        return 1
    gpu_count = _detect_gpu_count()
    if gpu_count <= 0:
        return 1
    return min(4, gpu_count)


def _default_fallback_nproc(primary_nproc: int) -> int:
    if primary_nproc <= 1:
        return 0
    return max(1, primary_nproc // 2)


def _default_gpu_list(nproc: int) -> str:
    if nproc <= 0:
        return ""
    return ",".join(str(idx) for idx in range(nproc))


def _default_preprocess_workers() -> int:
    cpu_count = max(1, int(os.cpu_count() or 1))
    conservative = max(1, min(8, cpu_count // 4))
    if os.name == "nt":
        return conservative
    return max(2, conservative)


def _clamp_worker_count(raw: int, *, allow_zero: bool = True, max_workers: int = 8) -> int:
    value = int(raw)
    lower = 0 if allow_zero else 1
    value = max(lower, value)
    return min(max_workers, value)


def _wandb_env() -> Dict[str, str]:
    wandb_root = REPO_ROOT / "wandb"
    return {
        "WANDB_ANONYMOUS": "allow",
        "WANDB_DIR": str(wandb_root),
        "WANDB_CACHE_DIR": str(wandb_root / ".cache"),
        "WANDB_CONFIG_DIR": str(wandb_root / ".config"),
        "WANDB_DATA_DIR": str(wandb_root / ".data"),
        "WANDB_ARTIFACT_DIR": str(wandb_root / "artifacts"),
    }


def _dataset_recipe_defaults(dataset: str) -> Dict[str, str]:
    if dataset == "webqsp":
        return {
            "phase1_batch_size": "1",
            "phase1_lr": "1e-4",
            "phase1_grad_accum": "4",
            "phase1_kl_no_positive_mode": "skip",
            "phase1_max_nodes": "4096",
            "phase1_max_edges": "16384",
            "phase2_batch_size": "1",
            "phase2_lr": "5e-5",
            "phase2_grad_accum": "8",
            "phase2_max_nodes": "1536",
            "phase2_max_edges": "6144",
            "phase2_best_metric": "dev_hit1",
        }
    if dataset == "biohopr":
        return {
            "phase1_batch_size": "4",
            "phase1_lr": "1e-4",
            "phase1_grad_accum": "1",
            "phase1_kl_no_positive_mode": "uniform",
            "phase1_max_nodes": "512",
            "phase1_max_edges": "2048",
            "phase2_batch_size": "4",
            "phase2_lr": "5e-5",
            "phase2_grad_accum": "1",
            "phase2_max_nodes": "512",
            "phase2_max_edges": "2048",
            "phase2_best_metric": "dev_hit1",
        }
    if dataset == "medhopqa":
        return {
            "phase1_batch_size": "4",
            "phase1_lr": "1e-4",
            "phase1_grad_accum": "1",
            "phase1_kl_no_positive_mode": "uniform",
            "phase1_max_nodes": "1024",
            "phase1_max_edges": "4096",
            "phase2_batch_size": "4",
            "phase2_lr": "5e-5",
            "phase2_grad_accum": "1",
            "phase2_max_nodes": "1024",
            "phase2_max_edges": "4096",
            "phase2_best_metric": "dev_hit1",
        }
    if dataset in {"primekgqa", "primekgqa_openkg"}:
        return {
            "phase1_batch_size": "4",
            "phase1_lr": "1e-4",
            "phase1_grad_accum": "1",
            "phase1_kl_no_positive_mode": "uniform",
            "phase1_max_nodes": "2048",
            "phase1_max_edges": "8192",
            "phase2_batch_size": "4",
            "phase2_lr": "5e-5",
            "phase2_grad_accum": "1",
            "phase2_max_nodes": "1536",
            "phase2_max_edges": "6144",
            "phase2_best_metric": "dev_hit1",
        }
    return {
        "phase1_batch_size": "1",
        "phase1_lr": "2e-4",
        "phase1_grad_accum": "8",
        "phase1_kl_no_positive_mode": "uniform",
        "phase1_max_nodes": "2048",
        "phase1_max_edges": "8192",
        "phase2_batch_size": "1",
        "phase2_lr": "5e-5",
        "phase2_grad_accum": "4",
        "phase2_max_nodes": "2048",
        "phase2_max_edges": "8192",
        "phase2_best_metric": "dev_hit1",
    }


def _variant_recipe_defaults(variant: str) -> Dict[str, str]:
    if variant == "dplus":
        return {
            "subgraph_gnn_variant": "rearev_dplus",
            "subgraph_rearev_latent_update_mode": "attn",
        }
    return {
        "subgraph_gnn_variant": "rearev_bfs",
        "subgraph_rearev_latent_update_mode": "gru",
    }


def _variant_display_name(variant: str) -> str:
    return "D+" if str(variant).strip().lower() == "dplus" else "D"


def _variant_path_tag(variant: str) -> str:
    return "Dplus" if str(variant).strip().lower() == "dplus" else "D"


def _recipe_display_name(recursion_steps: int, instructions: int) -> str:
    return f"r{int(recursion_steps)}i{int(instructions)}"


def _phase_batch_and_accum(
    *,
    args: argparse.Namespace,
    recipe: Dict[str, str],
    phase: str,
) -> Tuple[str, str]:
    batch_key = f"{phase}_batch_size"
    accum_key = f"{phase}_grad_accum"
    arg_batch = getattr(args, f"{phase}_batch_size", 0)
    arg_accum = getattr(args, f"{phase}_grad_accum", 0)
    batch_size = int(arg_batch) if int(arg_batch) > 0 else int(recipe[batch_key])
    grad_accum = int(arg_accum) if int(arg_accum) > 0 else int(recipe[accum_key])
    return str(max(1, batch_size)), str(max(1, grad_accum))


def _read_text_safe(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _looks_like_complete_json_array(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            pos = handle.tell() - 1
            while pos >= 0:
                handle.seek(pos)
                ch = handle.read(1)
                if ch and not ch.isspace():
                    return ch == b"]"
                pos -= 1
    except Exception:
        return False
    return False


def _count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _salvage_json_array_to_jsonl(source: Path, target: Path, *, chunk_chars: int = 1 << 20) -> int:
    if not source.exists():
        return 0
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime and target.stat().st_size > 0:
        return _count_nonempty_lines(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    decoder = json.JSONDecoder()
    record_count = 0
    buffer = ""
    started = False

    with source.open("r", encoding="utf-8", errors="ignore") as src, tmp_path.open("w", encoding="utf-8") as out:
        eof = False
        while True:
            chunk = src.read(chunk_chars)
            if chunk:
                buffer += chunk
            else:
                eof = True

            idx = 0
            while True:
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1

                if not started:
                    if idx >= len(buffer):
                        break
                    if buffer[idx] != "[":
                        raise ValueError(f"{source} is not a JSON array")
                    started = True
                    idx += 1
                    continue

                while idx < len(buffer) and (buffer[idx].isspace() or buffer[idx] == ","):
                    idx += 1
                if idx >= len(buffer):
                    break
                if buffer[idx] == "]":
                    idx += 1
                    buffer = buffer[idx:]
                    idx = 0
                    eof = True
                    break

                try:
                    obj, end_idx = decoder.raw_decode(buffer, idx)
                except json.JSONDecodeError:
                    if eof:
                        break
                    break

                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                record_count += 1
                idx = end_idx

            buffer = buffer[idx:]
            if eof:
                break

    if record_count <= 0:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        return 0

    tmp_path.replace(target)
    return record_count


def _log_has_oom_signature(log_path: Path) -> bool:
    return bool(OOM_PAT.search(_read_text_safe(log_path)))


def _cleanup_after_oom(sleep_sec: float) -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    wait_sec = max(0.0, float(sleep_sec))
    if wait_sec > 0.0:
        print(f"[oom] cleanup complete; sleep {wait_sec:.1f}s before retry")
        time.sleep(wait_sec)


def _scaled_down_int(current: int, factor: float, minimum: int, step: int) -> int:
    cur = max(0, int(current))
    min_allowed = max(0, int(minimum))
    granularity = max(1, int(step))
    if cur <= min_allowed:
        return cur
    next_val = int(cur * float(factor))
    next_val = max(min_allowed, next_val)
    next_val = (next_val // granularity) * granularity
    next_val = max(min_allowed, next_val)
    if next_val >= cur:
        next_val = max(min_allowed, cur - granularity)
    return max(min_allowed, next_val)


def _rescale_grad_accum(overrides: Dict[str, str], current_nproc: int, target_nproc: int) -> Dict[str, str]:
    updated = dict(overrides)
    current_accum = max(1, int(updated.get("subgraph_grad_accum_steps", "1")))
    scaled_accum = max(1, current_accum * max(1, int(current_nproc)) // max(1, int(target_nproc)))
    updated["subgraph_grad_accum_steps"] = str(scaled_accum)
    return updated


def _reduce_overrides_for_oom(
    overrides: Dict[str, str],
    *,
    reduce_factor: float,
    min_nodes: int,
    min_edges: int,
) -> Tuple[Optional[Dict[str, str]], List[str]]:
    updated = dict(overrides)
    changes: List[str] = []

    batch_size = max(1, int(updated.get("batch_size", "1")))
    if batch_size > 1:
        grad_accum = max(1, int(updated.get("subgraph_grad_accum_steps", "1")))
        next_batch = max(1, batch_size // 2)
        if next_batch < batch_size:
            scaled_accum = max(1, (grad_accum * batch_size + next_batch - 1) // next_batch)
            updated["batch_size"] = str(next_batch)
            updated["subgraph_grad_accum_steps"] = str(scaled_accum)
            changes.append(f"batch_size {batch_size}->{next_batch}")
            changes.append(f"grad_accum {grad_accum}->{scaled_accum}")
            return updated, changes

    if "subgraph_max_edges" in updated:
        cur_edges = max(0, int(updated["subgraph_max_edges"]))
        next_edges = _scaled_down_int(cur_edges, reduce_factor, min_edges, step=256)
        if next_edges < cur_edges:
            updated["subgraph_max_edges"] = str(next_edges)
            changes.append(f"max_edges {cur_edges}->{next_edges}")

    if "subgraph_max_nodes" in updated:
        cur_nodes = max(0, int(updated["subgraph_max_nodes"]))
        next_nodes = _scaled_down_int(cur_nodes, reduce_factor, min_nodes, step=128)
        if next_nodes < cur_nodes:
            updated["subgraph_max_nodes"] = str(next_nodes)
            changes.append(f"max_nodes {cur_nodes}->{next_nodes}")

    if changes:
        return updated, changes
    return None, []


def _run_with_tee(
    cmd: Sequence[str],
    *,
    env_overrides: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items() if v is not None})

    print(f"[run] {_format_cmd(cmd)}")
    if env_overrides:
        printable_env = " ".join(f"{k}={v}" for k, v in sorted(env_overrides.items()) if v is not None)
        if printable_env:
            print(f"[env] {printable_env}")

    if dry_run:
        return 0

    log_handle = None
    proc = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("wb")
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert proc.stdout is not None
        stdout_buffer = getattr(sys.stdout, "buffer", None)
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            if stdout_buffer is not None:
                stdout_buffer.write(chunk)
                stdout_buffer.flush()
            else:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            if log_handle is not None:
                log_handle.write(chunk)
                log_handle.flush()
        return proc.wait()
    finally:
        if proc is not None and proc.stdout is not None:
            proc.stdout.close()
        if log_handle is not None:
            log_handle.close()


def _run_simple(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    print(f"[run] {_format_cmd(cmd)}")
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=str(REPO_ROOT), check=True)


def _run_train_with_oom_guard(
    *,
    phase_name: str,
    dataset: str,
    model_impl: str,
    ckpt: str,
    base_overrides: Dict[str, str],
    primary_nproc: int,
    primary_master_port: int,
    primary_gpus: str,
    fallback_nproc: int,
    fallback_master_port: int,
    fallback_gpus: str,
    run_root: Path,
    dry_run: bool,
    oom_retry_sleep_sec: float,
    oom_max_retries: int,
    oom_reduce_factor: float,
    oom_min_nodes: int,
    oom_min_edges: int,
) -> Tuple[List[Path], Dict[str, str], Path]:
    current_overrides = dict(base_overrides)
    current_nproc = max(1, int(primary_nproc))
    current_master_port = int(primary_master_port)
    current_gpus = str(primary_gpus or "")
    fallback_used = False
    oom_retry_idx = 0
    log_paths: List[Path] = []

    while True:
        attempt_label = "primary"
        if fallback_used and oom_retry_idx <= 0:
            attempt_label = "fallback"
        elif oom_retry_idx > 0:
            attempt_label = f"oom_retry{oom_retry_idx}"

        log_path = run_root / f"{phase_name}_{attempt_label}.log"
        cmd = _build_train_command(
            dataset=dataset,
            model_impl=model_impl,
            ckpt=ckpt,
            overrides=current_overrides,
            nproc=current_nproc,
            master_port=current_master_port,
        )
        env = _wandb_env()
        if current_gpus:
            env["CUDA_VISIBLE_DEVICES"] = current_gpus

        rc = _run_with_tee(
            cmd,
            env_overrides=env,
            log_path=log_path,
            dry_run=dry_run,
        )
        log_paths.append(log_path)
        if rc == 0:
            return log_paths, current_overrides, log_path
        if not _log_has_oom_signature(log_path):
            raise RuntimeError(f"{phase_name.capitalize()} training failed without an OOM signature.")

        print(f"[{phase_name}] detected OOM; preparing retry")
        _cleanup_after_oom(oom_retry_sleep_sec)

        if (
            not fallback_used
            and int(fallback_nproc) > 0
            and int(fallback_nproc) != int(current_nproc)
        ):
            current_overrides = _rescale_grad_accum(
                current_overrides,
                current_nproc=current_nproc,
                target_nproc=int(fallback_nproc),
            )
            current_nproc = int(fallback_nproc)
            current_master_port = int(fallback_master_port)
            current_gpus = str(fallback_gpus or "")
            fallback_used = True
            print(
                f"[{phase_name}] retry with nproc={current_nproc} "
                f"grad_accum={current_overrides.get('subgraph_grad_accum_steps', '1')}"
            )
            continue

        if oom_retry_idx >= max(0, int(oom_max_retries)):
            raise RuntimeError(f"{phase_name.capitalize()} training failed after exhausting OOM retries.")

        reduced_overrides, changes = _reduce_overrides_for_oom(
            current_overrides,
            reduce_factor=oom_reduce_factor,
            min_nodes=oom_min_nodes,
            min_edges=oom_min_edges,
        )
        if reduced_overrides is None:
            raise RuntimeError(
                f"{phase_name.capitalize()} training hit OOM and no smaller safe configuration remains."
            )
        oom_retry_idx += 1
        current_overrides = reduced_overrides
        current_master_port = int(current_master_port) + 1
        print(f"[{phase_name}] retry #{oom_retry_idx} with " + ", ".join(changes))


def _build_run_command(
    *,
    dataset: str,
    stage: str,
    model_impl: str,
    ckpt: str = "",
    overrides: Optional[Dict[str, str]] = None,
) -> List[str]:
    cmd = [
        PYTHON_BIN,
        "-m",
        "trm_agent.run",
        "--dataset",
        dataset,
        "--stage",
        stage,
        "--model_impl",
        model_impl,
    ]
    if ckpt:
        cmd.extend(["--ckpt", ckpt])
    if overrides:
        cmd.append("--override")
        for key, value in overrides.items():
            cmd.append(f"{key}={value}")
    return cmd


def _build_train_command(
    *,
    dataset: str,
    model_impl: str,
    ckpt: str,
    overrides: Dict[str, str],
    nproc: int,
    master_port: int,
) -> List[str]:
    base_cmd = _build_run_command(
        dataset=dataset,
        stage="train",
        model_impl=model_impl,
        ckpt=ckpt,
        overrides=overrides,
    )
    if nproc <= 1:
        return base_cmd
    return [
        PYTHON_BIN,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--master_port={master_port}",
        *base_cmd[1:],
    ]


def _build_test_command(
    *,
    dataset: str,
    model_impl: str,
    ckpt: str,
    overrides: Dict[str, str],
) -> List[str]:
    return _build_run_command(
        dataset=dataset,
        stage="test",
        model_impl=model_impl,
        ckpt=ckpt,
        overrides=overrides,
    )


def _phase1_best_ckpt(ckpt_dir: Path, log_paths: Sequence[Path]) -> Path:
    best: Optional[Tuple[float, int, Path]] = None
    for log_path in log_paths:
        if not log_path.exists():
            continue
        cur_ep: Optional[int] = None
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                ep_match = PHASE1_EP_PAT.search(line)
                if ep_match:
                    cur_ep = int(ep_match.group(1))
                    continue
                hit_match = PHASE1_HIT_PAT.search(line)
                if hit_match is None or cur_ep is None:
                    continue
                ckpt_path = ckpt_dir / f"model_ep{cur_ep}.pt"
                if not ckpt_path.exists():
                    continue
                candidate = (float(hit_match.group(1)), cur_ep, ckpt_path)
                if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] > best[1]):
                    best = candidate
    if best is not None:
        return best[2]
    ckpts = sorted(
        ckpt_dir.glob("model_ep*.pt"),
        key=lambda path: int(re.search(r"model_ep(\d+)\.pt$", path.name).group(1)),
    )
    if not ckpts:
        raise FileNotFoundError(f"No phase1 checkpoint found in {ckpt_dir}")
    return ckpts[-1]


def _phase2_best_ckpts(ckpt_dir: Path, log_path: Path) -> Dict[str, Path]:
    best_hit: Optional[Tuple[float, int, Path]] = None
    best_f1: Optional[Tuple[float, int, Path]] = None
    cur_ep: Optional[int] = None
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                ep_match = PHASE1_EP_PAT.search(line)
                if ep_match:
                    cur_ep = int(ep_match.group(1))
                    continue
                if "[Dev-Subgraph]" not in line or cur_ep is None:
                    continue
                hit_match = PHASE2_HIT_PAT.search(line)
                f1_match = PHASE2_F1_PAT.search(line)
                if hit_match is None or f1_match is None:
                    continue
                ckpt_path = ckpt_dir / f"model_ep{cur_ep}.pt"
                if not ckpt_path.exists():
                    continue
                hit_candidate = (float(hit_match.group(1)), cur_ep, ckpt_path)
                f1_candidate = (float(f1_match.group(1)), cur_ep, ckpt_path)
                if best_hit is None or hit_candidate[0] > best_hit[0] or (
                    hit_candidate[0] == best_hit[0] and hit_candidate[1] > best_hit[1]
                ):
                    best_hit = hit_candidate
                if best_f1 is None or f1_candidate[0] > best_f1[0] or (
                    f1_candidate[0] == best_f1[0] and f1_candidate[1] > best_f1[1]
                ):
                    best_f1 = f1_candidate
    if best_hit is None or best_f1 is None:
        ckpts = sorted(
            ckpt_dir.glob("model_ep*.pt"),
            key=lambda path: int(re.search(r"model_ep(\d+)\.pt$", path.name).group(1)),
        )
        if not ckpts:
            raise FileNotFoundError(f"No phase2 checkpoint found in {ckpt_dir}")
        fallback = ckpts[-1]
        if best_hit is None:
            best_hit = (float("-inf"), -1, fallback)
        if best_f1 is None:
            best_f1 = (float("-inf"), -1, fallback)

    result = {
        "dev_hit1": best_hit[2],
        "dev_f1": best_f1[2],
    }
    for metric, ckpt_path in result.items():
        target = ckpt_dir / f"best_{metric}.txt"
        target.write_text(str(ckpt_path), encoding="utf-8")
    return result


def _latest_phase2_dir(dataset: str, variant: str) -> Path:
    ckpt_root = REPO_ROOT / "trm_agent" / "ckpt"
    pattern = f"{dataset}_*_rearev_{_variant_path_tag(variant)}_phase2_*"
    candidates = [path for path in ckpt_root.glob(pattern) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No phase2 checkpoint directory found for dataset={dataset}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _infer_resume_epoch(ckpt_path: Path) -> int:
    match = re.search(r"model_ep(\d+)\.pt$", ckpt_path.name)
    if match is None:
        return 0
    return int(match.group(1))


def _phase_graph_budget(
    *,
    args: argparse.Namespace,
    recipe: Dict[str, str],
    phase: str,
) -> Tuple[str, str]:
    phase_key = phase.lower()
    cli_nodes = int(getattr(args, f"{phase_key}_max_nodes", 0) or 0)
    cli_edges = int(getattr(args, f"{phase_key}_max_edges", 0) or 0)
    nodes = str(cli_nodes) if cli_nodes > 0 else recipe[f"{phase_key}_max_nodes"]
    edges = str(cli_edges) if cli_edges > 0 else recipe[f"{phase_key}_max_edges"]
    return nodes, edges


def _base_train_overrides(
    *,
    args: argparse.Namespace,
    dataset: str,
    emb_dir: Path,
    ckpt_dir: Path,
) -> Dict[str, str]:
    variant_defaults = _variant_recipe_defaults(args.variant)
    variant_name = _variant_display_name(args.variant)
    recipe_name = _recipe_display_name(args.recursion_steps, args.instructions)
    return {
        "emb_tag": args.emb_tag,
        "emb_dir": str(emb_dir),
        "ckpt_dir": str(ckpt_dir),
        "seed": str(args.seed),
        "deterministic": _bool_str(args.deterministic),
        "eval_every_epochs": "1",
        "eval_start_epoch": "1",
        "eval_limit": "-1",
        "wandb_mode": args.wandb_mode,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_group": args.wandb_group or f"{dataset}-{recipe_name}-{variant_name}-{args.run_tag}",
        "subgraph_reader_enabled": "true",
        "subgraph_add_reverse_edges": "true",
        "subgraph_split_reverse_relations": "true",
        "subgraph_direction_embedding_enabled": "true",
        "subgraph_rearev_num_ins": str(args.instructions),
        "subgraph_rearev_adapt_stages": str(args.adapt_stages),
        "subgraph_recursion_steps": str(args.recursion_steps),
        "subgraph_gnn_variant": variant_defaults["subgraph_gnn_variant"],
        "subgraph_trm_rel_topk_relations": "0",
        "subgraph_trm_rel_score_alpha": "1.0",
        "subgraph_trm_rel_use_relid_policy": "true",
        "subgraph_rearev_normalized_gnn": "false",
        "subgraph_rearev_latent_reasoning_enabled": _bool_str(args.latent_reasoning_enabled),
        "subgraph_rearev_latent_residual_alpha": "0.25",
        "subgraph_rearev_latent_update_mode": variant_defaults["subgraph_rearev_latent_update_mode"],
        "subgraph_rearev_global_gate_enabled": "true",
        "subgraph_rearev_logit_global_fusion_enabled": "true",
        "subgraph_rearev_dynamic_halting_enabled": _bool_str(args.dynamic_halting_enabled),
        "subgraph_rearev_dynamic_halting_min_steps": "3",
        "subgraph_rearev_dynamic_halting_threshold": "0.95",
        "subgraph_ddp_find_unused_parameters": "false",
    }


def _phase1_overrides(
    *,
    args: argparse.Namespace,
    dataset: str,
    emb_dir: Path,
    ckpt_dir: Path,
) -> Dict[str, str]:
    recipe = _dataset_recipe_defaults(dataset)
    variant_name = _variant_display_name(args.variant)
    recipe_name = _recipe_display_name(args.recursion_steps, args.instructions)
    batch_size, grad_accum = _phase_batch_and_accum(args=args, recipe=recipe, phase="phase1")
    max_nodes, max_edges = _phase_graph_budget(args=args, recipe=recipe, phase="phase1")
    out = _base_train_overrides(args=args, dataset=dataset, emb_dir=emb_dir, ckpt_dir=ckpt_dir)
    out.update(
        {
            "epochs": str(args.phase1_epochs),
            "batch_size": batch_size,
            "lr": recipe["phase1_lr"],
            "wandb_run_name": f"{dataset}-{recipe_name}-phase1-{variant_name}-{args.run_tag}",
            "subgraph_loss_mode": "rearev_kl",
            "subgraph_kl_no_positive_mode": recipe["phase1_kl_no_positive_mode"],
            "subgraph_kl_supervision_mode": "final",
            "subgraph_max_nodes": max_nodes,
            "subgraph_max_edges": max_edges,
            "subgraph_grad_accum_steps": grad_accum,
            "subgraph_ranking_enabled": "false",
            "subgraph_lr_scheduler": "cosine",
            "subgraph_lr_min": "1e-6",
            "subgraph_lr_step_size": "5",
            "subgraph_lr_gamma": "0.5",
            "subgraph_lr_plateau_factor": "0.5",
            "subgraph_lr_plateau_patience": "2",
            "subgraph_lr_plateau_threshold": "1e-4",
            "subgraph_lr_plateau_metric": "dev_hit1",
            "subgraph_early_stop_enabled": _bool_str(args.phase1_early_stop_enabled),
            "subgraph_early_stop_metric": args.phase1_early_stop_metric,
            "subgraph_early_stop_patience": str(args.phase1_early_stop_patience),
            "subgraph_early_stop_min_delta": str(args.phase1_early_stop_min_delta),
            "subgraph_early_stop_min_epochs": str(args.phase1_early_stop_min_epochs),
            "subgraph_deep_supervision_enabled": "false",
            "subgraph_deep_supervision_weight": "0.0",
            "subgraph_deep_supervision_ce_weight": "1.0",
            "subgraph_deep_supervision_halt_weight": "1.0",
            "subgraph_rearev_trm_tminus1_no_grad": "true",
            "subgraph_rearev_trm_detach_carry": "true",
            "subgraph_rearev_trm_supervise_all_stages": "false",
            "subgraph_rearev_act_stop_in_train": "false",
            "subgraph_rearev_asymmetric_yz_enabled": "false",
        }
    )
    return out


def _phase2_overrides(
    *,
    args: argparse.Namespace,
    dataset: str,
    emb_dir: Path,
    ckpt_dir: Path,
    phase1_ckpt: Path,
) -> Dict[str, str]:
    recipe = _dataset_recipe_defaults(dataset)
    variant_name = _variant_display_name(args.variant)
    recipe_name = _recipe_display_name(args.recursion_steps, args.instructions)
    batch_size, grad_accum = _phase_batch_and_accum(args=args, recipe=recipe, phase="phase2")
    max_nodes, max_edges = _phase_graph_budget(args=args, recipe=recipe, phase="phase2")
    out = _base_train_overrides(args=args, dataset=dataset, emb_dir=emb_dir, ckpt_dir=ckpt_dir)
    out.update(
        {
            "epochs": str(args.phase2_epochs),
            "batch_size": batch_size,
            "lr": recipe["phase2_lr"],
            "wandb_run_name": f"{dataset}-{recipe_name}-phase2-{variant_name}-{args.run_tag}",
            "auto_test_after_train": "false",
            "subgraph_loss_mode": "bce",
            "subgraph_max_nodes": max_nodes,
            "subgraph_max_edges": max_edges,
            "subgraph_grad_accum_steps": grad_accum,
            "subgraph_resume_epoch": str(_infer_resume_epoch(phase1_ckpt)),
            "subgraph_pos_weight_mode": "auto",
            "subgraph_pos_weight_max": "256",
            "subgraph_ranking_enabled": "true",
            "subgraph_ranking_weight": "0.15",
            "subgraph_ranking_margin": "0.2",
            "subgraph_hard_negative_topk": "32",
            "subgraph_bce_hard_negative_enabled": "true",
            "subgraph_bce_hard_negative_topk": "64",
            "subgraph_lr_scheduler": "plateau",
            "subgraph_lr_min": "1e-6",
            "subgraph_lr_plateau_factor": "0.7",
            "subgraph_lr_plateau_patience": "1",
            "subgraph_lr_plateau_threshold": "1e-4",
            "subgraph_lr_plateau_metric": "dev_hit1",
            "subgraph_early_stop_enabled": _bool_str(args.phase2_early_stop_enabled),
            "subgraph_early_stop_metric": args.phase2_early_stop_metric,
            "subgraph_early_stop_patience": str(args.phase2_early_stop_patience),
            "subgraph_early_stop_min_delta": str(args.phase2_early_stop_min_delta),
            "subgraph_early_stop_min_epochs": str(args.phase2_early_stop_min_epochs),
            "subgraph_rearev_asymmetric_yz_enabled": "false",
        }
    )
    return out


def _test_overrides(
    *,
    args: argparse.Namespace,
    dataset: str,
    emb_dir: Path,
    metric: str,
) -> Dict[str, str]:
    recipe = _dataset_recipe_defaults(dataset)
    variant_defaults = _variant_recipe_defaults(args.variant)
    return {
        "emb_tag": args.emb_tag,
        "emb_dir": str(emb_dir),
        "batch_size": "6",
        "eval_limit": "-1",
        "debug_eval_n": "5",
        "eval_no_cycle": "true",
        "eval_max_steps": "4",
        "eval_max_neighbors": "256",
        "eval_prune_keep": "64",
        "eval_beam": "8",
        "eval_start_topk": "5",
        "eval_pred_topk": "5",
        "eval_use_halt": "true",
        "eval_min_hops_before_stop": "2",
        "subgraph_reader_enabled": "true",
        "subgraph_hops": "3",
        "subgraph_max_nodes": recipe["phase2_max_nodes"],
        "subgraph_max_edges": recipe["phase2_max_edges"],
        "subgraph_recursion_steps": str(args.recursion_steps),
        "subgraph_pred_threshold": "0.5",
        "subgraph_split_reverse_relations": "true",
        "subgraph_direction_embedding_enabled": "true",
        "subgraph_gnn_variant": variant_defaults["subgraph_gnn_variant"],
        "subgraph_rearev_num_ins": str(args.instructions),
        "subgraph_rearev_adapt_stages": str(args.adapt_stages),
        "subgraph_rearev_latent_reasoning_enabled": _bool_str(args.latent_reasoning_enabled),
        "subgraph_rearev_latent_residual_alpha": "0.25",
        "subgraph_rearev_latent_update_mode": variant_defaults["subgraph_rearev_latent_update_mode"],
        "subgraph_rearev_global_gate_enabled": "true",
        "subgraph_rearev_logit_global_fusion_enabled": "true",
        "subgraph_rearev_dynamic_halting_enabled": _bool_str(args.dynamic_halting_enabled),
        "subgraph_rearev_dynamic_halting_min_steps": "3",
        "subgraph_rearev_dynamic_halting_threshold": "0.95",
    }


def _ensure_data_ready(dataset: str) -> List[Path]:
    if dataset == "cwq":
        return [
            REPO_ROOT / "data" / "CWQ" / "train_split.jsonl",
            REPO_ROOT / "data" / "CWQ" / "dev_split.jsonl",
            REPO_ROOT / "data" / "CWQ" / "test_split.jsonl",
            REPO_ROOT / "data" / "CWQ" / "entities.txt",
            REPO_ROOT / "data" / "CWQ" / "relations.txt",
        ]
    if dataset == "webqsp":
        return [
            REPO_ROOT / "data" / "webqsp" / "train.json",
            REPO_ROOT / "data" / "webqsp" / "dev.json",
            REPO_ROOT / "data" / "webqsp" / "test.json",
            REPO_ROOT / "data" / "webqsp" / "entities.txt",
            REPO_ROOT / "data" / "webqsp" / "relations.txt",
        ]
    dataset_root = REPO_ROOT / "data" / dataset
    return [
        dataset_root / "train.jsonl",
        dataset_root / "dev.jsonl",
        dataset_root / "test.jsonl",
        dataset_root / "entities.txt",
        dataset_root / "relations.txt",
    ]


def _ensure_processed_ready(dataset: str) -> List[Path]:
    root = REPO_ROOT / "trm_agent" / "processed" / dataset
    return [root / "train.jsonl", root / "dev.jsonl", root / "test.jsonl"]


def _ensure_emb_ready(emb_dir: Path) -> List[Path]:
    return [
        emb_dir / "entity_embeddings.npy",
        emb_dir / "relation_embeddings.npy",
        emb_dir / "query_train.npy",
        emb_dir / "query_dev.npy",
        emb_dir / "query_test.npy",
    ]


def _check_paths(paths: Iterable[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} files:\n" + "\n".join(missing))


def _prepare_common_overrides(args: argparse.Namespace) -> Dict[str, str]:
    common_prepare = {
        "emb_tag": args.emb_tag,
        "embed_device": args.embed_device,
        "embed_gpus": args.embed_gpus,
        "embed_batch_size": str(args.embed_batch_size),
        "embed_max_length": str(args.embed_max_length),
        "embed_style": args.embed_style,
        "embed_backend": args.embed_backend,
        "max_steps": str(args.max_steps),
        "max_paths": str(args.max_paths),
        "mine_max_neighbors": str(args.mine_max_neighbors),
        "preprocess_workers": str(args.preprocess_workers),
        "train_path_policy": args.train_path_policy,
        "train_shortest_k": str(args.train_shortest_k),
        "embed_auto_batch": _bool_str(args.embed_auto_batch),
        "embed_auto_batch_min": str(args.embed_auto_batch_min),
        "embed_auto_batch_max": str(args.embed_auto_batch_max),
        "embed_auto_batch_vram_frac": str(args.embed_auto_batch_vram_frac),
    }
    entity_names_path = Path(args.entity_names_json).resolve() if args.entity_names_json else None
    if entity_names_path and entity_names_path.exists():
        common_prepare["entity_names_json"] = args.entity_names_json
    return common_prepare


def _downloaded_medical_converter_cmd(args: argparse.Namespace, dataset: str) -> Optional[List[str]]:
    if dataset == "primekgqa":
        raw_root = REPO_ROOT / "data" / "primekgqa_raw"
        train_in = raw_root / "train_v2_call_bioLLM.json"
        if str(args.primekgqa_train_override).strip():
            train_in = Path(args.primekgqa_train_override).expanduser().resolve()
        dev_in = raw_root / "val_v2_call_bioLLM.json"
        test_in = raw_root / "test_v2_call_bioLLM.json"
        if all(path.exists() for path in (dev_in, test_in)):
            effective_train_in = train_in
            if not _looks_like_complete_json_array(train_in):
                salvaged_train = raw_root / "train_v2_call_bioLLM.salvaged.jsonl"
                salvaged_count = _salvage_json_array_to_jsonl(train_in, salvaged_train)
                if salvaged_count > 0:
                    print(
                        f"[prepare] primekgqa truncated train recovered: {salvaged_count} complete records -> {salvaged_train}"
                    )
                    effective_train_in = salvaged_train
                else:
                    print(
                        "[warn] primekgqa train_v2_call_bioLLM.json looks truncated and could not be recovered; "
                        "falling back to val_v2_call_bioLLM.json for train/dev and test_v2_call_bioLLM.json for test"
                    )
                    effective_train_in = dev_in
            return [
                PYTHON_BIN,
                "scripts/prepare_medical_kgqa.py",
                "--dataset",
                dataset,
                "--train-in",
                str(effective_train_in),
                "--dev-in",
                str(dev_in),
                "--test-in",
                str(test_in),
                "--out-root",
                ".",
            ]
    if dataset == "medhopqa":
        raw_root = REPO_ROOT / "data" / "medhopqa_raw"
        train_in = raw_root / "train.jsonl"
        validation_in = raw_root / "validation.jsonl"
        if all(path.exists() for path in (train_in, validation_in)):
            cmd = [
                PYTHON_BIN,
                "scripts/prepare_medical_kgqa.py",
                "--dataset",
                dataset,
                "--train-in",
                str(train_in),
                "--dev-in",
                str(validation_in),
                "--test-in",
                str(validation_in),
                "--out-root",
                ".",
            ]
            entity_names_json = str(args.entity_names_json or "").strip()
            if entity_names_json:
                cmd.extend(["--entity-names-json", entity_names_json])
            return cmd
    return None


def _maybe_build_medhop_entity_names(args: argparse.Namespace, dataset: str) -> bool:
    if dataset != "medhopqa":
        return False
    target = REPO_ROOT / "data" / "medhopqa" / "entity_names.json"
    if target.exists():
        return False
    raw_root = REPO_ROOT / "data" / "medhopqa_raw"
    if not (raw_root / "train.jsonl").exists():
        return False
    cmd = [
        PYTHON_BIN,
        "scripts/build_medhopqa_entity_names.py",
        "--raw-dir",
        str(raw_root),
        "--out-json",
        str(target),
    ]
    print("[prepare] build MedHopQA entity_names.json from DrugBank/UniProt IDs")
    _run_simple(cmd, dry_run=args.dry_run)
    return True


def _maybe_auto_convert_downloaded_medical_data(args: argparse.Namespace, dataset: str) -> bool:
    cmd = _downloaded_medical_converter_cmd(args, dataset)
    if cmd is None:
        return False
    print(f"[prepare] auto-convert downloaded raw {dataset} -> data/{dataset}")
    _run_simple(cmd, dry_run=args.dry_run)
    return True


def _ensure_raw_inputs(args: argparse.Namespace, dataset: str) -> List[Path]:
    raw_paths = _ensure_data_ready(dataset)
    raw_ready = all(path.exists() for path in raw_paths)

    if dataset in BUILTIN_HF_DATASETS and not args.skip_download and (args.force_prepare or not raw_ready):
        cmd = [
            PYTHON_BIN,
            "scripts/prepare_kgqa_data.py",
            "--dataset",
            dataset,
            "--cwq_name",
            args.rog_cwq_dataset,
            "--webqsp_name",
            args.rog_webqsp_dataset,
            "--out_root",
            ".",
        ]
        if args.hf_cache_dir:
            cmd.extend(["--cache_dir", args.hf_cache_dir])
        _run_simple(cmd, dry_run=args.dry_run)
    elif dataset not in BUILTIN_HF_DATASETS and (args.force_prepare or not raw_ready):
        auto_converted = _maybe_auto_convert_downloaded_medical_data(args, dataset)
        if auto_converted:
            if args.dry_run:
                return raw_paths
            raw_ready = all(path.exists() for path in raw_paths)
        if not raw_ready:
            missing = [str(path) for path in raw_paths if not path.exists()]
            raise FileNotFoundError(
                "Custom dataset files are missing.\n"
                f"Expected converted files under data/{dataset}/:\n" + "\n".join(missing) + "\n"
                "Run scripts/prepare_medical_kgqa.py first, then retry."
            )
    elif raw_ready:
        print(f"[skip] {dataset} raw data already present")

    if not args.dry_run:
        _check_paths(raw_paths, f"{dataset} raw")
    return raw_paths


def _run_preprocess_stage(args: argparse.Namespace, dataset: str) -> None:
    _ensure_raw_inputs(args, dataset)
    _maybe_build_medhop_entity_names(args, dataset)
    processed_paths = _ensure_processed_ready(dataset)
    processed_ready = all(path.exists() for path in processed_paths)
    common_prepare = _prepare_common_overrides(args)

    if args.force_prepare or not processed_ready:
        preprocess_cmd = [
            PYTHON_BIN,
            "-m",
            "trm_agent.run",
            "--dataset",
            dataset,
            "--stage",
            "preprocess",
            "--model_impl",
            args.model_impl,
            "--embedding_model",
            args.embedding_model,
            "--override",
            *[f"{k}={v}" for k, v in common_prepare.items()],
        ]
        _run_simple(preprocess_cmd, dry_run=args.dry_run)
    else:
        print(f"[skip] {dataset} processed jsonl already present")
    if not args.dry_run:
        _check_paths(processed_paths, f"{dataset} processed")


def _run_embed_stage(args: argparse.Namespace, dataset: str, emb_dir: Path) -> None:
    processed_paths = _ensure_processed_ready(dataset)
    processed_ready = all(path.exists() for path in processed_paths)
    if not processed_ready:
        print(f"[embed] {dataset} processed jsonl missing; running preprocess first")
        _run_preprocess_stage(args, dataset)
    elif not args.dry_run:
        _check_paths(processed_paths, f"{dataset} processed")

    emb_paths = _ensure_emb_ready(emb_dir)
    emb_ready = all(path.exists() for path in emb_paths)
    common_prepare = _prepare_common_overrides(args)

    if args.force_prepare or not emb_ready:
        embed_cmd = [
            PYTHON_BIN,
            "-m",
            "trm_agent.run",
            "--dataset",
            dataset,
            "--stage",
            "embed",
            "--model_impl",
            args.model_impl,
            "--embedding_model",
            args.embedding_model,
            "--override",
            *[f"{k}={v}" for k, v in common_prepare.items()],
        ]
        _run_simple(embed_cmd, dry_run=args.dry_run)
    else:
        print(f"[skip] {dataset} embeddings already present")
    if not args.dry_run:
        _check_paths(emb_paths, f"{dataset} embeddings")


def _prepare(args: argparse.Namespace, dataset: str, emb_dir: Path) -> None:
    _run_preprocess_stage(args, dataset)
    _run_embed_stage(args, dataset, emb_dir)


def _train(args: argparse.Namespace, dataset: str, emb_dir: Path) -> Tuple[Path, Dict[str, Path]]:
    if not args.dry_run:
        _check_paths(_ensure_emb_ready(emb_dir), f"{dataset} embeddings")

    run_root = REPO_ROOT / "logs" / "r6i5" / f"{dataset}_{args.variant}_{args.run_tag}"
    variant_tag = _variant_path_tag(args.variant)
    phase1_ckpt_dir = REPO_ROOT / "trm_agent" / "ckpt" / f"{dataset}_{args.model_impl}_rearev_{variant_tag}_phase1_{args.run_tag}"
    phase2_ckpt_dir = REPO_ROOT / "trm_agent" / "ckpt" / f"{dataset}_{args.model_impl}_rearev_{variant_tag}_phase2_{args.run_tag}"
    phase1_ckpt_dir.mkdir(parents=True, exist_ok=True)
    phase2_ckpt_dir.mkdir(parents=True, exist_ok=True)

    phase1_overrides = _phase1_overrides(args=args, dataset=dataset, emb_dir=emb_dir, ckpt_dir=phase1_ckpt_dir)
    phase1_logs, phase1_final_overrides, _ = _run_train_with_oom_guard(
        phase_name="phase1",
        dataset=dataset,
        model_impl=args.model_impl,
        ckpt="",
        base_overrides=phase1_overrides,
        primary_nproc=args.phase1_nproc,
        primary_master_port=args.phase1_master_port,
        primary_gpus=args.phase1_gpus,
        fallback_nproc=args.phase1_fallback_nproc,
        fallback_master_port=args.phase1_fallback_master_port,
        fallback_gpus=args.phase1_fallback_gpus,
        run_root=run_root,
        dry_run=args.dry_run,
        oom_retry_sleep_sec=args.oom_retry_sleep_sec,
        oom_max_retries=args.oom_max_retries,
        oom_reduce_factor=args.oom_reduce_factor,
        oom_min_nodes=args.oom_min_nodes,
        oom_min_edges=args.oom_min_edges,
    )

    if args.dry_run:
        phase1_best = phase1_ckpt_dir / f"model_ep{int(args.phase1_epochs)}.pt"
    else:
        phase1_best = _phase1_best_ckpt(phase1_ckpt_dir, phase1_logs)
    print(f"[phase1] best checkpoint: {phase1_best}")
    phase1_final_nodes = phase1_final_overrides.get("subgraph_max_nodes")
    phase1_final_edges = phase1_final_overrides.get("subgraph_max_edges")
    if phase1_final_nodes and phase1_final_edges:
        print(f"[phase1] final budget: max_nodes={phase1_final_nodes} max_edges={phase1_final_edges}")

    phase2_overrides = _phase2_overrides(
        args=args,
        dataset=dataset,
        emb_dir=emb_dir,
        ckpt_dir=phase2_ckpt_dir,
        phase1_ckpt=phase1_best,
    )
    if phase1_final_overrides.get("subgraph_max_nodes"):
        phase2_overrides["subgraph_max_nodes"] = phase1_final_overrides["subgraph_max_nodes"]
    if phase1_final_overrides.get("subgraph_max_edges"):
        phase2_overrides["subgraph_max_edges"] = phase1_final_overrides["subgraph_max_edges"]

    phase2_logs, phase2_final_overrides, phase2_log = _run_train_with_oom_guard(
        phase_name="phase2",
        dataset=dataset,
        model_impl=args.model_impl,
        ckpt=str(phase1_best),
        base_overrides=phase2_overrides,
        primary_nproc=args.phase2_nproc,
        primary_master_port=args.phase2_master_port,
        primary_gpus=args.phase2_gpus,
        fallback_nproc=args.phase2_fallback_nproc,
        fallback_master_port=args.phase2_fallback_master_port,
        fallback_gpus=args.phase2_fallback_gpus,
        run_root=run_root,
        dry_run=args.dry_run,
        oom_retry_sleep_sec=args.oom_retry_sleep_sec,
        oom_max_retries=args.oom_max_retries,
        oom_reduce_factor=args.oom_reduce_factor,
        oom_min_nodes=args.oom_min_nodes,
        oom_min_edges=args.oom_min_edges,
    )
    phase2_final_nodes = phase2_final_overrides.get("subgraph_max_nodes")
    phase2_final_edges = phase2_final_overrides.get("subgraph_max_edges")
    if phase2_final_nodes and phase2_final_edges:
        print(f"[phase2] final budget: max_nodes={phase2_final_nodes} max_edges={phase2_final_edges}")

    if args.dry_run:
        fake_phase2 = {
            "dev_hit1": phase2_ckpt_dir / f"model_ep{int(args.phase2_epochs)}.pt",
            "dev_f1": phase2_ckpt_dir / f"model_ep{int(args.phase2_epochs)}.pt",
        }
        return phase1_best, fake_phase2

    best_phase2 = _phase2_best_ckpts(phase2_ckpt_dir, phase2_log)
    print(f"[phase2] best dev_hit1 checkpoint: {best_phase2['dev_hit1']}")
    print(f"[phase2] best dev_f1 checkpoint: {best_phase2['dev_f1']}")
    return phase1_best, best_phase2


def _test_best(args: argparse.Namespace, dataset: str, emb_dir: Path) -> Path:
    if args.dry_run and not args.ckpt_dir:
        phase2_dir = (
            REPO_ROOT
            / "trm_agent"
            / "ckpt"
            / f"{dataset}_{args.model_impl}_rearev_{_variant_path_tag(args.variant)}_phase2_{args.run_tag}"
        )
    else:
        phase2_dir = Path(args.ckpt_dir) if args.ckpt_dir else _latest_phase2_dir(dataset, args.variant)
    metric = args.metric.lower()
    if metric in {"hit1", "dev_hit1"}:
        metric = "dev_hit1"
    elif metric in {"f1", "dev_f1"}:
        metric = "dev_f1"
    else:
        raise ValueError(f"Unsupported metric: {args.metric}")

    best_file = phase2_dir / f"best_{metric}.txt"
    if best_file.exists():
        ckpt_path = Path(best_file.read_text(encoding="utf-8").strip())
    elif args.dry_run:
        ckpt_path = phase2_dir / f"model_ep{int(args.phase2_epochs)}.pt"
    else:
        ckpts = sorted(
            phase2_dir.glob("model_ep*.pt"),
            key=lambda path: int(re.search(r"model_ep(\d+)\.pt$", path.name).group(1)),
        )
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {phase2_dir}")
        ckpt_path = ckpts[-1]

    if not args.dry_run and not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    test_cmd = _build_test_command(
        dataset=dataset,
        model_impl=args.model_impl,
        ckpt=str(ckpt_path),
        overrides=_test_overrides(args=args, dataset=dataset, emb_dir=emb_dir, metric=metric),
    )
    _run_simple(test_cmd, dry_run=args.dry_run)
    return ckpt_path


def _parse_test_metrics(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.exists():
        return None
    text = _read_text_safe(log_path)
    matches = TEST_SUBGRAPH_PAT.findall(text)
    if not matches:
        return None
    hit1, f1, precision, recall, skip = matches[-1]
    return {
        "hit1": float(hit1),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "skip": float(skip),
    }


def _test_range(args: argparse.Namespace, dataset: str, emb_dir: Path) -> Tuple[Path, List[Dict[str, float]]]:
    if args.dry_run and not args.ckpt_dir:
        phase2_dir = (
            REPO_ROOT
            / "trm_agent"
            / "ckpt"
            / f"{dataset}_{args.model_impl}_rearev_{_variant_path_tag(args.variant)}_phase2_{args.run_tag}"
        )
    else:
        phase2_dir = Path(args.ckpt_dir) if args.ckpt_dir else _latest_phase2_dir(dataset, args.variant)

    epoch_start = int(args.epoch_start)
    epoch_end = int(args.epoch_end)
    if epoch_end < epoch_start:
        raise ValueError(f"epoch_end({epoch_end}) must be >= epoch_start({epoch_start})")

    range_metric = str(args.range_metric or "hit1").strip().lower()
    if range_metric not in {"hit1", "f1"}:
        raise ValueError(f"Unsupported --range-metric={args.range_metric!r}. allowed=['hit1','f1']")

    run_root = REPO_ROOT / "logs" / "r6i5" / f"{dataset}_{args.variant}_{args.run_tag}" / "test_range"
    run_root.mkdir(parents=True, exist_ok=True)
    metric = args.metric.lower()
    if metric in {"hit1", "dev_hit1"}:
        metric = "dev_hit1"
    elif metric in {"f1", "dev_f1"}:
        metric = "dev_f1"
    else:
        raise ValueError(f"Unsupported metric: {args.metric}")

    rows: List[Dict[str, float]] = []
    best_row: Optional[Dict[str, float]] = None
    best_ckpt: Optional[Path] = None

    for epoch in range(epoch_start, epoch_end + 1):
        ckpt_path = phase2_dir / f"model_ep{epoch}.pt"
        if not ckpt_path.exists():
            print(f"[test-range] skip missing checkpoint: {ckpt_path}")
            continue
        test_cmd = _build_test_command(
            dataset=dataset,
            model_impl=args.model_impl,
            ckpt=str(ckpt_path),
            overrides=_test_overrides(args=args, dataset=dataset, emb_dir=emb_dir, metric=metric),
        )
        log_path = run_root / f"test_ep{epoch}.log"
        rc = _run_with_tee(
            test_cmd,
            env_overrides=_wandb_env(),
            log_path=log_path,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue
        if rc != 0:
            raise RuntimeError(f"Test failed for checkpoint: {ckpt_path}")
        metrics = _parse_test_metrics(log_path)
        if metrics is None:
            raise RuntimeError(f"Could not parse test metrics from log: {log_path}")
        row = {"epoch": float(epoch), **metrics}
        rows.append(row)
        print(
            "[test-range] "
            f"ep={epoch} hit1={metrics['hit1']:.4f} f1={metrics['f1']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
        )
        if (
            best_row is None
            or row[range_metric] > best_row[range_metric]
            or (row[range_metric] == best_row[range_metric] and row["epoch"] > best_row["epoch"])
        ):
            best_row = row
            best_ckpt = ckpt_path

    if args.dry_run:
        return phase2_dir / f"model_ep{epoch_start}.pt", rows

    if best_row is None or best_ckpt is None:
        raise FileNotFoundError(
            f"No checkpoints found in {phase2_dir} for epoch range [{epoch_start}, {epoch_end}]"
        )

    summary_path = run_root / f"summary_ep{epoch_start}_to_ep{epoch_end}.json"
    summary = {
        "dataset": dataset,
        "variant": args.variant,
        "phase2_dir": str(phase2_dir),
        "metric": range_metric,
        "best_ckpt": str(best_ckpt),
        "best_epoch": int(best_row["epoch"]),
        "rows": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[test-range] best by test_{range_metric}: ep={int(best_row['epoch'])} "
        f"hit1={best_row['hit1']:.4f} f1={best_row['f1']:.4f} ckpt={best_ckpt}"
    )
    print(f"[test-range] summary saved: {summary_path}")
    return best_ckpt, rows


def _parse_args() -> argparse.Namespace:
    default_nproc = _default_nproc()
    default_fallback_nproc = _default_fallback_nproc(default_nproc)
    default_gpus = _default_gpu_list(default_nproc)
    default_fallback_gpus = _default_gpu_list(default_fallback_nproc)

    ap = argparse.ArgumentParser(description="Windows-friendly recursive ReaRev launcher for CWQ/WebQSP/custom KGQA")
    ap.add_argument("--dataset", choices=list(SUPPORTED_LAUNCHER_DATASETS), required=True)
    ap.add_argument("--mode", choices=["preprocess", "embed", "prepare", "train", "all", "test-best", "test-range"], default="all")
    ap.add_argument("--variant", choices=["d", "dplus"], default="d")
    ap.add_argument("--model-impl", choices=["trm", "trm_hier6"], default="trm_hier6")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--recursion-steps", type=int, default=6)
    ap.add_argument("--instructions", type=int, default=5)
    ap.add_argument("--adapt-stages", type=int, default=2)
    ap.add_argument("--latent-reasoning-enabled", dest="latent_reasoning_enabled", action="store_true")
    ap.add_argument("--no-latent-reasoning", dest="latent_reasoning_enabled", action="store_false")
    ap.set_defaults(latent_reasoning_enabled=True)
    ap.add_argument("--dynamic-halting-enabled", dest="dynamic_halting_enabled", action="store_true")
    ap.add_argument("--no-dynamic-halting", dest="dynamic_halting_enabled", action="store_false")
    ap.set_defaults(dynamic_halting_enabled=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--embedding-model", default="")
    ap.add_argument("--emb-tag", default="e5")
    ap.add_argument("--embed-device", default=_default_embed_device())
    ap.add_argument("--embed-gpus", default="")
    ap.add_argument("--embed-batch-size", type=int, default=128)
    ap.add_argument("--embed-max-length", type=int, default=128)
    ap.add_argument("--embed-style", default="")
    ap.add_argument("--embed-backend", default="")
    ap.add_argument("--embed-auto-batch", dest="embed_auto_batch", action="store_true")
    ap.add_argument("--no-embed-auto-batch", dest="embed_auto_batch", action="store_false")
    ap.set_defaults(embed_auto_batch=True)
    ap.add_argument("--embed-auto-batch-min", type=int, default=4)
    ap.add_argument("--embed-auto-batch-max", type=int, default=512)
    ap.add_argument("--embed-auto-batch-vram-frac", type=float, default=0.85)
    ap.add_argument("--entity-names-json", default="")
    ap.add_argument("--primekgqa-train-override", default="")

    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force-prepare", action="store_true")
    ap.add_argument("--hf-cache-dir", default="")
    ap.add_argument("--rog-cwq-dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--rog-webqsp-dataset", default="rmanluo/RoG-webqsp")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--max-paths", type=int, default=4)
    ap.add_argument("--mine-max-neighbors", type=int, default=128)
    ap.add_argument("--preprocess-workers", type=int, default=_default_preprocess_workers())
    ap.add_argument("--train-path-policy", default="all")
    ap.add_argument("--train-shortest-k", type=int, default=1)
    ap.add_argument("--phase1-batch-size", type=int, default=0)
    ap.add_argument("--phase1-grad-accum", type=int, default=0)
    ap.add_argument("--phase2-batch-size", type=int, default=0)
    ap.add_argument("--phase2-grad-accum", type=int, default=0)
    ap.add_argument("--phase1-max-nodes", type=int, default=0)
    ap.add_argument("--phase1-max-edges", type=int, default=0)
    ap.add_argument("--phase2-max-nodes", type=int, default=0)
    ap.add_argument("--phase2-max-edges", type=int, default=0)
    ap.add_argument("--phase1-epochs", type=int, default=16)
    ap.add_argument("--phase2-epochs", type=int, default=16)
    ap.add_argument("--phase1-early-stop-enabled", dest="phase1_early_stop_enabled", action="store_true")
    ap.add_argument("--no-phase1-early-stop", dest="phase1_early_stop_enabled", action="store_false")
    ap.set_defaults(phase1_early_stop_enabled=True)
    ap.add_argument("--phase1-early-stop-metric", choices=["train_loss", "dev_hit1", "dev_f1"], default="dev_hit1")
    ap.add_argument("--phase1-early-stop-patience", type=int, default=4)
    ap.add_argument("--phase1-early-stop-min-delta", type=float, default=1e-3)
    ap.add_argument("--phase1-early-stop-min-epochs", type=int, default=8)
    ap.add_argument("--phase2-early-stop-enabled", dest="phase2_early_stop_enabled", action="store_true")
    ap.add_argument("--no-phase2-early-stop", dest="phase2_early_stop_enabled", action="store_false")
    ap.set_defaults(phase2_early_stop_enabled=True)
    ap.add_argument("--phase2-early-stop-metric", choices=["train_loss", "dev_hit1", "dev_f1"], default="dev_hit1")
    ap.add_argument("--phase2-early-stop-patience", type=int, default=4)
    ap.add_argument("--phase2-early-stop-min-delta", type=float, default=1e-3)
    ap.add_argument("--phase2-early-stop-min-epochs", type=int, default=8)

    ap.add_argument("--phase1-nproc", type=int, default=default_nproc)
    ap.add_argument("--phase1-gpus", default=default_gpus)
    ap.add_argument("--phase1-master-port", type=int, default=29675)
    ap.add_argument("--phase1-fallback-nproc", type=int, default=default_fallback_nproc)
    ap.add_argument("--phase1-fallback-gpus", default=default_fallback_gpus)
    ap.add_argument("--phase1-fallback-master-port", type=int, default=29676)
    ap.add_argument("--phase2-nproc", type=int, default=default_nproc)
    ap.add_argument("--phase2-gpus", default=default_gpus)
    ap.add_argument("--phase2-master-port", type=int, default=29679)
    ap.add_argument("--phase2-fallback-nproc", type=int, default=default_fallback_nproc)
    ap.add_argument("--phase2-fallback-gpus", default=default_fallback_gpus)
    ap.add_argument("--phase2-fallback-master-port", type=int, default=29680)

    ap.add_argument("--oom-retry-sleep-sec", type=float, default=8.0)
    ap.add_argument("--oom-max-retries", type=int, default=3)
    ap.add_argument("--oom-reduce-factor", type=float, default=0.75)
    ap.add_argument("--oom-min-nodes", type=int, default=512)
    ap.add_argument("--oom-min-edges", type=int, default=2048)

    ap.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY))
    ap.add_argument("--wandb-group", default="")

    ap.add_argument("--metric", default="dev_f1", help="Used only with --mode test-best")
    ap.add_argument("--ckpt-dir", default="", help="Optional phase2 checkpoint directory for --mode test-best")
    ap.add_argument("--epoch-start", type=int, default=20, help="Used only with --mode test-range")
    ap.add_argument("--epoch-end", type=int, default=26, help="Used only with --mode test-range")
    ap.add_argument("--range-metric", default="hit1", help="Used only with --mode test-range: hit1 or f1")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    args.emb_tag = _sanitize_tag(args.emb_tag or "e5")
    if not str(args.embedding_model).strip():
        args.embedding_model = _default_embedding_model(args.dataset)
    if not str(args.embed_style).strip():
        args.embed_style = _default_embed_style(args.dataset)
    if not str(args.embed_backend).strip():
        args.embed_backend = _default_embed_backend(args.dataset)
    if not str(args.entity_names_json).strip():
        args.entity_names_json = _default_entity_names_json(args.dataset)
    if not str(args.run_tag).strip():
        args.run_tag = f"{_recipe_display_name(args.recursion_steps, args.instructions)}_{_timestamp()}"
    original_preprocess_workers = int(args.preprocess_workers)
    args.preprocess_workers = _clamp_worker_count(args.preprocess_workers, allow_zero=True, max_workers=16)
    if args.preprocess_workers != original_preprocess_workers:
        print(
            f"[config] preprocess_workers clamped "
            f"{original_preprocess_workers}->{args.preprocess_workers}"
        )
    emb_dir = _default_emb_dir(args.dataset, args.emb_tag)

    if args.mode == "preprocess":
        _run_preprocess_stage(args, args.dataset)

    if args.mode == "embed":
        _run_embed_stage(args, args.dataset, emb_dir)

    if args.mode in {"prepare", "all"}:
        _prepare(args, args.dataset, emb_dir)

    if args.mode in {"train", "all"}:
        _train(args, args.dataset, emb_dir)

    if args.mode == "test-best":
        ckpt_path = _test_best(args, args.dataset, emb_dir)
        print(f"[test] evaluated checkpoint: {ckpt_path}")

    if args.mode == "test-range":
        best_ckpt, _ = _test_range(args, args.dataset, emb_dir)
        print(f"[test-range] selected checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
