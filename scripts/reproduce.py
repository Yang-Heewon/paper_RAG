#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "launch_experiment.py"
DEFAULT_WANDB_ENTITY = "heewon6205-chung-ang-university"
DEFAULT_WANDB_PROJECT = "paper_final"


EXPERIMENTS: Dict[str, Dict[str, object]] = {
    "cwq_dplus_r6i3_25x25": {
        "dataset": "cwq",
        "variant": "dplus",
        "model_impl": "trm_hier6",
        "recursion_steps": 6,
        "instructions": 3,
        "phase1_epochs": 25,
        "phase2_epochs": 25,
        "phase1_early_stop": False,
        "phase2_early_stop": False,
        "phase1_max_nodes": 2048,
        "phase1_max_edges": 8192,
        "phase2_max_nodes": 2048,
        "phase2_max_edges": 8192,
        "description": "CWQ main experiment: D+ / r6i3 / 25+25.",
        "reference_checkpoint": "paper/checkpoints/cwq_dplus_r6i3_25x25/best_checkpoint.pt",
    },
    "webqsp_dplus_r6i3_35x35": {
        "dataset": "webqsp",
        "variant": "dplus",
        "model_impl": "trm_hier6",
        "recursion_steps": 6,
        "instructions": 3,
        "phase1_epochs": 35,
        "phase2_epochs": 35,
        "phase1_early_stop": False,
        "phase2_early_stop": False,
        "phase1_max_nodes": 2048,
        "phase1_max_edges": 8192,
        "phase2_max_nodes": 2048,
        "phase2_max_edges": 8192,
        "description": "WebQSP main experiment: D+ / r6i3 / 35+35.",
        "reference_checkpoint": "paper/checkpoints/webqsp_dplus_r6i3_35x35/best_checkpoint.pt",
    },
}


def _variant_path_tag(variant: str) -> str:
    return "Dplus" if str(variant).strip().lower() == "dplus" else "D"


def _resolve_python(explicit: str) -> str:
    return explicit or sys.executable


def _quote(parts: List[str]) -> str:
    out: List[str] = []
    for part in parts:
        text = str(part)
        out.append(f'"{text}"' if any(ch.isspace() for ch in text) else text)
    return " ".join(out)


def _run(cmd: List[str], *, dry_run: bool) -> None:
    print(f"[run] {_quote(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _common_args(
    *,
    exp_id: str,
    exp: Dict[str, object],
    mode: str,
    run_tag: str,
    wandb_mode: str,
    wandb_project: str,
    wandb_entity: str,
) -> List[str]:
    return [
        "--dataset",
        str(exp["dataset"]),
        "--mode",
        mode,
        "--variant",
        str(exp["variant"]),
        "--model-impl",
        str(exp["model_impl"]),
        "--recursion-steps",
        str(exp["recursion_steps"]),
        "--instructions",
        str(exp["instructions"]),
        "--run-tag",
        run_tag,
        "--wandb-mode",
        wandb_mode,
        "--wandb-project",
        wandb_project,
        "--wandb-entity",
        wandb_entity,
        "--wandb-group",
        f"paper-{exp_id}",
    ]


def _phase2_ckpt_dir(exp: Dict[str, object], run_tag: str) -> Path:
    dataset = str(exp["dataset"])
    model_impl = str(exp["model_impl"])
    variant_tag = _variant_path_tag(str(exp["variant"]))
    return REPO_ROOT / "trm_agent" / "ckpt" / f"{dataset}_{model_impl}_rearev_{variant_tag}_phase2_{run_tag}"


def _reference_ckpt_path(exp: Dict[str, object]) -> Path:
    return REPO_ROOT / str(exp["reference_checkpoint"])


def _prepare(python_bin: str, dataset: str, dry_run: bool, extra_args: List[str]) -> None:
    cmd = [python_bin, str(LAUNCHER), "--dataset", dataset, "--mode", "prepare", *extra_args]
    _run(cmd, dry_run=dry_run)


def _train(
    python_bin: str,
    exp_id: str,
    exp: Dict[str, object],
    run_tag: str,
    wandb_mode: str,
    wandb_project: str,
    wandb_entity: str,
    dry_run: bool,
    extra_args: List[str],
) -> None:
    cmd = [
        python_bin,
        str(LAUNCHER),
        *_common_args(
            exp_id=exp_id,
            exp=exp,
            mode="train",
            run_tag=run_tag,
            wandb_mode=wandb_mode,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        ),
        "--phase1-epochs",
        str(exp["phase1_epochs"]),
        "--phase2-epochs",
        str(exp["phase2_epochs"]),
        "--phase1-max-nodes",
        str(exp["phase1_max_nodes"]),
        "--phase1-max-edges",
        str(exp["phase1_max_edges"]),
        "--phase2-max-nodes",
        str(exp["phase2_max_nodes"]),
        "--phase2-max-edges",
        str(exp["phase2_max_edges"]),
        "--phase1-early-stop-enabled" if exp["phase1_early_stop"] else "--no-phase1-early-stop",
        "--phase2-early-stop-enabled" if exp["phase2_early_stop"] else "--no-phase2-early-stop",
        *extra_args,
    ]
    _run(cmd, dry_run=dry_run)


def _test_best(
    python_bin: str,
    exp_id: str,
    exp: Dict[str, object],
    run_tag: str,
    metric: str,
    wandb_mode: str,
    wandb_project: str,
    wandb_entity: str,
    dry_run: bool,
    extra_args: List[str],
) -> None:
    cmd = [
        python_bin,
        str(LAUNCHER),
        *_common_args(
            exp_id=exp_id,
            exp=exp,
            mode="test-best",
            run_tag=run_tag,
            wandb_mode=wandb_mode,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        ),
        "--metric",
        metric,
        "--ckpt-dir",
        str(_phase2_ckpt_dir(exp, run_tag)),
        *extra_args,
    ]
    _run(cmd, dry_run=dry_run)


def _test_reference(
    python_bin: str,
    exp: Dict[str, object],
    dry_run: bool,
) -> None:
    dataset = str(exp["dataset"])
    variant = str(exp["variant"])
    recursion_steps = int(exp["recursion_steps"])
    instructions = int(exp["instructions"])
    model_impl = str(exp["model_impl"])
    ckpt_path = _reference_ckpt_path(exp)
    emb_dir = REPO_ROOT / "trm_agent" / "emb" / f"{dataset}_e5"
    phase2_max_nodes = str(exp["phase2_max_nodes"])
    phase2_max_edges = str(exp["phase2_max_edges"])
    latent_update_mode = "attn" if variant == "dplus" else "gru"
    gnn_variant = "rearev_dplus" if variant == "dplus" else "rearev_d"

    cmd = [
        python_bin,
        "-m",
        "trm_agent.run",
        "--dataset",
        dataset,
        "--stage",
        "test",
        "--model_impl",
        model_impl,
        "--ckpt",
        str(ckpt_path),
        "--override",
        "emb_tag=e5",
        f"emb_dir={emb_dir}",
        "batch_size=6",
        "eval_limit=-1",
        "debug_eval_n=5",
        "eval_no_cycle=true",
        "eval_max_steps=4",
        "eval_max_neighbors=256",
        "eval_prune_keep=64",
        "eval_beam=8",
        "eval_start_topk=5",
        "eval_pred_topk=5",
        "eval_use_halt=true",
        "eval_min_hops_before_stop=2",
        "subgraph_reader_enabled=true",
        "subgraph_hops=3",
        f"subgraph_max_nodes={phase2_max_nodes}",
        f"subgraph_max_edges={phase2_max_edges}",
        f"subgraph_recursion_steps={recursion_steps}",
        "subgraph_pred_threshold=0.5",
        "subgraph_split_reverse_relations=true",
        "subgraph_direction_embedding_enabled=true",
        f"subgraph_gnn_variant={gnn_variant}",
        f"subgraph_rearev_num_ins={instructions}",
        "subgraph_rearev_adapt_stages=2",
        "subgraph_rearev_latent_reasoning_enabled=true",
        "subgraph_rearev_latent_residual_alpha=0.25",
        f"subgraph_rearev_latent_update_mode={latent_update_mode}",
        "subgraph_rearev_global_gate_enabled=true",
        "subgraph_rearev_logit_global_fusion_enabled=true",
        "subgraph_rearev_dynamic_halting_enabled=true",
        "subgraph_rearev_dynamic_halting_min_steps=3",
        "subgraph_rearev_dynamic_halting_threshold=0.95",
    ]
    _run(cmd, dry_run=dry_run)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paper-facing reproduction entrypoint.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List named paper experiments.")

    prep = sub.add_parser("prepare", help="Prepare CWQ or WebQSP.")
    prep.add_argument("--dataset", choices=["cwq", "webqsp"], required=True)
    prep.add_argument("--python-bin", default="")
    prep.add_argument("--dry-run", action="store_true")
    prep.add_argument("--extra-arg", action="append", default=[])

    train = sub.add_parser("train", help="Run a named paper experiment.")
    train.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    train.add_argument("--run-tag", default="")
    train.add_argument("--python-bin", default="")
    train.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    train.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT))
    train.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY))
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--extra-arg", action="append", default=[])

    test_best = sub.add_parser("test-best", help="Run test for the best phase2 checkpoint.")
    test_best.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    test_best.add_argument("--metric", choices=["dev_hit1", "dev_f1"], default="dev_f1")
    test_best.add_argument("--run-tag", default="")
    test_best.add_argument("--python-bin", default="")
    test_best.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    test_best.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT))
    test_best.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY))
    test_best.add_argument("--dry-run", action="store_true")
    test_best.add_argument("--extra-arg", action="append", default=[])

    test_reference = sub.add_parser("test-reference", help="Run test for the bundled paper reference checkpoint.")
    test_reference.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    test_reference.add_argument("--python-bin", default="")
    test_reference.add_argument("--dry-run", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "list":
        for exp_id, spec in EXPERIMENTS.items():
            print(f"{exp_id}: {spec['description']}")
        return

    if args.cmd == "prepare":
        _prepare(
            python_bin=_resolve_python(args.python_bin),
            dataset=args.dataset,
            dry_run=args.dry_run,
            extra_args=list(args.extra_arg),
        )
        return

    exp = EXPERIMENTS[args.experiment]
    python_bin = _resolve_python(args.python_bin)

    if args.cmd == "test-reference":
        _test_reference(
            python_bin=python_bin,
            exp=exp,
            dry_run=args.dry_run,
        )
        return

    run_tag = args.run_tag or args.experiment

    if args.cmd == "train":
        _train(
            python_bin=python_bin,
            exp_id=args.experiment,
            exp=exp,
            run_tag=run_tag,
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            dry_run=args.dry_run,
            extra_args=list(args.extra_arg),
        )
        return

    _test_best(
        python_bin=python_bin,
        exp_id=args.experiment,
        exp=exp,
        run_tag=run_tag,
        metric=args.metric,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        dry_run=args.dry_run,
        extra_args=list(args.extra_arg),
    )


if __name__ == "__main__":
    main()
