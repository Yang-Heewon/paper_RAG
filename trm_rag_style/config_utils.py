import argparse
import json
import os
from types import SimpleNamespace
from typing import Dict


def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_dicts(*items: Dict) -> Dict:
    out = {}
    for d in items:
        if d:
            out.update(d)
    return out


def dict_to_ns(d: Dict):
    return SimpleNamespace(**d)


def apply_overrides(cfg: Dict, overrides):
    if not overrides:
        return cfg
    for ov in overrides:
        if '=' not in ov:
            continue
        k, v = ov.split('=', 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if v.lower() in {'true', 'false'}:
            val = v.lower() == 'true'
        else:
            try:
                if '.' in v:
                    val = float(v)
                else:
                    val = int(v)
            except Exception:
                val = v
        cfg[k] = val
    return cfg


def default_config_dir() -> str:
    return os.path.join(os.path.dirname(__file__), 'configs')


def build_parser():
    ap = argparse.ArgumentParser(description='TRM-agent orchestrator for TRM pipeline')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--stage', choices=['preprocess', 'embed', 'train', 'test', 'all'], default='all')
    ap.add_argument('--config_dir', default=default_config_dir())
    ap.add_argument('--model_impl', choices=['trm', 'trm_hier6'], default='trm_hier6')
    ap.add_argument('--embedding_model', default='intfloat/multilingual-e5-large')
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--override', nargs='*', default=[])
    return ap


def load_run_config(config_dir: str, dataset: str, model_impl: str, embedding_model: str, ckpt: str, overrides):
    dataset_cfg = load_json(os.path.join(config_dir, f'{dataset}.json'))
    train_cfg = load_json(os.path.join(config_dir, f'train_{model_impl}.json'))
    base_cfg = load_json(os.path.join(config_dir, 'base.json'))

    emb_cfg = {
        'model_name': embedding_model,
    }

    runtime = {}
    if ckpt:
        runtime['ckpt'] = ckpt

    cfg = merge_dicts(base_cfg, dataset_cfg, train_cfg, emb_cfg, runtime)
    cfg = apply_overrides(cfg, overrides)
    return cfg
