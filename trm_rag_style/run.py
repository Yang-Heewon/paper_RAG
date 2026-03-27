import os
import re
from contextlib import contextmanager

from .config_utils import build_parser, load_run_config
from .trm_pipeline import preprocess as trm_pre
from .trm_pipeline import embed as trm_emb
from .trm_pipeline import train as trm_train
from .trm_pipeline import test as trm_test


def _resolve_path(base_dir, raw):
    if not isinstance(raw, str) or not raw:
        return raw
    p = os.path.expanduser(os.path.expandvars(raw))
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


def _infer_resume_epoch_from_ckpt(ckpt_path: str) -> int:
    if not ckpt_path:
        return 0
    m = re.search(r"model_ep(\d+)\.pt$", os.path.basename(str(ckpt_path)))
    if not m:
        return 0
    try:
        return max(0, int(m.group(1)))
    except Exception:
        return 0


@contextmanager
def _single_process_dist_env():
    """Temporarily mask distributed env vars so test runs in single-process mode."""
    keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def normalize_config_paths(cfg):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    workspace_root = cfg.get('workspace_root') or repo_root
    workspace_root = _resolve_path(repo_root, workspace_root)
    cfg['workspace_root'] = workspace_root

    if not cfg.get('trm_root'):
        cfg['trm_root'] = os.environ.get('TRM_ROOT', '.')

    path_keys = [
        'trm_root',
        'train_in',
        'dev_in',
        'test_in',
        'entities_txt',
        'entity_names_json',
        'relations_txt',
        'custom_train_jsonl',
        'custom_dev_jsonl',
        'custom_test_jsonl',
        'merged_entities_txt',
        'processed_dir',
        'emb_dir',
        'ckpt_dir',
        'train_jsonl',
        'dev_jsonl',
        'test_jsonl',
        'train_json',
        'dev_json',
        'eval_json',
        'entity_emb_npy',
        'relation_emb_npy',
        'query_emb_train_npy',
        'query_emb_dev_npy',
        'query_emb_eval_npy',
        'ckpt',
    ]
    for k in path_keys:
        if k in cfg:
            cfg[k] = _resolve_path(workspace_root, cfg[k])
    return cfg


def enrich_paths(cfg):
    cfg['dataset'] = cfg['dataset'].lower()
    cfg['model_impl'] = cfg['model_impl']
    had_processed_dir = 'processed_dir' in cfg
    had_emb_dir = 'emb_dir' in cfg
    had_ckpt_dir = 'ckpt_dir' in cfg
    had_eval_json = 'eval_json' in cfg
    had_query_emb_eval_npy = 'query_emb_eval_npy' in cfg

    cfg.setdefault('processed_dir', os.path.join(cfg['workspace_root'], 'trm_agent', 'processed', cfg['dataset']))
    cfg.setdefault('emb_dir', os.path.join(cfg['workspace_root'], 'trm_agent', 'emb', f"{cfg['dataset']}_{cfg['emb_tag']}"))
    cfg.setdefault('ckpt_dir', os.path.join(cfg['workspace_root'], 'trm_agent', 'ckpt', f"{cfg['dataset']}_{cfg['model_impl']}"))

    # Backward compatibility: if new default dirs do not exist, transparently
    # reuse legacy trm_rag_style outputs.
    if not had_processed_dir and not os.path.exists(cfg['processed_dir']):
        legacy_processed = os.path.join(cfg['workspace_root'], 'trm_rag_style', 'processed', cfg['dataset'])
        if os.path.exists(legacy_processed):
            cfg['processed_dir'] = legacy_processed
    if not had_emb_dir:
        # Prefer legacy emb dir when default dir exists but is empty/incomplete.
        default_ent = os.path.join(cfg['emb_dir'], 'entity_embeddings.npy')
        if not os.path.exists(default_ent):
            legacy_emb = os.path.join(cfg['workspace_root'], 'trm_rag_style', 'emb', f"{cfg['dataset']}_{cfg['emb_tag']}")
            legacy_ent = os.path.join(legacy_emb, 'entity_embeddings.npy')
            if os.path.exists(legacy_ent):
                cfg['emb_dir'] = legacy_emb
    if not had_ckpt_dir and not os.path.exists(cfg['ckpt_dir']):
        legacy_ckpt = os.path.join(cfg['workspace_root'], 'trm_rag_style', 'ckpt', f"{cfg['dataset']}_{cfg['model_impl']}")
        if os.path.exists(legacy_ckpt):
            cfg['ckpt_dir'] = legacy_ckpt

    cfg['train_jsonl'] = os.path.join(cfg['processed_dir'], 'train.jsonl')
    cfg['dev_jsonl'] = os.path.join(cfg['processed_dir'], 'dev.jsonl')
    cfg['test_jsonl'] = os.path.join(cfg['processed_dir'], 'test.jsonl')
    cfg['train_json'] = cfg['train_jsonl']
    cfg['dev_json'] = cfg['dev_jsonl']
    if not had_eval_json:
        cfg['eval_json'] = cfg['test_jsonl'] if os.path.exists(cfg['test_jsonl']) else cfg['dev_jsonl']

    cfg['entity_emb_npy'] = os.path.join(cfg['emb_dir'], 'entity_embeddings.npy')
    cfg['relation_emb_npy'] = os.path.join(cfg['emb_dir'], 'relation_embeddings.npy')
    cfg['query_emb_train_npy'] = os.path.join(cfg['emb_dir'], 'query_train.npy')
    cfg['query_emb_dev_npy'] = os.path.join(cfg['emb_dir'], 'query_dev.npy')
    query_test = os.path.join(cfg['emb_dir'], 'query_test.npy')
    # Keep eval query embeddings aligned with eval split only when user did not override.
    if not had_query_emb_eval_npy:
        if cfg.get('eval_json', '') == cfg['test_jsonl']:
            cfg['query_emb_eval_npy'] = query_test
        else:
            cfg['query_emb_eval_npy'] = cfg['query_emb_dev_npy']

    return cfg


def main():
    ap = build_parser()
    args = ap.parse_args()

    cfg = load_run_config(
        config_dir=args.config_dir,
        dataset=args.dataset,
        model_impl=args.model_impl,
        embedding_model=args.embedding_model,
        ckpt=args.ckpt,
        overrides=args.override,
    )
    cfg['dataset'] = args.dataset
    cfg['model_impl'] = args.model_impl
    cfg = normalize_config_paths(cfg)
    cfg = enrich_paths(cfg)
    cfg = normalize_config_paths(cfg)

    stage = args.stage
    if stage in {'preprocess', 'all'}:
        trm_pre.run(cfg)

    if stage in {'embed', 'all'}:
        trm_emb.run(cfg)

    if stage in {'train', 'all'}:
        os.makedirs(cfg['ckpt_dir'], exist_ok=True)
        trm_train.run(cfg)
        auto_test_after_train = str(cfg.get('auto_test_after_train', False)).strip().lower() in {
            '1', 'true', 'yes', 'y', 'on'
        }
        if auto_test_after_train:
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            last_ep = int(cfg.get('epochs', 1))
            subgraph_enabled = str(cfg.get('subgraph_reader_enabled', False)).strip().lower() in {
                '1', 'true', 'yes', 'y', 'on'
            }
            if subgraph_enabled:
                try:
                    resume_ep = int(cfg.get('subgraph_resume_epoch', -1))
                except Exception:
                    resume_ep = -1
                if resume_ep < 0:
                    resume_ep = _infer_resume_epoch_from_ckpt(cfg.get('ckpt', ''))
                last_ep += max(0, int(resume_ep))
            ckpt_path = os.path.join(cfg['ckpt_dir'], f'model_ep{last_ep}.pt')
            if not os.path.exists(ckpt_path):
                print(f"[warn] auto_test_after_train enabled but checkpoint not found: {ckpt_path}")
            else:
                if world_size > 1 and rank != 0:
                    print("[AutoTest] skip on non-main rank (will run on rank0 in single-process mode)")
                    return
                test_cfg = dict(cfg)
                test_cfg['ckpt'] = ckpt_path
                print(f"[AutoTest] run test with {ckpt_path}")
                if world_size > 1:
                    with _single_process_dist_env():
                        trm_test.run(test_cfg)
                else:
                    trm_test.run(test_cfg)

    if stage == 'test':
        if not cfg.get('ckpt'):
            raise ValueError('stage=test requires --ckpt')
        trm_test.run(cfg)


if __name__ == '__main__':
    main()
