from trm_unified.embedder import build_embeddings


def run(cfg):
    meta = build_embeddings(
        model_name=cfg['model_name'],
        entities_txt=cfg['entities_txt'],
        relations_txt=cfg['relations_txt'],
        train_jsonl=cfg['train_jsonl'],
        dev_jsonl=cfg['dev_jsonl'],
        test_jsonl=cfg.get('test_jsonl', ''),
        out_dir=cfg['emb_dir'],
        batch_size=int(cfg['embed_batch_size']),
        max_length=int(cfg['embed_max_length']),
        device=cfg['embed_device'],
        embed_gpus=cfg.get('embed_gpus', ''),
        entity_names_json=cfg.get('entity_names_json', ''),
        embed_style=cfg.get('embed_style', 'default'),
        embed_backend=cfg.get('embed_backend', 'auto'),
        query_prefix=str(cfg.get('embed_query_prefix', '')),
        passage_prefix=str(cfg.get('embed_passage_prefix', '')),
        save_lists=bool(cfg.get('embed_save_lists', True)),
        auto_batch=bool(cfg.get('embed_auto_batch', False)),
        auto_batch_min=int(cfg.get('embed_auto_batch_min', 4)),
        auto_batch_max=int(cfg.get('embed_auto_batch_max', 512)),
        auto_batch_vram_frac=float(cfg.get('embed_auto_batch_vram_frac', 0.85)),
    )
    print("[ok] embed done:", meta)
    return meta
