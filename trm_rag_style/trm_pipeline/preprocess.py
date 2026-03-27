import json
import os
import shutil

from trm_unified.data import ensure_dir, preprocess_split


def _materialize_file(src: str, dst: str, mode: str = "symlink"):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _build_entities_with_names(entities_txt: str, entity_names_json: str, output_txt: str):
    with open(entity_names_json, "r", encoding="utf-8") as f:
        name_map = json.load(f)
    if not isinstance(name_map, dict):
        raise ValueError("entity_names_json must be a JSON object: {entity_id: name}")

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    written = 0
    with open(entities_txt, "r", encoding="utf-8") as fin, open(output_txt, "w", encoding="utf-8") as fout:
        for line in fin:
            eid = line.strip().split("\t")[0]
            if not eid:
                continue
            name = name_map.get(eid, eid)
            fout.write(f"{eid}\t{name}\n")
            written += 1
    return {"output": output_txt, "written": written}


def _custom_link_inputs(cfg):
    train_src = cfg.get("custom_train_jsonl", "")
    if not train_src:
        return None

    mode = cfg.get("custom_link_mode", "symlink")
    processed_dir = cfg["processed_dir"]
    train_dst = os.path.join(processed_dir, "train.jsonl")
    dev_src = cfg.get("custom_dev_jsonl", "") or train_src
    dev_dst = os.path.join(processed_dir, "dev.jsonl")

    _materialize_file(train_src, train_dst, mode=mode)
    _materialize_file(dev_src, dev_dst, mode=mode)

    out = {
        "train": {"mode": mode, "src": train_src, "dst": train_dst},
        "dev": {"mode": mode, "src": dev_src, "dst": dev_dst},
    }
    test_src = cfg.get("custom_test_jsonl", "")
    if test_src:
        test_dst = os.path.join(processed_dir, "test.jsonl")
        _materialize_file(test_src, test_dst, mode=mode)
        out["test"] = {"mode": mode, "src": test_src, "dst": test_dst}
    return out


def run(cfg):
    ensure_dir(cfg['processed_dir'])
    pp_workers = int(cfg.get('preprocess_workers', 0))
    if pp_workers <= 0:
        pp_workers = max(1, (os.cpu_count() or 2) - 1)

    out = {}
    entity_names_json = str(cfg.get("entity_names_json") or "").strip()
    if entity_names_json and os.path.exists(entity_names_json):
        merged_out = cfg.get("merged_entities_txt") or os.path.join(
            cfg["workspace_root"], "data", f"{cfg['dataset']}_entity_text.txt"
        )
        meta = _build_entities_with_names(
            entities_txt=cfg["entities_txt"],
            entity_names_json=entity_names_json,
            output_txt=merged_out,
        )
        cfg["entities_txt"] = merged_out
        out["entities_with_names"] = meta
    elif entity_names_json:
        print(f"[warn] entity_names_json not found: {entity_names_json} (fallback to raw entity IDs)")

    custom = _custom_link_inputs(cfg)
    if custom is not None:
        out["linked_inputs"] = custom
        print("[ok] preprocess done:", out)
        return out

    train_out = os.path.join(cfg['processed_dir'], 'train.jsonl')
    dev_out = os.path.join(cfg['processed_dir'], 'dev.jsonl')

    # Train: relation supervision from BFS-mined paths.
    tr = preprocess_split(
        dataset=cfg['dataset'],
        input_path=cfg['train_in'],
        output_path=train_out,
        entities_txt=cfg['entities_txt'],
        max_steps=int(cfg['max_steps']),
        max_paths=int(cfg['max_paths']),
        max_neighbors=int(cfg['mine_max_neighbors']),
        mine_paths=True,
        require_valid_paths=True,
        preprocess_workers=pp_workers,
        path_policy=str(cfg.get('train_path_policy', 'all')),
        shortest_k=int(cfg.get('train_shortest_k', 1)),
        progress_desc=f"{cfg['dataset']}:train",
    )
    # Dev/Test: keep endpoint-traversal tasks even when no mined path exists.
    # Evaluation uses start entities + subgraph to reach gold answer entities (Hit@1/F1).
    dv = preprocess_split(
        dataset=cfg['dataset'],
        input_path=cfg['dev_in'],
        output_path=dev_out,
        entities_txt=cfg['entities_txt'],
        max_steps=int(cfg['max_steps']),
        max_paths=int(cfg['max_paths']),
        max_neighbors=int(cfg['mine_max_neighbors']),
        mine_paths=False,
        require_valid_paths=False,
        preprocess_workers=pp_workers,
        progress_desc=f"{cfg['dataset']}:dev",
    )

    out.update({'train': tr, 'dev': dv})
    test_in = str(cfg.get('test_in') or '').strip()
    if test_in:
        if not os.path.exists(test_in):
            print(f"[warn] test_in not found: {test_in} (skip test preprocess)")
            print("[ok] preprocess done:", out)
            return out
        test_out = os.path.join(cfg['processed_dir'], 'test.jsonl')
        te = preprocess_split(
            dataset=cfg['dataset'],
            input_path=test_in,
            output_path=test_out,
            entities_txt=cfg['entities_txt'],
            max_steps=int(cfg['max_steps']),
            max_paths=int(cfg['max_paths']),
            max_neighbors=int(cfg['mine_max_neighbors']),
            mine_paths=False,
            require_valid_paths=False,
            preprocess_workers=pp_workers,
            progress_desc=f"{cfg['dataset']}:test",
        )
        out['test'] = te

    print("[ok] preprocess done:", out)
    return out
