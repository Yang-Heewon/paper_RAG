from types import SimpleNamespace

from trm_unified.train_core import test as run_test


def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def run(cfg):
    args = SimpleNamespace(
        trm_root=cfg['trm_root'],
        model_impl=cfg['model_impl'],
        ckpt=cfg['ckpt'],
        entities_txt=cfg['entities_txt'],
        entity_names_json=cfg.get('entity_names_json', ''),
        relations_txt=cfg['relations_txt'],
        entity_emb_npy=cfg['entity_emb_npy'],
        relation_emb_npy=cfg['relation_emb_npy'],
        eval_json=cfg.get('eval_json', ''),
        query_emb_eval_npy=cfg.get('query_emb_eval_npy', ''),
        trm_tokenizer=cfg['trm_tokenizer'],
        batch_size=int(cfg['batch_size']),
        max_steps=int(cfg.get('max_steps', 4)),
        max_q_len=int(cfg.get('max_q_len', 512)),
        max_neighbors=int(cfg.get('max_neighbors', 256)),
        prune_keep=int(cfg.get('prune_keep', 64)),
        beam=int(cfg.get('beam', 5)),
        start_topk=int(cfg.get('start_topk', 5)),
        eval_max_steps=int(cfg.get('eval_max_steps', cfg.get('max_steps', 4))),
        eval_max_neighbors=int(cfg.get('eval_max_neighbors', cfg.get('max_neighbors', 256))),
        eval_prune_keep=int(cfg.get('eval_prune_keep', cfg.get('prune_keep', 64))),
        eval_beam=int(cfg.get('eval_beam', cfg.get('beam', 5))),
        eval_start_topk=int(cfg.get('eval_start_topk', cfg.get('start_topk', 5))),
        eval_no_cycle=_as_bool(cfg.get('eval_no_cycle', False)),
        eval_limit=int(cfg.get('eval_limit', -1)),
        debug_eval_n=int(cfg.get('debug_eval_n', 5)),
        eval_pred_topk=int(cfg.get('eval_pred_topk', 1)),
        eval_use_halt=_as_bool(cfg.get('eval_use_halt', False)),
        eval_min_hops_before_stop=int(cfg.get('eval_min_hops_before_stop', 1)),
        eval_random_sample_size=int(cfg.get('eval_random_sample_size', 0)),
        eval_random_seed=int(cfg.get('eval_random_seed', 42)),
        query_residual_enabled=_as_bool(cfg.get('query_residual_enabled', False)),
        query_residual_alpha=float(cfg.get('query_residual_alpha', 0.0)),
        query_residual_mode=str(cfg.get('query_residual_mode', 'sub_rel')),
        subgraph_reader_enabled=_as_bool(cfg.get('subgraph_reader_enabled', False)),
        subgraph_hops=int(cfg.get('subgraph_hops', 3)),
        subgraph_max_nodes=int(cfg.get('subgraph_max_nodes', 256)),
        subgraph_max_edges=int(cfg.get('subgraph_max_edges', 2048)),
        subgraph_add_reverse_edges=_as_bool(cfg.get('subgraph_add_reverse_edges', False)),
        subgraph_split_reverse_relations=_as_bool(cfg.get('subgraph_split_reverse_relations', False)),
        subgraph_direction_embedding_enabled=_as_bool(cfg.get('subgraph_direction_embedding_enabled', False)),
        subgraph_outer_reasoning_enabled=_as_bool(cfg.get('subgraph_outer_reasoning_enabled', False)),
        subgraph_outer_reasoning_steps=int(cfg.get('subgraph_outer_reasoning_steps', 3)),
        subgraph_gnn_variant=str(cfg.get('subgraph_gnn_variant', 'rearev_bfs')),
        subgraph_trm_rel_topk_relations=int(cfg.get('subgraph_trm_rel_topk_relations', 0)),
        subgraph_trm_rel_score_alpha=float(cfg.get('subgraph_trm_rel_score_alpha', 1.0)),
        subgraph_trm_rel_use_relid_policy=_as_bool(cfg.get('subgraph_trm_rel_use_relid_policy', True)),
        subgraph_recursion_steps=int(cfg.get('subgraph_recursion_steps', 8)),
        subgraph_rearev_num_ins=int(cfg.get('subgraph_rearev_num_ins', 3)),
        subgraph_rearev_adapt_stages=int(cfg.get('subgraph_rearev_adapt_stages', 1)),
        subgraph_rearev_normalized_gnn=_as_bool(cfg.get('subgraph_rearev_normalized_gnn', False)),
        subgraph_rearev_latent_reasoning_enabled=_as_bool(
            cfg.get('subgraph_rearev_latent_reasoning_enabled', False)
        ),
        subgraph_rearev_latent_residual_alpha=float(cfg.get('subgraph_rearev_latent_residual_alpha', 0.25)),
        subgraph_rearev_latent_update_mode=str(
            cfg.get('subgraph_rearev_latent_update_mode', 'gru')
        ),
        subgraph_rearev_global_gate_enabled=_as_bool(
            cfg.get('subgraph_rearev_global_gate_enabled', False)
        ),
        subgraph_rearev_logit_global_fusion_enabled=_as_bool(
            cfg.get('subgraph_rearev_logit_global_fusion_enabled', False)
        ),
        subgraph_rearev_dynamic_halting_enabled=_as_bool(
            cfg.get('subgraph_rearev_dynamic_halting_enabled', False)
        ),
        subgraph_rearev_dynamic_halting_threshold=float(
            cfg.get('subgraph_rearev_dynamic_halting_threshold', 0.9)
        ),
        subgraph_rearev_dynamic_halting_min_steps=int(
            cfg.get('subgraph_rearev_dynamic_halting_min_steps', 1)
        ),
        subgraph_rearev_trm_style_enabled=_as_bool(
            cfg.get('subgraph_rearev_trm_style_enabled', False)
        ),
        subgraph_rearev_trm_tminus1_no_grad=_as_bool(
            cfg.get('subgraph_rearev_trm_tminus1_no_grad', True)
        ),
        subgraph_rearev_trm_detach_carry=_as_bool(
            cfg.get('subgraph_rearev_trm_detach_carry', True)
        ),
        subgraph_rearev_trm_supervise_all_stages=_as_bool(
            cfg.get('subgraph_rearev_trm_supervise_all_stages', False)
        ),
        subgraph_rearev_act_stop_in_train=_as_bool(
            cfg.get('subgraph_rearev_act_stop_in_train', False)
        ),
        subgraph_rearev_asymmetric_yz_enabled=_as_bool(
            cfg.get('subgraph_rearev_asymmetric_yz_enabled', False)
        ),
        subgraph_rearev_asym_inner_y_ema_enabled=_as_bool(
            cfg.get('subgraph_rearev_asym_inner_y_ema_enabled', False)
        ),
        subgraph_rearev_asym_inner_y_ema_alpha=float(
            cfg.get('subgraph_rearev_asym_inner_y_ema_alpha', 0.0)
        ),
        subgraph_trace_relation_topk_enabled=_as_bool(
            cfg.get('subgraph_trace_relation_topk_enabled', False)
        ),
        subgraph_trace_relation_topk=int(
            cfg.get('subgraph_trace_relation_topk', 5)
        ),
        subgraph_trace_log_examples=int(
            cfg.get('subgraph_trace_log_examples', 5)
        ),
        subgraph_trace_dump_max_examples=int(
            cfg.get('subgraph_trace_dump_max_examples', 1000)
        ),
        subgraph_trace_max_examples=int(
            cfg.get('subgraph_trace_max_examples', 3)
        ),
        subgraph_trace_path_dump_jsonl=str(
            cfg.get('subgraph_trace_path_dump_jsonl', '')
        ),
        subgraph_trace_supervision_enabled=_as_bool(
            cfg.get('subgraph_trace_supervision_enabled', False)
        ),
        subgraph_trace_supervision_examples=int(
            cfg.get('subgraph_trace_supervision_examples', 5)
        ),
        subgraph_trace_supervision_dump_jsonl=str(
            cfg.get('subgraph_trace_supervision_dump_jsonl', '')
        ),
        subgraph_trace_supervision_plot_png=str(
            cfg.get('subgraph_trace_supervision_plot_png', '')
        ),
        subgraph_dropout=float(cfg.get('subgraph_dropout', 0.1)),
        subgraph_pred_threshold=float(cfg.get('subgraph_pred_threshold', 0.5)),
        eval_dump_jsonl=str(cfg.get('eval_dump_jsonl', '')),
        seq_len=int(cfg['seq_len']),
        hidden_size=int(cfg['hidden_size']),
        num_heads=int(cfg['num_heads']),
        expansion=float(cfg['expansion']),
        H_cycles=int(cfg['H_cycles']),
        L_cycles=int(cfg['L_cycles']),
        L_layers=int(cfg['L_layers']),
        puzzle_emb_len=int(cfg['puzzle_emb_len']),
        pos_encodings=cfg['pos_encodings'],
        forward_dtype=cfg['forward_dtype'],
        halt_max_steps=int(cfg['halt_max_steps']),
        halt_exploration_prob=float(cfg['halt_exploration_prob']),
    )
    run_test(args)
