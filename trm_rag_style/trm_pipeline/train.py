from types import SimpleNamespace

from trm_unified.train_core import train as run_train


def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def run(cfg):
    args = SimpleNamespace(
        trm_root=cfg['trm_root'],
        model_impl=cfg['model_impl'],
        train_json=cfg['train_json'],
        entities_txt=cfg['entities_txt'],
        entity_names_json=cfg.get('entity_names_json', ''),
        relations_txt=cfg['relations_txt'],
        entity_emb_npy=cfg['entity_emb_npy'],
        relation_emb_npy=cfg['relation_emb_npy'],
        query_emb_train_npy=cfg['query_emb_train_npy'],
        query_emb_dev_npy=cfg.get('query_emb_dev_npy', ''),
        dev_json=cfg.get('dev_json', ''),
        ckpt=cfg.get('ckpt', ''),
        out_dir=cfg['ckpt_dir'],
        trm_tokenizer=cfg['trm_tokenizer'],
        batch_size=int(cfg['batch_size']),
        epochs=int(cfg['epochs']),
        lr=float(cfg['lr']),
        num_workers=int(cfg['num_workers']),
        max_steps=int(cfg['max_steps']),
        max_q_len=int(cfg['max_q_len']),
        max_neighbors=int(cfg['max_neighbors']),
        prune_keep=int(cfg['prune_keep']),
        prune_rand=int(cfg['prune_rand']),
        beam=int(cfg.get('beam', 5)),
        start_topk=int(cfg.get('start_topk', 5)),
        eval_beam=int(cfg.get('eval_beam', cfg.get('beam', 5))),
        eval_start_topk=int(cfg.get('eval_start_topk', cfg.get('start_topk', 5))),
        eval_max_steps=int(cfg.get('eval_max_steps', cfg['max_steps'])),
        eval_max_neighbors=int(cfg.get('eval_max_neighbors', cfg['max_neighbors'])),
        eval_prune_keep=int(cfg.get('eval_prune_keep', cfg['prune_keep'])),
        eval_no_cycle=_as_bool(cfg.get('eval_no_cycle', False)),
        eval_limit=int(cfg.get('eval_limit', -1)),
        debug_eval_n=int(cfg.get('debug_eval_n', 0)),
        eval_pred_topk=int(cfg.get('eval_pred_topk', 1)),
        eval_use_halt=_as_bool(cfg.get('eval_use_halt', False)),
        eval_min_hops_before_stop=int(cfg.get('eval_min_hops_before_stop', 1)),
        eval_random_sample_size=int(cfg.get('eval_random_sample_size', 0)),
        eval_random_seed=int(cfg.get('eval_random_seed', 42)),
        eval_random_resample_each_eval=_as_bool(cfg.get('eval_random_resample_each_eval', False)),
        eval_every_epochs=int(cfg.get('eval_every_epochs', 1)),
        eval_start_epoch=int(cfg.get('eval_start_epoch', 1)),
        endpoint_loss_mode=str(cfg.get('endpoint_loss_mode', 'aux')),
        relation_aux_weight=float(cfg.get('relation_aux_weight', 1.0)),
        endpoint_aux_weight=float(cfg.get('endpoint_aux_weight', 0.0)),
        metric_align_aux_weight=float(cfg.get('metric_align_aux_weight', cfg.get('policy_aux_weight', 0.0))),
        halt_aux_weight=float(cfg.get('halt_aux_weight', 0.0)),
        phase2_start_epoch=int(cfg.get('phase2_start_epoch', 0)),
        phase2_endpoint_loss_mode=str(cfg.get('phase2_endpoint_loss_mode', '')),
        phase2_relation_aux_weight=cfg.get('phase2_relation_aux_weight', ''),
        phase2_endpoint_aux_weight=cfg.get('phase2_endpoint_aux_weight', ''),
        phase2_metric_align_aux_weight=cfg.get('phase2_metric_align_aux_weight', ''),
        phase2_halt_aux_weight=cfg.get('phase2_halt_aux_weight', ''),
        phase2_auto_enabled=_as_bool(cfg.get('phase2_auto_enabled', False)),
        phase2_auto_metric=str(cfg.get('phase2_auto_metric', 'dev_f1')),
        phase2_auto_threshold=cfg.get('phase2_auto_threshold', ''),
        phase2_auto_patience=cfg.get('phase2_auto_patience', 0),
        phase2_auto_min_epoch=cfg.get('phase2_auto_min_epoch', 1),
        phase2_auto_min_delta=cfg.get('phase2_auto_min_delta', 1e-4),
        phase2_rl_reward_metric=str(cfg.get('phase2_rl_reward_metric', 'f1')),
        phase2_rl_entropy_weight=float(cfg.get('phase2_rl_entropy_weight', 0.0)),
        phase2_rl_sample_temp=float(cfg.get('phase2_rl_sample_temp', 1.0)),
        phase2_rl_use_greedy_baseline=_as_bool(cfg.get('phase2_rl_use_greedy_baseline', True)),
        phase2_rl_no_cycle=_as_bool(cfg.get('phase2_rl_no_cycle', cfg.get('eval_no_cycle', False))),
        phase2_rl_adv_clip=cfg.get('phase2_rl_adv_clip', ''),
        train_acc_mode=str(cfg.get('train_acc_mode', 'endpoint_proxy')),
        train_sanity_eval_every_pct=int(cfg.get('train_sanity_eval_every_pct', 0)),
        train_sanity_eval_limit=int(cfg.get('train_sanity_eval_limit', 5)),
        train_sanity_eval_beam=int(cfg.get('train_sanity_eval_beam', 5)),
        train_sanity_eval_start_topk=int(cfg.get('train_sanity_eval_start_topk', cfg.get('start_topk', 5))),
        train_sanity_eval_pred_topk=int(cfg.get('train_sanity_eval_pred_topk', 1)),
        train_sanity_eval_no_cycle=_as_bool(cfg.get('train_sanity_eval_no_cycle', cfg.get('eval_no_cycle', False))),
        train_sanity_eval_use_halt=_as_bool(cfg.get('train_sanity_eval_use_halt', False)),
        train_sanity_eval_max_neighbors=int(cfg.get('train_sanity_eval_max_neighbors', cfg.get('eval_max_neighbors', cfg['max_neighbors']))),
        train_sanity_eval_prune_keep=int(cfg.get('train_sanity_eval_prune_keep', cfg.get('eval_prune_keep', cfg['prune_keep']))),
        query_residual_enabled=_as_bool(cfg.get('query_residual_enabled', False)),
        query_residual_alpha=float(cfg.get('query_residual_alpha', 0.0)),
        query_residual_mode=str(cfg.get('query_residual_mode', 'sub_rel')),
        subgraph_reader_enabled=_as_bool(cfg.get('subgraph_reader_enabled', False)),
        subgraph_hops=int(cfg.get('subgraph_hops', 3)),
        subgraph_max_nodes=int(cfg.get('subgraph_max_nodes', 256)),
        subgraph_max_edges=int(cfg.get('subgraph_max_edges', 2048)),
        subgraph_add_reverse_edges=_as_bool(cfg.get('subgraph_add_reverse_edges', False)),
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
        subgraph_rearev_trm_halt_bce_weight=float(
            cfg.get('subgraph_rearev_trm_halt_bce_weight', 1.0)
        ),
        subgraph_rearev_trm_ce_weight=float(
            cfg.get('subgraph_rearev_trm_ce_weight', 1.0)
        ),
        subgraph_rearev_trm_weight=float(
            cfg.get('subgraph_rearev_trm_weight', 1.0)
        ),
        subgraph_deep_supervision_enabled=_as_bool(
            cfg.get('subgraph_deep_supervision_enabled', False)
        ),
        subgraph_deep_supervision_weight=float(
            cfg.get('subgraph_deep_supervision_weight', 0.0)
        ),
        subgraph_deep_supervision_ce_weight=float(
            cfg.get('subgraph_deep_supervision_ce_weight', 1.0)
        ),
        subgraph_deep_supervision_halt_weight=float(
            cfg.get('subgraph_deep_supervision_halt_weight', 1.0)
        ),
        subgraph_dropout=float(cfg.get('subgraph_dropout', 0.1)),
        subgraph_pred_threshold=float(cfg.get('subgraph_pred_threshold', 0.5)),
        subgraph_loss_mode=str(cfg.get('subgraph_loss_mode', 'rearev_kl')),
        subgraph_kl_no_positive_mode=str(cfg.get('subgraph_kl_no_positive_mode', 'uniform')),
        subgraph_kl_supervision_mode=str(cfg.get('subgraph_kl_supervision_mode', 'final')),
        subgraph_pos_weight_mode=str(cfg.get('subgraph_pos_weight_mode', 'auto')),
        subgraph_pos_weight=float(cfg.get('subgraph_pos_weight', 1.0)),
        subgraph_pos_weight_max=float(cfg.get('subgraph_pos_weight_max', 256.0)),
        subgraph_split_reverse_relations=_as_bool(cfg.get('subgraph_split_reverse_relations', False)),
        subgraph_direction_embedding_enabled=_as_bool(cfg.get('subgraph_direction_embedding_enabled', False)),
        subgraph_outer_reasoning_enabled=_as_bool(cfg.get('subgraph_outer_reasoning_enabled', False)),
        subgraph_outer_reasoning_steps=int(cfg.get('subgraph_outer_reasoning_steps', 3)),
        subgraph_gnn_variant=str(cfg.get('subgraph_gnn_variant', 'rearev_bfs')),
        subgraph_trm_rel_topk_relations=int(cfg.get('subgraph_trm_rel_topk_relations', 0)),
        subgraph_trm_rel_score_alpha=float(cfg.get('subgraph_trm_rel_score_alpha', 1.0)),
        subgraph_trm_rel_use_relid_policy=_as_bool(cfg.get('subgraph_trm_rel_use_relid_policy', True)),
        subgraph_ranking_enabled=_as_bool(cfg.get('subgraph_ranking_enabled', False)),
        subgraph_ranking_weight=float(cfg.get('subgraph_ranking_weight', 0.0)),
        subgraph_ranking_margin=float(cfg.get('subgraph_ranking_margin', 0.2)),
        subgraph_hard_negative_topk=int(cfg.get('subgraph_hard_negative_topk', 16)),
        subgraph_bce_hard_negative_enabled=_as_bool(cfg.get('subgraph_bce_hard_negative_enabled', False)),
        subgraph_bce_hard_negative_topk=int(cfg.get('subgraph_bce_hard_negative_topk', 64)),
        subgraph_grad_accum_steps=int(cfg.get('subgraph_grad_accum_steps', 1)),
        subgraph_lr_scheduler=str(cfg.get('subgraph_lr_scheduler', 'none')),
        subgraph_lr_min=float(cfg.get('subgraph_lr_min', 0.0)),
        subgraph_lr_step_size=int(cfg.get('subgraph_lr_step_size', 5)),
        subgraph_lr_gamma=float(cfg.get('subgraph_lr_gamma', 0.5)),
        subgraph_lr_plateau_factor=float(cfg.get('subgraph_lr_plateau_factor', 0.5)),
        subgraph_lr_plateau_patience=int(cfg.get('subgraph_lr_plateau_patience', 2)),
        subgraph_lr_plateau_threshold=float(cfg.get('subgraph_lr_plateau_threshold', 1e-4)),
        subgraph_lr_plateau_metric=str(cfg.get('subgraph_lr_plateau_metric', 'train_loss')),
        subgraph_early_stop_enabled=_as_bool(cfg.get('subgraph_early_stop_enabled', False)),
        subgraph_early_stop_metric=str(cfg.get('subgraph_early_stop_metric', 'dev_hit1')),
        subgraph_early_stop_patience=int(cfg.get('subgraph_early_stop_patience', 0)),
        subgraph_early_stop_min_delta=float(cfg.get('subgraph_early_stop_min_delta', 1e-4)),
        subgraph_early_stop_min_epochs=int(cfg.get('subgraph_early_stop_min_epochs', 1)),
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
        subgraph_ddp_find_unused_parameters=_as_bool(
            cfg.get('subgraph_ddp_find_unused_parameters', cfg.get('ddp_find_unused', False))
        ),
        subgraph_resume_epoch=int(cfg.get('subgraph_resume_epoch', -1)),
        train_min_path_hops=int(cfg.get('train_min_path_hops', 1)),
        freeze_lm_head=_as_bool(cfg.get('freeze_lm_head', True)),
        oracle_diag_enabled=_as_bool(cfg.get('oracle_diag_enabled', True)),
        oracle_diag_limit=int(cfg.get('oracle_diag_limit', -1)),
        oracle_diag_fail_threshold=float(cfg.get('oracle_diag_fail_threshold', -1.0)),
        oracle_diag_only=_as_bool(cfg.get('oracle_diag_only', False)),
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
        ddp_find_unused=bool(cfg.get('ddp_find_unused', False)),
        wandb_project=cfg.get('wandb_project', ''),
        wandb_entity=cfg.get('wandb_entity', ''),
        wandb_run_name=cfg.get('wandb_run_name', ''),
        wandb_group=cfg.get('wandb_group', ''),
        wandb_mode=cfg.get('wandb_mode', 'disabled'),
        seed=int(cfg.get('seed', 42)),
        deterministic=_as_bool(cfg.get('deterministic', False)),
    )
    run_train(args)
