# Paper Reproduction

This repository is trimmed for the public paper workflow:

1. prepare the dataset
2. train a named experiment
3. test the best checkpoint

The public `prepare` flow in this trimmed repo is intended for `CWQ` and `WebQSP`.

## Public Entry Points

- `scripts/reproduce.py`
- `scripts/reproduce.ps1`
- `scripts/reproduce.cmd`

## Named Experiments

| Experiment | Dataset | Variant | Recipe | Budget |
| --- | --- | --- | --- | --- |
| `cwq_dplus_r6i3_25x25` | CWQ | D+ | `r=6, i=3` | phase1=25, phase2=25 |
| `webqsp_dplus_r6i3_35x35` | WebQSP | D+ | `r=6, i=3` | phase1=35, phase2=35 |

## Environment

```powershell
pip install -r requirements.txt
```

## Prepare

```powershell
python scripts/reproduce.py prepare --dataset cwq
python scripts/reproduce.py prepare --dataset webqsp
```

## Train

```powershell
python scripts/reproduce.py train --experiment cwq_dplus_r6i3_25x25
python scripts/reproduce.py train --experiment webqsp_dplus_r6i3_35x35
```

W&B defaults:

- `wandb_mode=online`
- `wandb_project=paper_final`
- `wandb_entity=heewon6205-chung-ang-university`

Override example:

```powershell
python scripts/reproduce.py train --experiment cwq_dplus_r6i3_25x25 --wandb-mode offline
```

Inspect the exact generated launcher command:

```powershell
python scripts/reproduce.py train --experiment webqsp_dplus_r6i3_35x35 --dry-run
```

## Test

```powershell
python scripts/reproduce.py test-best --experiment cwq_dplus_r6i3_25x25 --metric dev_f1
python scripts/reproduce.py test-best --experiment webqsp_dplus_r6i3_35x35 --metric dev_hit1
```

`test-best` resolves the phase2 checkpoint directory from the experiment name and run tag automatically.

## Bundled Reference Checkpoints

This release also ships paper reference checkpoints under `paper/checkpoints/`.

Direct evaluation:

```powershell
python scripts/reproduce.py test-reference --experiment cwq_dplus_r6i3_25x25
python scripts/reproduce.py test-reference --experiment webqsp_dplus_r6i3_35x35
```

Reference metrics currently bundled in this repo:

- `CWQ D+ r6i3`: see `paper/results/cwq_dplus_r6i3_25x25/`
- `WebQSP D+ r6i3`: Hit@1 `0.7193`, F1 `0.6709`, see `paper/results/webqsp_dplus_r6i3_35x35/`

## Output Layout

- logs: `logs/r6i5/<dataset>_<variant>_<run_tag>/`
- phase1 checkpoints: `trm_agent/ckpt/<dataset>_trm_hier6_rearev_<D or Dplus>_phase1_<run_tag>/`
- phase2 checkpoints: `trm_agent/ckpt/<dataset>_trm_hier6_rearev_<D or Dplus>_phase2_<run_tag>/`

## Windows

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 list
```

`cmd.exe`:

```bat
scripts\reproduce.cmd test-best --experiment cwq_dplus_r6i3_25x25 --metric dev_f1
```
