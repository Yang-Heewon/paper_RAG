# Recurrent Latent Graph Reasoning

This is a paper-facing release repository for the `D` / `D+` graph reasoning models.

The public surface is intentionally small:

- `scripts/reproduce.py`
- `scripts/reproduce.ps1`
- `scripts/reproduce.cmd`
- `paper/README.md`

If you are reproducing the paper, start with the commands below and treat the rest of the codebase as implementation detail.

This trimmed release is currently set up for the paper's `CWQ` and `WebQSP` reproduction path. The `prepare` command downloads and builds those datasets through the included preparation script.

## Named Paper Experiments

- `cwq_dplus_r6i3_25x25`
- `webqsp_dplus_r6i3_35x35`

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

List available experiments:

```powershell
python scripts/reproduce.py list
```

Prepare data:

```powershell
python scripts/reproduce.py prepare --dataset cwq
python scripts/reproduce.py prepare --dataset webqsp
```

Train:

```powershell
python scripts/reproduce.py train --experiment cwq_dplus_r6i3_25x25
python scripts/reproduce.py train --experiment webqsp_dplus_r6i3_35x35
```

Test the best checkpoint:

```powershell
python scripts/reproduce.py test-best --experiment cwq_dplus_r6i3_25x25 --metric dev_f1
python scripts/reproduce.py test-best --experiment webqsp_dplus_r6i3_35x35 --metric dev_hit1
```

Test the bundled paper reference checkpoint directly:

```powershell
python scripts/reproduce.py test-reference --experiment cwq_dplus_r6i3_25x25
python scripts/reproduce.py test-reference --experiment webqsp_dplus_r6i3_35x35
```

The repository bundles reviewer-facing reference checkpoints as:

- `paper/checkpoints/cwq_dplus_r6i3_25x25/best_checkpoint.pt`
- `paper/checkpoints/webqsp_dplus_r6i3_35x35/best_checkpoint.pt`

These files are intentionally exposed under a generic `best_checkpoint.pt` name so they can be downloaded and evaluated directly without relying on internal epoch numbering.

Windows wrappers:

```bat
scripts\reproduce.cmd list
scripts\reproduce.cmd train --experiment cwq_dplus_r6i3_25x25
```

Full paper-oriented instructions are in `paper/README.md`.
