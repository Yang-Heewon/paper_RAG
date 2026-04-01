# CWQ D+ r6i3

This directory documents the released paper reference checkpoint for the `cwq_dplus_r6i3_25x25` experiment.

Reference test metrics:

- `Hit@1 = 0.5783`
- `F1 = 0.5607`
- `Precision = 0.5575`
- `Recall = 0.6129`

The matching bundled checkpoint is:

- `paper/checkpoints/cwq_dplus_r6i3_25x25/best_checkpoint.pt`

The raw reference test log is included as `best_test.log`.

Direct evaluation command:

```powershell
python scripts/reproduce.py test-reference --experiment cwq_dplus_r6i3_25x25
```
