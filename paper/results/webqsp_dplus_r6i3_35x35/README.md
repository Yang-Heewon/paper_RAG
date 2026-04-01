## WebQSP D+ r6i3

This directory documents the released paper reference checkpoint for the `webqsp_dplus_r6i3_35x35` experiment.

Reference test metrics:

- `Hit@1 = 0.7193`
- `F1 = 0.6709`
- `Precision = 0.6732`
- `Recall = 0.7371`

The matching bundled checkpoint is:

- `paper/checkpoints/webqsp_dplus_r6i3_35x35/best_checkpoint.pt`

The raw reference test log is included as `best_test.log`.

Direct evaluation command:

```powershell
python scripts/reproduce.py test-reference --experiment webqsp_dplus_r6i3_35x35
```
