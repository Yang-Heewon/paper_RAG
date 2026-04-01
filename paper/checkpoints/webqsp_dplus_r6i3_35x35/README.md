## Bundled Best Checkpoint

This directory stores the released paper reference checkpoint for `webqsp_dplus_r6i3_35x35`.

Files:

- `best_checkpoint.pt`: bundled paper reference checkpoint

The checkpoint filename is intentionally generic so reviewers can download and evaluate it directly without depending on internal epoch numbering.

Evaluate it with:

```powershell
python scripts/reproduce.py test-reference --experiment webqsp_dplus_r6i3_35x35
```
