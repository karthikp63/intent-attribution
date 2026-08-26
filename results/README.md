# Results

`sample_runs.json` — 6000 hands (2000 per condition, seed 1) against a faithful
simulated subject. One record per hand, including the surviving intent set after
every observed action.

Regenerate:

```bash
python3 ../kuhn_intent.py --hands 2000 --condition all --seed 1 --log sample_runs.json
```
