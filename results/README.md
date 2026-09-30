# Results

Additional gridworld LLM benchmark notes:

- [Diagnostic choices](diagnostic_choices_2026-09-24.md)
- [Named movable landmark](novel_landmark_2026-09-27.md)
- [Unmarked fixed place](latent_place_2026-09-27.md)
- [Open rules with active feedback](active_open_rule_2026-09-27.md)

The selected model responses for these notes are committed as
`../fixtures/published_responses.json`. Each script's `--from-cache` mode
replays them without an API key.

`sample_runs.json` — 6000 hands (2000 per condition, seed 1) against a faithful
simulated subject. One record per hand, including the surviving intent set after
every observed action.

Regenerate:

```bash
python3 ../kuhn_intent.py --hands 2000 --condition all --seed 1 --log sample_runs.json
```
