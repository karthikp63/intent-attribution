# Diagnostic-choice prompt: fresh-rule pilot benchmark

Date: 2026-09-24. Experimental code: `diagnostic_benchmark.py`.

## Question

Does showing only the most informative branching decisions, with feature
values for each possible step, help the same LLM infer a routing rule more
reliably than showing all 36 complete trajectories?

## Locked setup

- Model: `gpt-5-mini`, 2 independent samples per rule and prompt, 16 calls.
- Four rules absent from the original held-out-rule benchmark:
  `momentum/max`, `momentum/min`, `observer_dist/min`, and
  `openness/min` followed by `momentum/max`.
- The full-path condition supplies all 36 training trajectories. The choice
  condition uses exact elimination over the project's seven known features to
  select four diagnostic branching decisions **from those same 36 episodes**
  and lists each candidate move's feature values. No other LLM feedback or
  revision occurs. Each condition makes one model call per attempt.
- Neither prompt uses the 36 held-back episodes (the other two starts). The
  symbolic scorer checks whether the answer matches every shown trajectory
  and every held-back trajectory. A second score requires the exact authored
  criterion list, not just equivalent behavior on these episodes.
- The v2 constrained criterion instructions are used in both conditions,
  with the word "trajectories" changed to "choices" where necessary. The new condition
  changes evidence selection, evidence format, and prompt length together;
  this test does **not** isolate those three effects.

## Results

| Held-out rule | Full paths, unseen episodes | Diagnostic choices, unseen episodes |
|---|---:|---:|
| Keep heading (`momentum/max`) | 0/2 | 2/2 |
| Change heading (`momentum/min`) | 1/2 | 2/2 |
| Approach observer (`observer_dist/min`) | 1/2 | 2/2 |
| Hug wall, then keep heading (two criteria) | 0/2 | 1/2 |
| **Total behavioral recovery** | **2/8** | **7/8** |
| **Exact authored criterion list** | **2/8** | **4/8** |

Every recovered proposal also fit all 36 shown trajectories. Three of the
diagnostic-choice successes had extra criteria that were behaviorally
indistinguishable on these 72 episodes; one two-criterion case still failed.

## Interpretation and limits

The focused evidence improved behavioral recovery in this small, fresh-rule
test at the same number of model calls. It does not show that the LLM uniquely
identified the subject's stated rule: different rules can make identical
predictions on the evaluated episodes. The selection algorithm already knows
the seven-feature DSL and uses observed choices to eliminate candidates, so
the gain belongs to a **symbolic evidence-selection + LLM** pipeline. It is
not evidence that the LLM can invent a rule outside the DSL.
The benchmark generates observed choices from a simulated true policy; a
human-data version would have to build them from recorded actions.

Eight attempts are not an estimate of broad reliability. All rules and
subjects here are simulated; the human pilot remains necessary. Subsequent
prompt changes should use new held-out rules or maps rather than repeatedly
tuning on these four cases. The old v1/v2 numbers in `RESULTS.md` remain
unchanged and are not directly comparable to this new task set.

## Reproduce

Run `python3 diagnostic_benchmark.py --selftest` to check the harness, then
`python3 diagnostic_benchmark.py openai gpt-5-mini --samples 2 --from-cache`
to rescore the published, prompt-verified model responses without an API key.
For fresh calls, set `OPENAI_API_KEY` and omit `--from-cache`.
