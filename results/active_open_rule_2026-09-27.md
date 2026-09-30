# Open rule proposals with observable options and active feedback

Date: 2026-09-27 (Pacific). Experimental code: `active_open_rule.py`.

## Question

Would supplying the subject's available actions and circumstances, then
selecting follow-up situations and giving exact counterexamples, enable an
LLM to discover a fixed-location preference without being shown the location
or a candidate-rule menu?

## Protocol

- Simulated gridworld subject always takes a shortest route. The hidden cases
  prefer the next square nearest to (2, 0) or (4, 2), respectively. The
  no-place control prefers squares with few free neighboring cells (`wall_hug`).
- Each task starts with eight selected branching decisions from the 36
  **training** episodes (starts 0 and 1). Rows show current square, goal,
  observer, gate configuration, previous move, legal shortest next squares,
  the five relevant feature values per option, and the subject's actual move.
  The selection method used the earlier symbolic diagnostic procedure on the
  training episodes. Its candidate set included coordinates and old features;
  those candidates and the hidden location were never put in the model prompt.
- The model may propose at most two English rules with executable arithmetic
  scores each call. It is not given a list of possible rules or locations.
  The expression language has coordinates and observable feature primitives,
  bounded arithmetic, and safe symbolic execution. Extraneous response fields
  are ignored; arbitrary Python execution is disallowed.
- If no candidate fits every training episode, a symbolic controller chooses
  one previously unqueried training episode where the model's predictions
  differ. It prefers a different start/goal/gate configuration when possible,
  simulates the subject there, shows all branching choices, and reports any
  contradictions between proposals and observed moves. At most **two** such
  probes and **three** LLM calls per case. Selection sees proposal predictions
  before the new episode's actual moves; held-out episodes never guide it.
- The acceptance gate demands exact prediction of all 36 training routes.
  Accepted rules are then tested on 36 episodes from different starts (2 and
  3). One `gpt-5-mini` sample per task, maximum 24,576 completion tokens.
  Responses are cached under ignored `fixtures/cache` by full prompt, model,
  and nonce. The two location cases each have a unique coordinate fit among
  all 64 possible fixed points in the training split.

## Results

| Subject | LLM calls | New episodes shown | Accepted training rule | Held-out result |
| --- | ---: | ---: | --- | --- |
| Near (2, 0) | 3 | 2 | No; abstained | No rule to test |
| Near (4, 2) | 3 | 2 | No; abstained | No rule to test |
| Wall-hug control | 1 | 0 | Yes; one behaviorally equivalent non-place rule | 36/36 routes predicted |

All three responses in each hidden-location task were parsed and checked.
Both probes per hidden-location task produced branching observations and
contradicted the model's prior proposals; no final candidate fit all training
routes. The accepted control rule included goal-alignment terms along with
openness, so the result is **behavioral prediction**, not recovery of the
exact rule the subject was programmed to use.

An earlier exploratory run with a smaller response limit had truncated and
format-rejected answers; those outcomes are not used in this table. The final
run used the corrected parser and larger limit. An intermediate probe-selection
revision made the two new situations vary the gate setting rather than only
moving the observer. The recorded cached prompts identify the executed protocol.

## Interpretation

The proposed pipeline now exposes the agent's options and circumstances,
generates open-ended hypotheses, chooses active follow-up episodes, checks
counterexamples, allows revisions, and abstains on training mismatch. In this
small live test these improvements **did not recover either unmarked-location
rule**. The symbolic baseline that explicitly enumerates 64 possible points
recovers each once that rule family is available. The LLM has not yet shown
incremental value for discovering that feature unaided.

This is a simulated proof of concept with only two positive cases and one
control, not a reliability estimate or human validation. Initial examples
were selected by an offline symbolic teacher using training behavior; the
model was not shown its candidates, but selection is part of the system's
work. Most importantly, exact prediction of behavior does not establish the
person's mental intent when distinct rules behave identically.

## Reproduce

```
python3 active_open_rule.py --selftest
python3 active_open_rule.py openai gpt-5-mini --cases toward_2_0 toward_4_2 wall_hug --revisions 2 --from-cache
```

`--from-cache` reads the checked-in, prompt-verified model response fixture.
Fresh calls require the API key in `OPENAI_API_KEY` and omission of that flag.
No API key is recorded in code, prompts, or result files.
