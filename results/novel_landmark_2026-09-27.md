# Movable-landmark rule discovery, 2026-09-27

## Question

Can an LLM propose a routing rule that the existing seven-feature symbolic
vocabulary cannot express? This is a simulated proof of concept for adding
new candidate rules, separate from the original four-rule recovery and the
human pilot.

## Setup

- Subject follows a shortest route, choosing the next step that minimizes its
  Manhattan distance to a movable, passable landmark. Landmark is an episode
  input; the seven old criteria do not refer to it.
- Training: 72 episodes from starts 0 and 1 with landmarks at (2, 0) and
  (7, 4). The LLM saw 13 selected branching decisions from these episodes.
  Every training episode was checked before accepting a proposal.
- Test: 72 episodes from starts 2 and 3 with new landmarks at (2, 6) and
  (5, 1). These episodes were never shown to the LLM or used to select the
  example decisions.
- Symbolic-only baseline: exhaustive list of empty, one-criterion, and two
  ordered criteria from the original seven features, in both directions: 197
  policies. A candidate must match *every* training trajectory to be usable.
- LLM: OpenAI gpt-5-mini, three independent calls to the same prompt, one
  proposal per call, no revision. Proposals contain a plain-English rule and
  a numeric expression compiled by a restricted arithmetic interpreter.
  The evaluator accepts only rules matching all 72 training trajectories;
  test episodes are scored after acceptance.

## Results

| Method | Accepted training rules | Perfect held-out predictions |
| --- | ---: | ---: |
| Existing symbolic vocabulary | 0/197 candidates | Abstains |
| LLM plus symbolic checks | 3/3 calls | 3/3 accepted rules predict 72/72 episodes |

Each LLM response proposed the same score,
`abs(row - landmark_row) + abs(col - landmark_col)`, minimized over shortest
next steps, with a corresponding English description. The safe compiler,
shown-data gate, and held-out scoring all passed. The code self-checks also
reject wrong-direction and unsupported executable expressions.

## Interpretation

This demonstrates one concrete way the LLM can supplement the old symbolic
vocabulary: it proposed an executable relation to a variable landmark that
none of the existing criteria could represent. The symbolic layer retained
control over admissibility and prediction checking.

The prompt explicitly identified the movable landmark and allowed arithmetic
over its coordinates. That makes this an **assisted new-feature test**, not
unprompted discovery of a human's intent. A programmer who anticipated the
landmark could also add its distance by hand. Three samples from one synthetic
rule do not establish reliability on other unseen rule families or people.
The next informative evaluation would preregister several different hidden
feature families, include uninformative landmarks as distractors, and compare
against a symbolic baseline with the same primitive variables and expression
budget. Human descriptions and behavior would still need separate validation.

## Reproduce

Run `python3 novel_rule_benchmark.py --selftest` without an API key. To run
the published model responses again, run
`python3 novel_rule_benchmark.py openai gpt-5-mini --samples 3 --from-cache`.
The checked-in fixture verifies the exact prompt digest. For fresh LLM calls,
set `OPENAI_API_KEY` in the environment and omit `--from-cache`. Live response
caching uses the ignored `fixtures/cache` directory; no API key is written.
