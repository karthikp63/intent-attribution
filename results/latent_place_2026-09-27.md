# Inferring an unmarked preferred place from routes, 2026-09-27

## Question and design

The earlier movable-landmark test named the landmark and supplied its
coordinates. This test withholds both. In three synthetic cases, the hidden
tie-break rule favors shortest steps nearer to a fixed but unmarked coordinate:
(2, 0), (4, 2), or (7, 4). Three control cases use no fixed location: direct
move-order tie-break, fewest free neighboring squares, or keep heading.

Every case has 36 shown routes from two starting positions and 36 held-out
routes from two different starts. Destinations, gate settings, and observer
positions vary. The map contains walls, gates, and destinations, but does not
identify any preferred place. The model is told to propose an English rule
and a small executable numeric score or abstain. The model name is gpt-5-mini;
there was one independent API response per case and input format.

Two input formats were tested on the *same* six cases:

1. Full: all 36 shown move sequences, with their starting states.
2. Choices: up to 12 exact branching observations, selected solely from the
   shown routes by symbolic candidate elimination. Each explicitly gives
   current square, shortest available next squares, and observed move. No
   preferred coordinate or place hypothesis is disclosed.

Proposals are compiled under a bounded arithmetic grammar and checked against
**all 36 shown routes**. Only training fits are evaluated on the 36 held-out
routes. An independent symbolic baseline checks 197 existing policies (up to
two ordered tie-break criteria) and 64 possible fixed-coordinate Manhattan
distance rules. In each of the three fixed-place cases, none of the 197 old
policies fits shown behavior, while exactly one of the 64 coordinates does.
The three controls all have an old-policy fit; the direct control also fits
35 different coordinate policies, illustrating behavioral ambiguity.

## Live results

| Hidden case | Full routes | Selected branching choices | Symbolic coordinate search |
| --- | --- | --- | --- |
| Prefer (2, 0) | Wrong rule, rejected by training check | Abstained | Unique correct coordinate |
| Prefer (4, 2) | Wrong rule, rejected by training check | Abstained | Unique correct coordinate |
| Prefer (7, 4) | Wrong rule, rejected by training check | Abstained | Unique correct coordinate |
| Direct tie-break | Non-place rule fit shown and held | Non-place rule fit shown and held | 35 equivalent coordinates on shown routes |
| Wall-hug | Abstained | Abstained | No coordinate fits shown routes |
| Keep heading | Abstained | Proposed non-place rule outside allowed grammar; rejected | No coordinate fits shown routes |

The focused keep-heading proposal used `8*momentum+openness`, which the
advertised grammar excludes (`*` is unsupported). Its proposed ordering does
match all shown and held trajectories if interpreted separately, but it was
not accepted by the strict compiler. It made no fixed-place claim.

**Observed fixed-place recovery: 0/3 with full routes, 0/3 with choices.**
**Incorrect fixed-place proposals on no-place controls: 0/3 in each format.**
The latter count includes abstentions and one expression rejected for syntax;
it is not a 3/3 accuracy claim for identifying control rules.

## What this means

This trial does not show an LLM inferring an unmarked landmark just by seeing
more routes. It did not do so even when the exact coordinate was unique among
the 64 fixed-place hypotheses. The earlier 3/3 marked-landmark result therefore
should not be described as blind discovery. Exhaustive symbolic search handled
the fixed-place cases reliably *once fixed-place distance was included as a
candidate family*. Neither search establishes a person's mental intent:
different policy families can produce the same observed behavior.

This is a small exploratory test, with one response per case and three hand
chosen identifiable locations. It cannot prove the LLM could never infer an
unmarked place. More repeats and different maps or people would be needed to
estimate success frequency. A useful next test of the supplementary LLM role
would present several possible new feature families, including distractors,
and measure gains over a symbolic baseline given the same primitives and
compute budget. If no rule is identified, the system can retain its symbolic
answer or abstain; no landmark need exist in every episode.

## Reproduce

`python3 latent_place_benchmark.py --selftest` checks splits, identifiability,
compiler behavior, and candidate selection. To replay the recorded responses
without an API key, run:

```
python3 latent_place_benchmark.py openai gpt-5-mini --samples 1 --from-cache
python3 latent_place_benchmark.py openai gpt-5-mini --samples 1 --view choices --from-cache
```

`--from-cache` reads the checked-in, prompt-verified response fixture. For
fresh responses, omit the flag and set `OPENAI_API_KEY` in the environment.
Fresh call caches remain in the ignored `fixtures/cache` directory. No secret
is written to the repository.
