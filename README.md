# Adaptive Counterfactual Inquiry for Intent Attribution

MVP for the Yale ROSE group project on machine inference of human intent.

An observer watches a human play Kuhn poker (or Leduc hold'em) and tries to work out *what the
human is trying to do* — not which card they hold. The observer maintains an
explicit hypothesis set over intents, eliminates hypotheses inconsistent with
what it sees, and chooses its own actions to learn as much as possible.

No LLM. No learned distance metric. Exact set elimination over a finite space.

## Quick start

```bash
python3 kuhn_intent.py  --hands 2000 --condition all --seeds 1 2 3          # Kuhn, faithful subject
python3 kuhn_intent.py  --hands 2000 --condition all --seeds 1 2 3 --subject both  # + model misfit
python3 leduc_intent.py --hands 2000 --condition all --seeds 1 2 3 --subject both  # Leduc hold'em
python3 sweep.py verify                                                       # gate: Kuhn greedy == lookahead
python3 sweep.py deception                                                    # deception-aware observer
python3 sweep.py misattribute                                                 # the in-model attack
python3 proposer.py --backend fixture                                         # LLM proposer rejection rate
python3 sweep.py all                                                          # gate + cost sweeps, seeds 1-3
```

Requires Python 3.8+. No dependencies. Full numbers and interpretation in
[RESULTS.md](RESULTS.md).

Kuhn, faithful subject, n = 6000 (2000 hands x seeds 1-3), mean [min, max];
all three conditions on identical deals:

```
condition   exact ID              |H| final          sound   reveal  chips/hand
-------------------------------------------------------------------------------
passive      6.8% [6.6%,7.0%]     2.85 [2.84,2.86]   100%     100%     +0.001
random      10.6% [10.2%,10.9%]   2.55 [2.54,2.57]   100%      63%     +0.032
adaptive    20.8% [19.9%,21.6%]   2.06 [2.04,2.08]   100%      60%     +0.411
```

Adaptive probing roughly triples exact identification and cuts the surviving
intent set from 2.85 to 2.06.

## Play it yourself

```bash
python3 kuhn_intent.py --human --hands 20 --condition adaptive
```

You declare an intent *before* each action. The observer never sees the
declaration. At the end of the hand you see what it concluded.

## Design

**Intents are policies, not labels.** Each intent is a total mapping from
(decision node, card) to action — it says what the subject *would* do in
situations that did not occur. This is what makes intent a counterfactual
object rather than a category, and it is why counterfactual reasoning is the
substance of the method rather than an add-on.

| intent | policy |
|---|---|
| `value_bet` | bet a strong hand; call a raise with Q or better |
| `bluff` | bet a weak hand; fold if raised |
| `probe` | bet regardless of holding |
| `give_up` | check and fold |
| `trap` | check a strong hand to induce a bet, then call |

**The hypothesis space is joint.** The observer does not see the subject's
card, so a hypothesis is an `(intent, card)` pair. The observer's own card
removes one holding. Ten hypotheses at the start of a hand.

**The selection rule.** For each legal action, partition the current hypothesis
set by the outcome that action would induce, score by the expected size of the
resulting intent set under a uniform prior, and take the minimiser. Greedy
expected-posterior-size reduction — the simplest defensible instance of the
rule. Model counting over the constraint space is the upgrade path when a
uniform prior is inadequate.

**Conditions.**

| condition | observer behaviour |
|---|---|
| `passive` | fixed strategy; never probes |
| `random` | uniform over legal actions |
| `adaptive` | minimises expected posterior intent-set size |

**Subjects: faithful, and model misfit.** `--subject faithful` plays the
declared policy. `--subject misfit` declares honestly (ground truth unchanged),
then picks each action to maximise the observer's expected final intent-set
size — it knows the observer's rule but not its card. In Leduc
`--misfit-mode in_model` must stay consistent with some intent;
`--misfit-mode out_of_model` may play like no intent at all. (Kuhn has no
split: `sweep.py verify` proves by enumeration that all 15 (card, history)
pairs there are explained by some intent, so the model can never be left.)

> **The misfit generator is an instrument, not a subject model.** It produces
> behaviour the intent model cannot explain, at controlled rates and with
> worst-case coverage. It is not a claim that anyone plays this way.
>
> This condition used to be called "adversarial", and that name was wrong: it
> implied a subject who declares an intent and then plays against it, which is
> incoherent as a model of a person — nobody lies to a dropdown and then acts
> against their own answer. What it actually produces is **model misfit**:
> behaviour the intent vocabulary does not cover. With real subjects that is
> the common case, for ordinary reasons — the vocabulary is incomplete, they
> picked the nearest option from a menu that did not fit, they changed their
> mind mid-hand, they misread a label, they played badly. **All of those look
> identical to the observer.** The old names (`adversarial`, `impersonate`,
> `refute`) still work on the command line and in logged records.

**The consistent misattributor.** `--misfit-mode misattribute`. Declares one
intent, then plays a *different* one faithfully — the most pinnable one for its
card. It never contradicts, so the deception-aware observer never fires, and it
drives the adaptive observer to a confident wrong single intent in **98.6%** of
Leduc hands (79.2% in Kuhn). The set of never-contradicting behaviours is
exactly "play some intent's policy", so this is the whole in-model attack
surface, not one instance of it. Competence is the vulnerability: the observer
best at pinning is the most reliably pinned onto the wrong answer.

**Deception-aware observer.** `--deception-aware K` (Leduc). An empty
hypothesis set is *proof* that the subject is faithful to no intent — the
premise every conclusion rests on. After `K` such proofs the observer reports
"contradicted" rather than a single intent, and once refuted mid-hand it plays
for chips instead of drifting on. Removes 4–5 wrong confident claims per
correct one given up, and changes nothing at all against a faithful subject.

**Costs.** Both sides can be made chip-aware with a weighted objective:
`--lam` (subject: concealment − λ·chips lost) and `--mu` (observer:
|H| − μ·chips). `lam=mu=0` is the pure information game; large values are
chip maximisers. `sweep.py` produces the curves.

**Leduc observer.** `leduc_intent.py` uses exact lookahead over the rest of
the hand (`V`) instead of the one-step greedy rule. Kuhn has one betting round,
so the two must coincide there; `python3 sweep.py verify` checks it over 36,000
hands per rule, every record and every metric. The Leduc observer is a strict
generalisation of the Kuhn one.

## Metrics

| metric | meaning |
|---|---|
| `exact ID` | hypothesis set collapsed to exactly the declared intent |
| `\|H\| final` | mean intents still standing at hand's end — lower is better |
| `sound` | declared intent never wrongly eliminated |
| `reveal` | fraction of hands reaching showdown |
| `chips/hand` | observer's mean profit — the cost side of the information trade |
| `misID` | set collapsed to exactly one intent, the wrong one |
| `contra` | every hypothesis eliminated: the play matched no intent (Leduc) |
| `abstain` | observer reported "contradicted" rather than a single intent |
| `deviate` | fraction of subject decisions off the declared policy |

`|H| final` is the project's working proxy for **worst-case distinctiveness**,
the metric from the goal recognition design literature (Keren, Gal & Karpas,
ICAPS 2014).

### The gap this project addresses

**GRD assumes the actor's behaviour is generated by one of the modelled goals.
Real subjects are not.**

That is the whole of it. Goal recognition design asks how to arrange a world so
that whichever goal an agent *has* becomes evident as early as possible — and
"whichever goal" ranges over a closed set the designer wrote down. The
guarantee is conditional on the actor being in that set.

Humans are not in the set. Not because they conceal — concealment is the
special case everyone reaches for, and it is the *least* common reason — but
because a finite vocabulary of intents will not cover what a person actually
does. They pick the nearest label, change their mind, misread the menu, or act
on something nobody enumerated.

This is why the misfit condition is the centre of the project rather than a
robustness check, and why the metrics that matter are `contra` (the vocabulary
provably does not cover this) and `misID` (the observer named one intent and
was wrong) rather than `exact ID` alone. It also applies to **every**
human-subject setting, not only to games where bluffing is possible.

**Reporting.** `--seeds 1 2 3` runs several seeds and reports mean `[min, max]`
per metric; every table carries its `n`. All three conditions are replayed on
**identical deals** (the deal stream is re-seeded per condition and the observer
draws from a separate stream), so condition comparisons are **paired**.

## Why soundness matters

With a faithful simulated subject, soundness is 100% by construction. With real
humans it will not be: people do not always act on the intent they stated.

**That gap is a measurement, not a bug.** It quantifies the self-report
limitation directly rather than leaving it as a caveat. A hand where the
observer confidently converges on the wrong intent because the subject deviated
from their own declaration is a data point about human intent attribution, and
it is exactly the phenomenon the project exists to study.

## Reading the trace

`--log runs.json` writes one record per hand, including the surviving intent set
after every observed action. Example of the observer's reasoning at a single
node (subject checked; observer holds K):

```
check -> expected |intent set| = 3.571
bet   -> expected |intent set| = 2.429   <- chosen
      fold        -> [bluff, give_up, value_bet]
      call, Q     -> [trap, value_bet]
      call, J     -> [trap]
```

Betting wins because a call reveals both the card and the fold/call decision;
checking reveals only the card.

## Roadmap

- [x] Kuhn poker, 5 intents, exact elimination, three-condition comparison
- [x] Model-misfit condition — behaviour the intent model cannot explain
- [x] Leduc hold'em (larger tree, still enumerable)
- [x] Cost-aware misfit generator and observer (λ / μ sweeps)
- [x] Deception-aware observer — abstain once the model has been refuted
- [x] Consistent misattributor — in-model attack that defeats the above
- [x] Second environment (gridworld) + real GRD claim: online probing vs design
- [x] Soft elimination (eps-noise likelihood layer) — gates the human pilot
- [x] Gridworld human-pilot instrument (built, not run)
- [ ] Human pilot — run it
- [x] LLM as *proposer only* — pipeline built and validated against ground truth
- [ ] LLM proposer: an actual measured rejection rate (needs an API key)
- [ ] Concordia wrapper: custom Game Master delegating resolution to this code

## Gridworld — second environment

`python3 gridworld.py --map`. 8x8 grid, a wall with three gates, three
destinations, four routing rules; an intent is a (destination, routing rule)
pair. A probe is legible in one sentence: **close a corridor and see which way
they turn.** n = 48 per cell, **enumerated exhaustively**.

**Map rebuilt 2026-09-14 as a genuine GRD instance.** GRD requires design to
*preserve the optimal cost* to every goal, not merely keep them reachable. On the
old map every closure lengthened some route, so a faithful designer could do
nothing — and the separation reported on 2026-09-09 was measured against a
baseline with its hands tied. **That result is withdrawn.** The new map has
redundant equal-length routes; 6 of 8 configurations are now legal.

| mode | observer | exact ID | \|H\| final |
|---|---|---|---|
| online probing | passive | 37.5% | 2.00 |
| online probing | adaptive | **100.0%** | 1.00 |
| environment design (GRD) | adaptive | 87.5% | 1.12 |

**No formal separation survives.** Brute-forced over all 18 legal configurations:
best single layout 87.5%, per-start oracle 95.8%, pooled bound 100% — and online
adaptive reaches 100%, *equalling* the bound rather than exceeding it.
`gridworld.py --witness` searches for a witness pair and finds none. What remains
is a 12.5-point heuristic gap between reacting and committing in advance.

**wcd vs our metric.** GRD optimises wcd over goals; we optimise exact ID over
intents. On this map wcd is constant at 13 across every design while exact ID
swings 15/48 → 42/48 — a lever worthless for goal recognition is decisive for
intent recognition.

The misattribution result is unaffected: adaptive is misattributed on 100% of
episodes in both modes, random safest.

## Soft elimination

`python3 soft.py --verify`. Hard elimination assumes the subject is inside the
model; real people are not. A likelihood layer,
`P(a|intent) = (1-eps)[policy says a] + eps/|legal|`, turns the hypothesis set
into a posterior. Metrics are defined so that **eps = 0 reproduces every
committed number exactly** — verified per episode over 9000 episodes, 0
differences, inside `sweep.py verify`.

Against a noisy subject, hard scoring keeps naming intents confidently while its
soundness collapses (46.9% at eps = 0.5); matched scoring holds soundness at
~97% and pays in honest uncertainty. Re-scoring the real 20-hand pilot, soundness
rises 45% → 100%, but posterior mass on the declared intent stays at ≈ 0.24
against a 0.20 prior — the metric was brittle, and the underlying signal is also
weak.

## Vocabulary proposal (gridworld)

`python3 vocab.py --selftest`. The live problem is not inference, it is
vocabulary: the pilot put posterior mass on the declared intent at 0.24 against
a 0.20 prior. Proposing from an already-enumerated set adds nothing, so this
holds a routing rule **out** of the observer's vocabulary, asks an LLM to
propose a replacement from trajectories, compiles it to an executable policy,
and then **runs it on 36 episodes the LLM never saw**. Recovery = predicts
held-out behaviour exactly. Every rule is held out in turn.

**No recovery rate yet** — needs an API key. The harness is validated
(DSL expresses all built-ins 72/72; episodes distinguish all 4 rules; oracle
proposer 4/4; null and malformed proposals correctly rejected) and runs inside
`sweep.py verify` with no key.

It has already paid for itself: check 1 failed on first run and exposed that
`wall_hug` and `open_field` were inverted relative to their names — invisible to
every numeric check in the project, because it is a pure relabelling.

## LLM as proposer (poker)

`proposer.py`. The LLM **proposes**, never decides, and is never in the trust
path: it does not eliminate hypotheses, rank them, or select probes. Every
proposal is compiled to a total 36-cell policy and validated, or rejected.

**No rejection rate is reported yet.** An earlier draft carried 12.5% / 60% / 75%;
those came from a fixture written in-session by the author, so they measured the
author rather than a sampled model, and have been removed. The pipeline itself is
real and validated against ground truth (all 7 built-in intents compile to total,
legal tables). Numbers will appear when a real run happens.

`--proposer anthropic|gemini|ollama`, key from the environment only, responses
cached to disk by prompt hash so re-runs are free and reproducible. Never
imported by the experiment path — tasks 1–2 and `sweep.py verify` run with zero
dependencies and no key.

## Presentation

`docs/probing.html` — a standalone page for the group: what a probe is (the
forced-decision decomposition and a worked Kuhn node), what probing buys, the
information/chip frontier, and the case where it makes the observer confidently
wrong. Published as an artifact; open the file directly or serve the directory.

## Human pilot instrument

`python3 pilot.py --check --irb`. Built, **not run**. Intents assigned and
balanced (never self-chosen); headline metric is posterior mass on the assigned
intent against the uniform prior, not soundness — soundness is what hid the
failure last time. Ties are scored as equiprobable rather than against our
arbitrary N/E/S/W tie-break, which no human would reproduce.

It already predicts one thing: subjects following a *permissive* rule will be
systematically attributed to a more *specific* one (Occam's razor over the tie
set). Recorded in advance rather than discovered afterwards.

## Background

- Keren, Gal & Karpas, *Goal Recognition Design*, ICAPS 2014 — worst-case
  distinctiveness; the formal literature this project extends
- Kobalczyk, Astorga, Liu & van der Schaar, *Active Task Disambiguation with
  LLMs*, ICLR 2025 (arXiv:2502.04485) — Bayesian Experimental Design for
  question selection; finds LLMs cannot pick informative questions unaided
- Judson et al., *SOID* — counterfactual verification for accountability; the
  philosophical starting point
- DeepMind **Concordia** — generative agent-based modelling; candidate substrate
  for the social ("Goosietown") setting
