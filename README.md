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
python3 kuhn_intent.py  --hands 2000 --condition all --seed 1 --subject both  # + adversarial subject
python3 leduc_intent.py --hands 2000 --condition all --seed 1 --subject both  # Leduc hold'em
python3 sweep.py verify                                                       # gate: Kuhn greedy == lookahead
python3 sweep.py deception                                                    # deception-aware observer
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

**Subjects.** `--subject faithful` plays the declared policy. `--subject
adversarial` declares honestly (ground truth unchanged), then picks each action
to maximise the observer's expected final intent-set size — it knows the
observer's rule but not its card. In Leduc `--adversary impersonate` must stay
consistent with some intent; `--adversary refute` may play like no intent at
all.

**Deception-aware observer.** `--deception-aware K` (Leduc). An empty
hypothesis set is *proof* that the subject is faithful to no intent — the
premise every conclusion rests on. After `K` such proofs the observer reports
"contradicted" rather than a single intent, and once refuted mid-hand it plays
for chips instead of drifting on. Removes 4–5 wrong confident claims per
correct one given up, and changes nothing at all against a faithful subject.

**Costs.** Both sides can be made chip-aware with a weighted objective:
`--lam` (adversary: concealment − λ·chips lost) and `--mu` (observer:
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
- [x] Adversarial condition — subject instructed to defeat the observer
- [x] Leduc hold'em (larger tree, still enumerable)
- [x] Cost-aware adversary and observer (λ / μ sweeps)
- [x] Deception-aware observer — abstain once the model has been refuted
- [ ] Human pilot — soundness rate for real subjects
- [ ] LLM as *proposer only* — suggests hypotheses, symbolic layer validates;
      measure the rejection rate
- [ ] Concordia wrapper: custom Game Master delegating resolution to this code

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
