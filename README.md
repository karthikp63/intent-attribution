# Adaptive Counterfactual Inquiry for Intent Attribution

MVP for the Yale ROSE group project on machine inference of human intent.

An observer watches a human play Kuhn poker and tries to work out *what the
human is trying to do* — not which card they hold. The observer maintains an
explicit hypothesis set over intents, eliminates hypotheses inconsistent with
what it sees, and chooses its own actions to learn as much as possible.

No LLM. No learned distance metric. Exact set elimination over a finite space.

## Quick start

```bash
python3 kuhn_intent.py --hands 2000 --condition all --seed 1
```

Requires Python 3.8+. No dependencies.

```
condition   exact ID  |H| final   sound   reveal  chips/hand
------------------------------------------------------------
passive        6.7%       2.84   100%    100%      -0.004
random         9.8%       2.58   100%     63%      +0.054
adaptive      21.1%       2.04   100%     61%      +0.409
```

Adaptive probing roughly triples exact identification and cuts the surviving
intent set from 2.84 to 2.04. Stable across seeds (19.7–21.6%).

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

## Metrics

| metric | meaning |
|---|---|
| `exact ID` | hypothesis set collapsed to exactly the declared intent |
| `\|H\| final` | mean intents still standing at hand's end — lower is better |
| `sound` | declared intent never wrongly eliminated |
| `reveal` | fraction of hands reaching showdown |
| `chips/hand` | observer's mean profit — the cost side of the information trade |

`|H| final` is the project's working proxy for **worst-case distinctiveness**,
the metric from the goal recognition design literature (Keren, Gal & Karpas,
ICAPS 2014).

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
- [ ] Human pilot — soundness rate for real subjects
- [ ] Leduc hold'em (larger tree, still enumerable)
- [ ] LLM as *proposer only* — suggests hypotheses, symbolic layer validates;
      measure the rejection rate
- [ ] Adversarial condition — subject instructed to defeat the observer
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
