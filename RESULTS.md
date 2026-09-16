# Results

**n and seeds.** Every cell below is **n = 6000 hands** (2000 hands x seeds
1, 2, 3) unless the table says otherwise, and every cell shows the seed mean
with `[min, max]` across the three seeds. Seed stability is no longer a manual
step: `--seeds 1 2 3` on either game does this directly, and the sweeps use the
same three seeds.

**Conditions are compared on IDENTICAL deals — comparisons are PAIRED.** The
deal stream is re-seeded per condition, so `passive`, `random` and `adaptive`
each replay exactly the same shuffles and the same declarations. Differences
between conditions are therefore within-deal differences, not two independent
samples, and the `[min, max]` ranges understate the precision of a
between-condition comparison.

> **Correction (task 5).** This pairing claim used to be *false for the
> `random` condition*. The random observer drew its own actions from the same
> stream that dealt the cards, so it consumed draws the other conditions did
> not and dealt itself different hands: only **5 of 200** deals matched
> `passive`. The observer now has a separate random stream and all three
> conditions match on 200/200. **Every `random` row below therefore changed;
> no `passive` or `adaptive` row did**, and every sweep in Tasks 2-4 is
> untouched (they never used the `random` condition). The faithful
> `passive`/`adaptive` numbers reproduce the previously committed values
> exactly.

**Subjects.** **faithful** plays its declared policy. **misfit** declares
honestly (ground truth unchanged), then picks each action to maximise the
observer's expected final intent-set size, knowing the observer's selection
rule and condition but not its card (worst case, as in GRD). Ties break toward
the declared policy, so it departs from it only when that widens the gap.

> **The misfit generator is an INSTRUMENT, not a subject model.** It exists to
> produce behaviour the intent model cannot explain, at controlled rates and
> with worst-case coverage. It is *not* a claim that anyone plays this way.
>
> It used to be called "adversarial", and that name was wrong. It implied a
> subject who declares an intent and then plays against it — incoherent as a
> model of a person. Nobody lies to a dropdown and then acts against their own
> answer. What the condition actually produces is **model misfit**: behaviour
> the intent vocabulary does not cover. That happens constantly with real
> subjects, for entirely ordinary reasons — the vocabulary is incomplete, they
> picked the nearest option from a menu that did not fit, they changed their
> mind mid-hand, they misread a label, they played badly.
>
> **All of those look identical to the observer**: same signal, same
> contradiction, same failure mode. The generator maximises misfit because
> that is how you get an upper bound on it, not because subjects optimise.
> Every "the subject beats the observer" result below should be read as "the
> observer fails this badly when its vocabulary does not fit the subject".
>
> The two modes name the two ways a vocabulary can fail. `in_model`: the
> behaviour still matches *some* intent, so the vocabulary covers it and the
> misfit is only in *which* intent — the nearest-option case. `out_of_model`:
> the behaviour matches no intent at all, so the vocabulary does not cover it.
> The old names `adversarial`, `impersonate` and `refute` still work on the
> command line and in logged records.

# Headline findings

Two results are stronger than anything else in this document, and both came out
of trying to do something else. They are stated here first; the rest of the file
is the work that produced them.

---

## Goal recognition and intent recognition are different problems

This is the project's cleanest contribution, and it is a concrete instance rather
than an argument.

Goal recognition design (Keren, Gal & Karpas, ICAPS 2014) measures a design by
**wcd** — the longest prefix of an optimal path an agent can take before its
*goal* becomes clear. We measure exact identification of an **intent**, which in
the gridworld is a **(destination, routing rule)** pair: not just *where* someone
is going but *how* they are going there.

On the rebuilt gridworld these two come apart completely.

| | GRD's wcd | our exact ID |
|---|---|---|
| what it recognises | the **goal** — 3 destinations | the **intent** — 12 (destination x routing rule) pairs |
| across all 6 legal designs | **constant at 13** | **15/48 → 42/48** |
| across all 8 designs, legal *and* illegal | **constant at 13** | 15/48 → 42/48 |

Reproduce: `python3 gridworld.py --bounds`.

**A design lever that is entirely worthless for goal recognition is decisive for
intent recognition.** Closing gate 2 nearly triples exact intent identification
(15/48 → 42/48) while leaving wcd untouched. An optimal GRD designer, handed this
map, would correctly report that no redesign helps — and would be right about
goals and blind to intents.

Note the second row: wcd is constant across the *illegal* configurations too, so
this is not GRD's cost-preservation constraint biting. It is structural. All
three destinations sit in the right-hand column, so optimal paths to them diverge
late no matter which gate is open, while *how* an agent routes to them is
sensitive to exactly that choice.

> **Decision, 2026-09-14: we are not building a wcd-reducing map.** It would be
> easy to reposition destinations so different gates lead to different goals, and
> then report wcd reductions in the literature's own currency. We are not doing
> that. We went looking to *be* GRD and instead found an instance where GRD's
> objective is blind to the thing we care about; that is worth more than
> speaking their language. Revisit only if (i) a reviewer demands the comparison,
> or (ii) intent recognition proves intractable on human subjects and we need to
> retreat to goal recognition as the tractable version.

## Confident misattribution — a negative result about our own method

The most robust finding we have. It has survived a port from poker to a grid, a
complete map rebuild, and both interaction modes.

A subject declares one intent and then faithfully plays **a different one**. It
never leaves the model — every action is generated by some intent in the
vocabulary, so nothing is ever contradicted and no deception is involved.

```
environment   mode                observer    misID
Leduc         (poker)             adaptive    98.6%
Kuhn          (poker)             adaptive    79.2%
gridworld     online probing      adaptive   100.0%
gridworld     environment design  adaptive   100.0%
```

In every case the ordering is the same: the observer that is **best** at
narrowing the hypothesis set is the one most reliably narrowed onto the wrong
answer, and the *random* observer is safest because it learns least.

**Why this is not yet a contribution.** A negative result about one's own method
earns a place only if it comes with a structural explanation or a fix. **We have
neither**, and the document should not imply otherwise:

* The observer is **right about the policy and wrong only about the label.** It
  correctly identifies the behaviour as `probe`, or as `A/open_field`; the
  subject simply called it something else.
* **No action history separates those two situations.** "Is executing X" and
  "believes they are executing Y while executing X" generate identical
  trajectories by construction. There is no probe, no layout, and no amount of
  elimination that distinguishes them, because the distinguishing information is
  not in the behaviour.

What we can currently say is a bound on the whole behavioural approach in this
setting, not a flaw in one implementation of it.

## The two together

They point at one thing: **behaviour underdetermines intent, in ways the existing
literature's objective does not capture.**

wcd says a design is useless when it cannot separate *goals*, and is silent when
the same design separates *intents* perfectly. Misattribution says that even
perfect separation of intents-as-policies does not recover the intent a person
would *report*, because the label and the policy can come apart with no
behavioural trace.

The pilot rescore is the empirical counterpart: posterior mass on the intent a
real person declared was **0.24 against a 0.20 uniform prior** (see *Soft
elimination*). The vocabulary does not fit, and better inference over a bad
vocabulary does not help. That is what makes vocabulary proposal the live
problem rather than a nice-to-have.

---

# The work

## Kuhn (`kuhn_intent.py`, 5 intents, 10 hypotheses)

`python3 kuhn_intent.py --hands 2000 --condition all --subject both --seeds 1 2 3
--observer lookahead`  (n = 6000 per row)

```
condition                    exact ID           |H| final              sound   misID           reveal          deviate   chips/hand
passive/faithful       6.8% [6.6%,7.0%]   2.85 [2.84,2.86]              100%    0.0%             100%             0.0%   +0.001 [-0.011,+0.018]
random/faithful      10.6% [10.2%,10.9%]  2.55 [2.54,2.57]              100%    0.0%   63.4% [62.5,63.9]          0.0%   +0.032 [+0.011,+0.049]
adaptive/faithful    20.8% [19.9%,21.6%]  2.06 [2.04,2.08]              100%    0.0%   59.8% [58.3,61.1]          0.0%   +0.411 [+0.375,+0.449]
passive/misfit            0.0%       3.32 [3.31,3.33]  66.3% [65.5,67.2]   0.0%             100%   33.7% [32.8,34.5]  +0.007 [-0.009,+0.021]
random/misfit             0.0%       3.00 [2.99,3.02]  59.9% [58.9,61.2]   0.0%   50.5% [50.3,50.7]  40.3% [39.7,41.2]  +0.489 [+0.471,+0.506]
adaptive/misfit           0.0%       2.67 [2.66,2.68]  53.5% [52.0,54.5]   0.0%             0.0%    43.5% [42.6,44.4]  +1.000 [+1.000,+1.000]
```

Degradation under misfit: exact ID 20.8% → 0%, |H| final 2.06 → 2.67,
soundness 100% → 53.5%. Adaptive still ends with the smallest set. The generator's
best reply to the adaptive observer is check-then-fold every hand (reveal 0%):
the largest reachable set in Kuhn is the check-fold bucket {bluff, give_up,
value_bet}. It pays 1 chip/hand for that concealment — this generator has no chip
objective, which is a modelling choice to revisit (a cost-aware one is the
natural next variant, and is Task 2).

### Greedy vs. lookahead observer (gate)

Kuhn's original adaptive rule is one-step greedy (`expected_posterior_size`);
Leduc's is exact lookahead over the rest of the hand (`V`). `kuhn_intent.py`
carries an independent history-based implementation of the lookahead rule
(`--observer lookahead`, the same code shape as Leduc's).

**These must coincide in Kuhn.** Kuhn has one betting round, so at every
observer decision the remaining subtree is at most one subject reply plus the
showdown — there is no "later" to look ahead to, and the lookahead recursion
bottoms out in exactly the expectation the greedy rule already scores. A
difference anywhere would mean one implementation is wrong, not that one rule
is better.

Verified over **36,000 hands per rule** (2000 hands x 3 seeds x 3 conditions x
2 subject types; 72,000 hands played), comparing **every field of every
per-hand record** and **every aggregate metric** (exact ID, |H| final,
soundness, misID, contradiction, deviation, chips/hand, reveal):

```
per-hand records differing:   0
aggregate metrics differing:  0 cells (18 cells)
```

The comparison is per hand, not only in aggregate: two different rules could
produce identical means over 2000 hands while disagreeing on individual hands,
so an aggregate-only check would not catch a real divergence.

**The Leduc observer is a strict generalisation of the Kuhn one.** Kuhn is the
one-round special case of the same recursion; the greedy rule is not a separate
method but that recursion evaluated at depth 1.

The same run confirms that **no Kuhn hand ever reaches an empty hypothesis
set** (0 of 72,000) — and `kuhn_coverage()` in the same gate proves it
**exhaustively** rather than by sampling: all **15** (card, complete history)
pairs in the Kuhn tree are explained by at least one intent, so the hypothesis
set can never empty there. This is why Kuhn has no `in_model`/`out_of_model`
split: the two modes differ only in how they score an empty final set, an
outcome the game cannot produce. `--misfit-mode` exists only in Leduc.

Reproduce:

```
python3 sweep.py verify
```

## Leduc (`leduc_intent.py`, 7 intents, 35 hypotheses, two betting rounds)

`python3 leduc_intent.py --hands 2000 --condition all --subject both --seeds 1 2 3`
(n = 6000 per row)

```
condition                              exact ID           |H| final              sound              misID             contra            deviate   chips/hand
passive/faithful                 28.0% [27.4,28.7]  3.16 [3.12,3.23]              100%               0.0%               0.0%               0.0%   -0.121 [-0.207,-0.072]
random/faithful                  20.4% [19.9,20.9]  3.29 [3.27,3.33]              100%               0.0%               0.0%               0.0%   -0.243 [-0.270,-0.225]
adaptive/faithful                38.5% [37.8,39.4]  2.65 [2.62,2.66]              100%               0.0%               0.0%               0.0%   -0.509 [-0.539,-0.490]
passive/misfit:in_model            0.0%     4.19 [4.17,4.22]  60.5% [59.8,61.8]              0.0%               0.0%  24.3% [23.7,24.6]   -0.016 [-0.036,-0.004]
random/misfit:in_model             0.0%     4.72 [4.71,4.74]  67.4% [66.3,68.6]              0.0%               0.0%  35.6% [35.0,36.0]   +0.740 [+0.736,+0.744]
adaptive/misfit:in_model           0.0%     4.64 [4.63,4.65]  65.7% [64.3,67.0]              0.0%               0.0%  36.6% [36.4,36.8]   +0.903 [+0.901,+0.907]
passive/misfit:out_of_model                 0.0%     1.96 [1.88,2.00]  28.3% [27.1,29.3]              0.0%  46.2% [45.0,48.0]  54.0% [52.8,56.0]   -0.182 [-0.216,-0.147]
random/misfit:out_of_model          1.2% [1.1,1.4]   1.85 [1.78,1.91]  27.0% [26.2,27.9]     5.4% [5.2,5.9]  51.9% [51.0,53.1]  38.8% [38.6,39.2]   +0.488 [+0.471,+0.501]
adaptive/misfit:out_of_model        1.5% [1.2,1.8]   1.64 [1.60,1.66]  23.3% [23.0,23.9]     6.4% [6.0,6.7]  56.7% [56.3,57.5]  51.9% [51.5,52.7]   +0.433 [+0.279,+0.515]
```

* **Not a Kuhn artifact.** Adaptive beats passive and random on both exact ID
  (38.5% vs 28.0% / 20.4%) and |H| final (2.65 vs 3.16 / 3.29), on identical
  deals in all three conditions. The observer's
  rule here is exact lookahead over the rest of the hand (in Kuhn that collapses
  to the one-step greedy rule).
* **The information trade now costs chips.** In Kuhn adaptive made money; in
  Leduc it loses 0.51/hand because probing is a 4-chip bet in round 2. Both are
  real; the selection rule ignores payoff by design, and this is what a
  cost-aware rule would need to fix.
* **Two misfit modes.** `in_model` must stay consistent with some intent
  (empty set scores 0 for it); `out_of_model` may play like nobody (empty set scores
  as full concealment). They are qualitatively different opponents:
  in_model drives |H| up (4.64) with 0% exact ID; out_of_model drives the observer
  into contradiction 50–58% of the time and produces misattribution (6.7% of
  hands end on exactly one intent, the wrong one — worse than ambiguity).
  Which one matters is a question for the group; both are reported. They are
  two different ways for a vocabulary to fail, not two different opponents.
* **Soundness is now a measurement.** 100% by construction for faithful play;
  60–68% for in_model, 23–28% for out_of_model. This is the number a human pilot
  will fill in for real subjects.

Representative Leduc trace (declared bluff, subject K1, observer J0):

```
check → bet → call → board J1 → check → bet → call     final: [bluff, represent]
  after subject checks, live intents: bluff give_up pot_control represent trap value_bet
  observer check → expected final |intent set| = 3.060
  observer bet   → expected final |intent set| = 3.040   ← chosen
```


## Task 2: cost-aware misfit generator — how much concealment does a chip buy?

Adversary objective: maximise `E[final |intent set|] - lam * E[chips lost]`.
**Weighted, not a hard budget**, because the generator is already an
expectation-max recursion over the tree: a weight folds into the terminal
value and every node stays a plain max, whereas a hard constraint on expected
loss needs a Lagrangian (i.e. this `lam`, found by search) or a constrained
search over mixed strategies. `lam=0` is the pure concealer from the first
tables; `lam=1000` is effectively a pure chip maximiser (concealment only
breaks ties, toward the declared policy). Observer: adaptive, `mu=0`.
Cells: mean over seeds 1–3 [min, max], 2000 hands each.
`python3 sweep.py misfit`

### Kuhn
```
setting                        exact                     H                 sound                 misID                contra               deviate                 chips
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
faithful         20.8% [19.9%,21.6%]      2.06 [2.04,2.08]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.411 [+0.375,+0.449]
lam=0               0.0% [0.0%,0.0%]      2.67 [2.66,2.68]   53.5% [52.0%,54.5%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   43.5% [42.6%,44.4%]+1.000 [+1.000,+1.000]
lam=0.1             0.0% [0.0%,0.0%]      2.67 [2.66,2.68]   53.5% [52.0%,54.5%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   43.5% [42.6%,44.4%]+1.000 [+1.000,+1.000]
lam=0.25            0.0% [0.0%,0.0%]      2.50 [2.50,2.51]   50.5% [49.5%,51.4%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   43.8% [42.8%,44.7%]-0.008 [-0.026,+0.025]
lam=0.5             0.0% [0.0%,0.0%]      2.41 [2.40,2.42]   59.3% [58.8%,60.3%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   32.4% [30.9%,33.3%]-0.191 [-0.195,-0.186]
lam=1               0.0% [0.0%,0.0%]      2.34 [2.34,2.35]   45.8% [45.2%,46.9%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   40.5% [38.9%,41.3%]-0.328 [-0.354,-0.303]
lam=2               0.0% [0.0%,0.0%]      2.34 [2.34,2.35]   45.8% [45.2%,46.9%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   40.5% [38.9%,41.3%]-0.328 [-0.354,-0.303]
lam=4               0.0% [0.0%,0.0%]      2.34 [2.34,2.35]   45.8% [45.2%,46.9%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   40.5% [38.9%,41.3%]-0.328 [-0.354,-0.303]
lam=1000            0.0% [0.0%,0.0%]      2.34 [2.34,2.35]   45.8% [45.2%,46.9%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   40.5% [38.9%,41.3%]-0.328 [-0.354,-0.303]
```

### Leduc, in_model
```
setting                        exact                     H                 sound                 misID                contra               deviate                 chips
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
faithful         38.5% [37.8%,39.4%]      2.65 [2.62,2.66]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]-0.509 [-0.539,-0.490]
lam=0               0.0% [0.0%,0.0%]      4.64 [4.63,4.65]   65.7% [64.3%,67.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   36.6% [36.4%,36.8%]+0.903 [+0.901,+0.907]
lam=0.1             0.0% [0.0%,0.0%]      4.64 [4.63,4.65]   65.7% [64.3%,67.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   36.6% [36.4%,36.8%]+0.903 [+0.901,+0.907]
lam=0.25            0.0% [0.0%,0.0%]      4.57 [4.56,4.58]   64.6% [63.2%,65.8%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   36.3% [36.2%,36.4%]+0.573 [+0.551,+0.587]
lam=0.5             0.9% [0.8%,1.1%]      3.95 [3.94,3.96]   58.2% [56.8%,59.7%]      0.0% [0.0%,0.0%]      6.7% [6.4%,7.3%]   30.7% [30.4%,31.2%]-0.741 [-0.798,-0.696]
lam=1               8.6% [8.3%,9.1%]      1.87 [1.84,1.91]   29.1% [27.8%,31.3%]   40.7% [39.9%,41.9%]    10.1% [9.6%,11.1%]   38.2% [37.7%,38.8%]-4.304 [-4.399,-4.193]
lam=2             10.0% [9.4%,10.7%]      1.57 [1.54,1.59]   25.0% [23.2%,27.3%]   49.2% [48.0%,50.6%]    10.1% [9.6%,11.1%]   39.6% [39.1%,40.3%]-4.503 [-4.577,-4.409]
lam=4            10.8% [10.4%,11.5%]      1.30 [1.29,1.31]   21.2% [20.2%,22.9%]   55.0% [53.8%,56.1%]    10.1% [9.6%,11.1%]   41.2% [40.8%,41.9%]-4.604 [-4.679,-4.510]
lam=1000         10.8% [10.4%,11.5%]      1.30 [1.29,1.31]   21.2% [20.2%,22.9%]   55.0% [53.8%,56.1%]    10.1% [9.6%,11.1%]   41.2% [40.8%,41.9%]-4.604 [-4.679,-4.510]
```

### Leduc, out_of_model
```
setting                        exact                     H                 sound                 misID                contra               deviate                 chips
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
faithful         38.5% [37.8%,39.4%]      2.65 [2.62,2.66]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]-0.509 [-0.539,-0.490]
lam=0               1.5% [1.2%,1.8%]      1.64 [1.60,1.66]   23.3% [23.0%,23.9%]      6.4% [6.0%,6.7%]   56.7% [56.3%,57.5%]   51.9% [51.5%,52.7%]+0.433 [+0.279,+0.515]
lam=0.1             1.5% [1.2%,1.8%]      1.46 [1.42,1.50]   20.6% [20.2%,21.2%]      8.7% [8.5%,8.9%]   59.4% [58.6%,60.3%]   56.5% [55.8%,57.7%]-0.068 [-0.245,+0.029]
lam=0.25            1.5% [1.2%,1.8%]      1.46 [1.42,1.50]   20.6% [20.2%,21.2%]      8.7% [8.5%,8.9%]   59.4% [58.6%,60.3%]   56.5% [55.8%,57.7%]-0.068 [-0.245,+0.029]
lam=0.5             1.1% [0.9%,1.1%]      1.10 [1.06,1.14]   16.4% [15.8%,16.7%]      5.8% [5.5%,6.1%]   56.4% [54.6%,58.1%]   54.1% [53.2%,55.9%]-1.312 [-1.436,-1.208]
lam=1               5.0% [5.0%,5.1%]      1.70 [1.67,1.75]   25.0% [24.1%,26.6%]   30.3% [30.2%,30.5%]   20.7% [19.6%,21.6%]   41.8% [41.3%,42.3%]-3.966 [-4.103,-3.857]
lam=2             10.0% [9.4%,10.7%]      1.54 [1.51,1.56]   25.0% [23.2%,27.3%]   46.6% [45.8%,47.5%]   12.8% [12.3%,13.4%]   39.3% [38.8%,39.9%]-4.422 [-4.508,-4.328]
lam=4            10.8% [10.4%,11.5%]      1.30 [1.29,1.31]   21.2% [20.2%,22.9%]   55.0% [53.8%,56.1%]    10.1% [9.6%,11.1%]   41.2% [40.8%,41.9%]-4.604 [-4.679,-4.510]
lam=1000         10.8% [10.4%,11.5%]      1.30 [1.29,1.31]   21.2% [20.2%,22.9%]   55.0% [53.8%,56.1%]    10.1% [9.6%,11.1%]   41.2% [40.8%,41.9%]-4.604 [-4.679,-4.510]
```

**Reading.**

* **Kuhn: concealment is free.** The pure concealer (`lam=0`) pays a chip a
  hand for 0% exact ID. At `lam=0.25` the subject is break-even (−0.008)
  and exact ID is still 0%, |H| 2.50 vs 2.06 faithful. At `lam>=1` the
  subject is a pure chip maximiser: it *beats* the adaptive observer
  (−0.33/hand for the observer, which made +0.41 against a faithful subject),
  and exact ID is still 0%. The observer's identification advantage in Kuhn
  existed only because the subject was faithful; a player who simply plays to
  win conceals intent as a side effect. "Your misfit generator is unrealistic"
  does not rescue the observer — the realistic subject is worse for it. This is
  the sharpest form of the reframing: a person who is simply playing to win,
  with no interest in the observer at all, produces the same signal.
* **Leduc, in_model: break-even still hurts.** Break-even sits between
  `lam=0.25` (observer +0.57) and `lam=0.5` (observer −0.74); on both sides
  exact ID is ≤1% and |H| ≥ 3.95 (faithful: 38.5%, 2.65). At `lam>=1` the chip
  maximiser wins 4.3–4.6 chips/hand from the `mu=0` observer — the
  information-only rule is fully exploitable — and produces **41–55%
  misattribution**: the observer collapses to exactly one intent and it is
  the wrong one. Exact ID actually rises to ~11% there, because a profit
  maximiser plays *some* intents faithfully by coincidence.
* **Leduc, out_of_model:** at `lam=0.1` the subject is break-even (−0.07) with
  59% contradictions and 8.7% misattribution. Above `lam=1` the two modes
  converge on the same chip-maximising play.
* At break-even, in every game and in both misfit modes, the subject still
  holds exact ID at ≤1.5%. Concealment does not have to be
  bought; playing well delivers it.


## Task 3: cost-aware observer — how much identification does a chip buy?

Observer objective: minimise `E[final |intent set|] - mu * E[chips]`,
computed by the same exact lookahead (`V`), with the expectation over
hypotheses in the current set under the uniform prior. Parameterised the same
way as the misfit generator so the two are directly comparable. `mu=0` is the
information-only rule; `mu=1000` is effectively a chip maximiser among
information ties. Subject: faithful. `python3 sweep.py observer`

**Framing.** This moves the work *toward* goal recognition design, not away
from it. GRD (Keren, Gal & Karpas 2014) minimises worst-case distinctiveness
subject to not preventing agents from achieving their goals — it is a
constrained optimisation. `mu` is our version of that constraint: the
observer may not buy information at unbounded cost to its own goal.

### Kuhn
```
setting                        exact                     H                 sound                 misID                contra               deviate                 chips
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
passive             6.8% [6.6%,7.0%]      2.85 [2.84,2.86]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.001 [-0.011,+0.018]
mu=0             20.8% [19.9%,21.6%]      2.06 [2.04,2.08]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.411 [+0.375,+0.449]
mu=0.1           20.8% [19.9%,21.6%]      2.06 [2.04,2.08]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.411 [+0.375,+0.449]
mu=0.25          20.8% [19.9%,21.6%]      2.06 [2.04,2.08]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.411 [+0.375,+0.449]
mu=0.5           17.4% [16.4%,17.9%]      2.09 [2.07,2.12]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.513 [+0.473,+0.554]
mu=1             17.4% [16.4%,17.9%]      2.09 [2.07,2.12]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.513 [+0.473,+0.554]
mu=2             17.4% [16.4%,17.9%]      2.09 [2.07,2.12]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.513 [+0.473,+0.554]
mu=4             17.4% [16.4%,17.9%]      2.09 [2.07,2.12]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.513 [+0.473,+0.554]
mu=1000          17.4% [16.4%,17.9%]      2.09 [2.07,2.12]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.513 [+0.473,+0.554]
```

### Leduc
```
setting                        exact                     H                 sound                 misID                contra               deviate                 chips
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
passive          28.0% [27.4%,28.7%]      3.16 [3.12,3.23]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]-0.121 [-0.207,-0.072]
mu=0             38.5% [37.8%,39.4%]      2.65 [2.62,2.66]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]-0.509 [-0.539,-0.490]
mu=0.1           39.0% [38.4%,39.6%]      2.65 [2.63,2.66]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.147 [+0.078,+0.210]
mu=0.25          37.1% [36.8%,37.8%]      2.68 [2.67,2.70]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.307 [+0.232,+0.389]
mu=0.5           33.2% [32.8%,33.7%]      2.72 [2.71,2.74]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.386 [+0.325,+0.462]
mu=1             30.0% [29.6%,30.6%]      2.85 [2.85,2.86]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.652 [+0.586,+0.745]
mu=2             22.4% [22.1%,22.6%]      3.04 [3.03,3.06]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.792 [+0.738,+0.862]
mu=4             22.4% [22.1%,22.6%]      3.04 [3.03,3.06]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.792 [+0.738,+0.862]
mu=1000          22.4% [22.1%,22.6%]      3.04 [3.03,3.06]100.0% [100.0%,100.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]+0.792 [+0.738,+0.862]
```

### Cost-aware observer against the pure concealer (Leduc, `lam=0`)
in_model:
```
setting                        exact                     H                 sound                 misID                contra               deviate                 chips
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
mu=0                0.0% [0.0%,0.0%]      4.64 [4.63,4.65]   65.7% [64.3%,67.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   36.6% [36.4%,36.8%]+0.903 [+0.901,+0.907]
mu=0.25             0.0% [0.0%,0.0%]      4.67 [4.66,4.68]   66.2% [64.8%,67.5%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   35.8% [35.7%,36.0%]+0.903 [+0.901,+0.907]
mu=1                0.0% [0.0%,0.0%]      4.81 [4.80,4.82]   67.9% [66.4%,69.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   40.7% [40.4%,41.1%]+0.903 [+0.901,+0.907]
```
out_of_model:
```
setting                        exact                     H                 sound                 misID                contra               deviate                 chips
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
mu=0                1.5% [1.2%,1.8%]      1.64 [1.60,1.66]   23.3% [23.0%,23.9%]      6.4% [6.0%,6.7%]   56.7% [56.3%,57.5%]   51.9% [51.5%,52.7%]+0.433 [+0.279,+0.515]
mu=0.25             0.4% [0.3%,0.4%]      1.51 [1.50,1.53]   22.4% [22.1%,22.6%]      0.5% [0.4%,0.6%]   63.3% [62.7%,63.7%]   56.7% [56.1%,57.4%]-0.062 [-0.160,+0.005]
mu=1                0.0% [0.0%,0.0%]      0.93 [0.92,0.93]   13.2% [12.7%,13.6%]      0.0% [0.0%,0.0%]   79.5% [79.5%,79.7%]   48.1% [47.3%,48.7%]+0.350 [+0.272,+0.394]
```

**Reading.**

* **Kuhn:** nothing to buy. `mu<=0.25` is identical to `mu=0`; from `mu=0.5`
  up the observer trades 3.4 points of exact ID (20.8% → 17.4%) for +0.10
  chips/hand and then stops changing. The information rule was already
  nearly chip-optimal.
* **Leduc: yes, there is a chip-neutral setting that keeps the advantage.**
  `mu=0.1` turns −0.51/hand into **+0.15/hand** while exact ID goes from 38.5%
  to **39.0%** and |H| stays 2.65 — the whole loss came from probes that
  bought no information the observer could not get more cheaply. `mu=0.25`
  keeps 37.1% at +0.31. Beyond that the trade is real: `mu=1` gives 30.0% at
  +0.65, `mu>=2` gives 22.4% at +0.79 — still above passive (28.0%) on |H| but
  below it on exact ID, so at that point the observer is a chip maximiser
  that happens to learn things.
* **Against the concealer, `mu` is not a defence.** Against in_model the
  chip-aware observer wins the same +0.90 and identifies nothing at any `mu`.
  Against out_of_model it makes things worse for identification (contradictions
  rise to 80% at `mu=1`) but it does remove misattribution (6.4% → 0.5% →
  0.0%): a chip-aware observer probes less, so it commits to a single wrong
  intent less often. That is a reason to run `mu>0` even if the number one
  cares about is misID.


## Hardening the misattribution result

The out_of_model generator (Leduc, adaptive observer, `mu=lam=0`) makes the
observer collapse onto exactly one *wrong* intent. `python3 sweep.py harden`:

```
seed 1: misID 134/2000 = 6.7%   95% Wilson CI [5.7%, 7.9%]
seed 2: misID 128/2000 = 6.4%   95% Wilson CI [5.4%, 7.6%]
seed 3: misID 120/2000 = 6.0%   95% Wilson CI [5.0%, 7.1%]
pooled: misID 382/6000 = 6.4%   95% Wilson CI [5.8%, 7.0%]
pooled: contradiction 3403/6000 = 56.7%   95% Wilson CI [55.5%, 58.0%]

### Do contradiction and misattribution co-occur within a hand?
hands with both: 0  (structurally impossible: the hypothesis set only shrinks, so a hand that reaches the empty set ends empty; misID means it never did)

### Is the contradiction visible before the observer commits?
contradiction hands: 3403
  H became empty BEFORE at least one observer decision: 397 (11.7%)
  H became empty at the subject's last move (no observer decision left): 540
  H became empty only at showdown (play fit an intent; the card fit none):   2466 (72.5%)
  where H emptied (history index or showdown) -> count: 4:25, 6:912, showdown:2466
  observer chips/hand in those 397 hands (it currently just plays passive after the model is refuted): -0.612

### Misattributed hands: what did the observer see?
  declared give_up      -> observer concluded value_bet    x37
  declared value_bet    -> observer concluded represent    x34
  declared probe        -> observer concluded value_bet    x34
  declared pot_control  -> observer concluded probe        x32
  declared value_bet    -> observer concluded probe        x30
  declared represent    -> observer concluded bluff        x30
  declared pot_control  -> observer concluded value_bet    x29
  declared bluff        -> observer concluded value_bet    x29
  declared give_up      -> observer concluded probe        x28
  declared trap         -> observer concluded value_bet    x26
  declared trap         -> observer concluded probe        x26
  declared bluff        -> observer concluded probe        x25
  declared represent    -> observer concluded value_bet    x22
  most common histories:
    bet call board:J1 bet fold                                   x93
    bet call board:J0 bet fold                                   x84
    bet call board:K0 check check                                x77
    bet call board:K1 check check                                x64
    check check board:Q0 bet fold                                x25
```

* **The number holds up.** Pooled over 6000 hands, misID = 6.4%, 95% Wilson
  CI [5.8%, 7.0%]; per-seed 6.0–6.7%. Contradiction = 56.7% [55.5%, 58.0%].
  Note from Task 2 that misID is *far* higher (41–55%) against a
  chip-maximising subject than against the pure out_of_model generator; 6.4% is the
  conservative end.
* **Contradiction and misattribution never co-occur** in a hand, by
  construction: the hypothesis set only shrinks, so once empty it stays
  empty, and misID means it never emptied. They are two different failure
  modes of the same assumption (that the subject is faithful to some intent).
* **Is "no intent explains this" a usable signal? Mostly not, as things
  stand.** Of 3403 contradiction hands, only 397 (11.7%) emptied the set
  before an observer decision remained. 540 emptied at the subject's last
  move, and **2466 (72.5%) emptied only at showdown** — the betting was
  consistent with some intent, and it was the revealed card that fit none.
  So the exact-elimination observer detects deception mostly after the fact.
  Where it does see it early the plain observer just plays passive and loses
  −0.61/hand in those hands. **Task 4c acts on exactly these 397 hands** — and
  the "fold on refutation" rule this paragraph originally proposed turned out
  to be wrong when measured (−6.75/hand, worse than doing nothing). See Task
  4c, M1.
* **What the observer sees when it is wrong.** The commonest misattributed
  histories are `bet call board:J bet fold` and `bet call board:K check
  check` — the subject bets round 1 (looking like value_bet/bluff/probe/
  pot_control) and then plays round 2 in a way only one of them would. Every
  intent gets misread as `value_bet` or `probe` most often: those are the two
  policies that bet round 1 with the widest range of cards, so they are the
  easiest to be mistaken for. This is the Paris-neuron point in our own data —
  a bet looks like value only until you ask what else produces that bet.

## Task 4c: a deception-aware observer

`python3 sweep.py deception`. Two mechanisms. Each has a measurement that would
show it is *not* working, and one of them failed that measurement first time —
see M1.

### The structural fact that shapes the whole thing

Within a single hand, contradiction and misattribution are **disjoint by
construction**: the hypothesis set only shrinks, so a hand that reaches the
empty set ends empty, and misID means it never did. So **no within-hand rule
can convert a misattribution into an abstention** — in the hand that goes
wrong, the signal is not there to be read. Reporting "contradicted" instead of
collapsing is still the right thing to do, but on its own it converts exactly
zero misattributions.

The signal exists **across** hands. A contradiction is not evidence, it is
*proof* that the subject is not faithful to any intent — and "the subject is
faithful to some intent" is the premise every elimination conclusion rests on.
Once the observer has seen one, every later confident claim rests on a premise
it has already disproved. That is the deduction M2 acts on, and it uses only
what the observer can see (its own set going empty); it never touches the
declaration.

### M1: play for chips once the model is refuted

**The first version of this mechanism was wrong, and the measurement caught
it.** The rule suggested by the earlier hardening pass was *fold* on refutation
— "nothing left to learn, so stop paying for it." Measured on exactly the hands
that refute the model while an observer decision remains:

```
                               hands   chips/hand passive   chips/hand FOLD    delta
out_of_model lam=0               397               -0.612            -6.748   -6.136
```

Six chips a hand *worse*. The premise conflated two different things: refuting
the **intent model** says nothing about **card equity**. The observer still
holds the stronger hand in **45.3%** of those spots (62/137, seed 1), and
folding forfeits every one of them.

The corrected rule plays for chips with *no model at all* — exact enumeration
over the remaining subtree, uniform over the subject's possible cards and over
its future actions, because an empty hypothesis set means there is no model
left to plan against (`chip_value_no_model`):

```
                               hands   chips/hand passive   chips/hand no-model    delta
out_of_model lam=0               397               -0.612                +1.705   +2.317
out_of_model lam=2               608              -11.000                -7.000   +4.000
in_model lam=2                   608              -11.000                -7.000   +4.000
```

A losing spot becomes a winning one. The comparison is paired: M1 only changes
what the observer does *after* the set is empty, so the same hands qualify
under both rules (397 and 397, 608 and 608).

### M2: cross-hand abstention — the confident-wrong / known-unknown trade

`k` = report "contradicted" instead of a single intent once the subject has
refuted the model in `k` prior hands. `k=0` is the plain observer. Mean over
seeds 1–3 [min, max], **n = 6000** (2000 hands x 3 seeds).

```
subject                    k    exact ID (report)   misID (report)      abstain    chips/hand
---------------------------------------------------------------------------------------------
faithful (control)         0              38.5%             0.0%          0.0%        -0.509
faithful (control)         1              38.5%             0.0%          0.0%        -0.509
faithful (control)         5              38.5%             0.0%          0.0%        -0.509

out_of_model, lam=0        0               1.5%             6.4%         56.7%        +0.433
out_of_model, lam=0        1               0.0%             0.0%         64.6%        +0.587
out_of_model, lam=0        5               0.0%             0.0%         64.6%        +0.587

chip maximiser, lam=2      0              10.0%            46.6%         12.8%        -4.422
chip maximiser, lam=2      1               0.1%             0.1%         69.2%        -4.016
chip maximiser, lam=2      5               0.2%             0.7%         68.4%        -4.016

in_model, lam=2            0              10.0%            49.2%         10.1%        -4.503
in_model, lam=2            1               0.1%             0.5%         68.8%        -4.098
in_model, lam=2            5               0.3%             1.4%         67.7%        -4.098
```

**The trade, stated as a rate:**

| subject | misID removed | exact ID given up | confident-wrong removed per confident-right lost |
|---|---|---|---|
| `out_of_model`, `lam=0` | 6.4% → 0.0% | 1.5% → 0.0% | **4.3** |
| chip maximiser, `lam=2` | 46.6% → 0.7% | 10.0% → 0.2% | **4.7** |
| `in_model`, `lam=2` | 49.2% → 1.4% | 10.0% → 0.3% | **4.9** |

**How fast the flag fires** (hands until the subject first refutes the model):

```
faithful                  never (0 refutations in 6000 hands)
out_of_model lam=0        median 1,  mean 1.0
chip maximiser lam=2      median 6,  mean 6.3
in_model lam=2            median 6,  mean 16.7
```

**Reading.**

* **The control is the point.** A faithful subject never refutes the model, so
  the flag never fires, so **every faithful number is unchanged at every `k`**
  — 38.5% exact ID, 0.0% misID, −0.509 chips, identical to three decimal
  places. This is the falsifier for M2: if the trigger were reading anything
  it should not, a faithful number would move. None does. Abstention costs
  nothing against honest subjects.
* **The conversion is real and it is cheap.** Against every misfit setting the
  observer trades roughly **one correct confident claim for between four and
  five wrong ones removed**. Against the chip maximiser — the realistic
  opponent, and the one that produced the worst misattribution in Task 2 — it
  converts a 46.6% confident-wrong rate into a 0.7% one.
* **`k=1` is the right setting, and higher `k` is strictly worse.** misID
  creeps back up (0.1% → 0.7% at `k=5`) because every unflagged hand before the
  threshold is another chance to commit. One proof is enough; waiting for
  corroboration only buys mistakes.
* **What is left is honest ambiguity, not knowledge.** Abstention is ~65–69% of
  hands. The observer is not identifying intent under misfit — it is
  now *saying so* instead of guessing. That is the whole accountability claim
  and it should not be oversold as identification.
* **Limitations.** The abstention is coarse: it suppresses every singleton
  claim after refutation, correct ones included (that is exactly the 1.5–10%
  of exact ID given up). And it is a defence against subjects who *break* the
  model, not against behaviour that stays inside it — pure `in_model` misfit
  at `lam=0` never refutes, so the flag never fires. It needs no defence there
  only because that mode drives |H| *up* (4.64) rather than onto a wrong
  singleton: its misID is already 0.0%. A subject that could misattribute
  without ever contradicting would defeat M2, and nothing here rules one out.

## The consistent misattributor — M2's guarantee has a hole, and it is a large one

`python3 sweep.py misattribute`. The open limitation from the deception-aware
observer was that M2 abstains on **contradiction**, so it defends only against
behaviour that *leaves* the intent model. This asks the obvious next question:
can a subject stay strictly inside the model and still drive the observer to a
confident, wrong, single intent? **Yes. Almost always.**

### The constraint has an exact characterisation

"Never empty the hypothesis set" sounds like a loose constraint. It is not.

At showdown the observer filters its set down to the subject's revealed card,
so the set survives only if **some intent generates the subject's entire action
path with that card**. The set of never-contradicting behaviours is therefore
exactly

> { play intent *i*'s policy faithfully : *i* ∈ INTENTS }

and nothing else. A subject cannot mix two intents across a hand unless a third
intent happens to generate the mixture. Two consequences follow immediately:

1. If the subject plays *i\**'s policy, ***i\** can never be eliminated*** — it
   generated every action and it matches the revealed card. So *i\** is in the
   final set always, and **any singleton final set is necessarily exactly
   {*i\**}**. Enumerated over all 840 (card, intent, observer card, board)
   combinations: 0 eliminations, 0 singletons that are not *i\**.
2. So misattribution means: the observer pins *i\**, and *i\** is not what the
   subject declared.

**The entire attack is therefore: declare one intent, then play a different one
faithfully — the most pinnable one available for your card.**

### The numbers

Leduc, n = 6000 per row (2000 hands x seeds 1–3), mean [min, max]. `k` is M2's
abstention threshold.

```
condition      k(M2)   exact ID   |H| final        sound        misID      contra   chips/hand
passive            0       0.0%   1.16          0.0%   95.5% [94.9,96.2]     0.0%       -0.157
passive            1       0.0%   1.16          0.0%   95.5% [94.9,96.2]     0.0%       -0.157
random             0       0.0%   2.53   22.6% [21.6,23.1]   52.1% [50.5,53.8]   0.0%   -0.680
random             1       0.0%   2.53   22.6% [21.6,23.1]   52.1% [50.5,53.8]   0.0%   -0.680
adaptive           0       0.0%   1.05          0.0%   98.6% [98.0,99.0]     0.0%       -0.490
adaptive           1       0.0%   1.05          0.0%   98.6% [98.0,99.0]     0.0%       -0.490
```

Kuhn, where *every* behaviour is in-model so the constraint costs nothing:

```
passive            -       0.0%   2.22   21.4% [20.7,22.4]   25.5% [24.8,26.2]   0.0%   -0.194
random             -       0.0%   2.04   20.5% [20.0,21.1]   38.9% [38.6,39.5]   0.0%   -0.147
adaptive           -       0.0%   1.28          0.0%   79.2% [78.5,80.1]     0.0%       -0.060
```

The same adaptive observer, for scale:

```
faithful                  38.5%   2.65          100%          0.0%           0.0%       -0.509
misfit:in_model            0.0%   4.64   65.7%                0.0%           0.0%       +0.903
misfit:out_of_model        1.5%   1.64   23.3%                6.4%          56.7%       +0.433
```

### Verifications

These are checks that would show the construction is broken rather than strong,
and they are run, not assumed:

```
              n=6000  contradictions  exact ID  played intent eliminated  played==declared
passive                            0         0                         0                 0
random                             0         0                         0                 0
adaptive                           0         0                         0                 0

M2 on vs off, identical on every metric:  passive True   random True   adaptive True
```

Contradiction is **exactly** 0, not approximately 0 — as it must be, since that
is what "in-model" means. Exact ID is exactly 0 structurally: the played intent
always survives and is never the declared one, so the set can never be exactly
{declared}. And M2 is **bit-for-bit irrelevant**: no contradiction ever fires,
so the flag can never be raised.

### What the attack does at the table

Nothing clever. It plays the most *stereotyped* policy available for its card.

Pin probability — P(the observer's final set is exactly {i}) if the subject
plays *i*, adaptive observer:

```
card      value_bet    bluff    probe  give_up     trap  represent  pot_control
J             0.200    1.000    1.000    0.000    0.200      0.600        0.000
Q             0.000    0.000    1.000    0.000    0.800      0.400        0.000
K             1.000    0.500    0.000    0.000    0.900      0.600        0.000
```

Every card has at least one intent the observer pins **with probability 1.000**
— J: bluff or probe; Q: probe; K: value_bet. So whatever the subject declares,
an alternative that gets pinned with *certainty* is nearly always available.
Across all (card, declared) pairs the played intent is `probe` x14, `bluff` x12,
`value_bet` x12, `trap` x4.

In plain language: **the attack is to be maximally legible as the wrong thing.**
It is the opposite of hiding. The subject bets its J like a textbook bluff, or
its K like a textbook value bet, and the observer — which is *good* at this —
reads the policy correctly and reports an intent the subject never declared.

### Reading

* **M2's scope is now precisely bounded.** It defends against `out_of_model`
  behaviour and against nothing else. Against in-model misattribution it is not
  weak, it is *inert*: identical on every metric to the observer without it.
  The earlier limitation note said "a subject that could misattribute without
  ever contradicting would defeat M2, and nothing here rules one out." One
  exists, it is easy to build, and it wins 98.6% of hands.
* **Competence is the vulnerability.** Ranked by misID: adaptive 98.6% >
  passive 95.5% > random 52.1%. The observer that is *best* at collapsing the
  hypothesis set is the one most reliably collapsed onto the wrong answer. The
  random observer is the safest of the three, purely because it learns less.
  The same ordering holds in Kuhn (79.2% > 25.5%, with random at 38.9%).
* **A larger action space makes this worse, not better.** Leduc's 98.6% against
  Kuhn's 79.2%. More room to distinguish intents means intents are more
  distinguishable — which is exactly what makes the wrong one identifiable with
  certainty. Every improvement to worst-case distinctiveness is also an
  improvement to worst-case *mis*-distinctiveness.
* **This needs no deception, and that is the point.** Read through the misfit
  reframing, this construction is not an attacker: it is a subject **whose
  declared label and executed policy come apart** — the "picked the nearest
  option from a menu that did not fit" case, which is the single most likely
  thing to happen in a human pilot. The observer is not being lied to. It is
  being told the truth by someone whose vocabulary does not match their
  behaviour, and it responds with confident, precise, wrong certainty.
* **What would actually defend against it.** Nothing in the elimination
  framework, because the observer's conclusion is *correct* about the policy
  and wrong only about the label. Distinguishing "this person is executing
  probe" from "this person believes they are executing value_bet" requires
  evidence the action history does not contain. That is a limit of the
  behavioural approach, not of this implementation — and it is the strongest
  argument in the project for why the declaration has to be measured against
  behaviour rather than assumed to match it.

## LLM as proposer — architecture built, proposer NOT yet measured

`python3 proposer.py --backend fixture`. The architecture commitment: **the LLM
proposes, never decides, and is never in the trust path.** It does not eliminate
hypotheses, rank them, or select probes — those stay symbolic. Every proposal
either compiles and validates, or is rejected.

Pipeline: serialise the observed history → LLM proposes → **compile** to the
internal policy representation (a total mapping over all **36** decision cells:
2 rounds x 3 situations x 3 card ranks, with a board rank in round 2) →
**validate** (total? legal actions? consistent with what was observed?) →
classify as `ill_formed` / `inconsistent` / `duplicate` / `novel_valid`.

### Status: no rejection rate is reported here, deliberately

An earlier draft of this section carried rejection rates of 12.5% / 60% / 75%.
**Those have been removed.** They came from a fixture written by the author in
the same session that built the classifier — they measured the author, not a
sampled model, and reporting them as a property of LLMs would have been wrong.
No numbers will appear here until a real run happens.

What *is* established, and is a real (if modest) result:

* The compile → validate → classify pipeline is implemented and **validated
  against ground truth**: all 7 built-in Leduc intents compile to total,
  legal 36-cell tables, so proposals and incumbents are checked as like for like.
* The free-form compiler is **symbolic on purpose**. A second LLM pass would put
  the model back in the trust path, which the architecture forbids.
* The consistency test a proposal must pass is the *same* one the built-in
  intents face. The proposer gets no special treatment.

### The obvious objection, stated up front

Our intent space is already enumerable, so on *this* setup the LLM adds nothing
we could not hand-write. That is deliberate: it is the only setting where a
rejection rate can be measured against known ground truth. The ordering is
**measure the proposer where it can be verified, then deploy it where it is
needed** — and "where it is needed" is the repair case, where a contradiction
proves the vocabulary is incomplete.

### Ready for a real run

`--proposer anthropic|gemini|ollama`, key from the environment only, responses
cached to disk by prompt hash so re-runs are free and reproducible. Three bugs in
the previously-untested API path are fixed:

| bug | consequence | fix |
|---|---|---|
| `REPAIR_PROMPT` defined, never called | the repair case — the only scientifically interesting one — silently reported n = 0 | repair proposals are now generated against real contradiction histories drawn from a seeded `out_of_model` run |
| `max_tokens = 1500` | too small for a 36-cell JSON table; truncation registers as `ill_formed` and would have *inflated* the constrained rejection rate | raised to 4096, and truncated responses are flagged as `truncated` rather than silently counted as malformed |
| no retry handling | one HTTP failure discarded every call in the run | bounded retry with backoff, and every response cached as it arrives so a crash loses nothing |

Tasks 1–2 and `sweep.py verify` run with zero dependencies and no key;
`proposer.py` is never imported by the experiment path and uses only stdlib
`urllib`.

## Gridworld — second environment, and an honest GRD comparison

`python3 gridworld.py --map` &middot; `--bounds` &middot; `--witness` &middot; `python3 sweep.py grid`

> ### Map rebuilt 2026-09-14 — every gridworld number below is new
>
> **Why.** The first map was checked against Keren, Gal & Karpas (ICAPS 2014) and
> failed. GRD's design constraint is **cost preservation**, not reachability:
>
> > minimize<sub>A₋</sub> ( wcd(D<sub>A∖A₋</sub>), |A₋| ) subject to ∀G ∈ 𝒢,
> > C\*<sub>D</sub>(G) = C\*<sub>D∖A₋</sub>(G)
> >
> > — "we require the solution to preserve the original optimal solution length
> > of all goals"
>
> On the old map **every** closure lengthened some route, so a faithful GRD
> designer could remove nothing at all and the design arm was dead on arrival.
> The 95.8% vs 39.6% "separation" reported on 2026-09-09 was measured against a
> baseline with its hands tied. **That result is withdrawn.**
>
> **What changed.** The new map has redundant equal-length routes by
> construction: every gate row (3, 4, 5) lies between every start row (0–3) and
> every destination row (5–7), so vertical distance through *any* gate is
> identical. Closing one removes optimal paths **without** increasing the optimal
> cost to any destination — exactly the move GRD is built around. `legal_close`
> now implements cost preservation directly and is asserted in the gate.
>
> **Legal design space: 6 of 8 configurations** (all but {g0,g1} and all-closed),
> against **1 of 8** — the empty one — on the old map.

8x8 grid, a wall down column 3 with three gates, four starts, three
destinations, four routing rules. An intent is a **(destination, routing rule)**
pair — 12 of them. Episodes are **enumerated, not sampled**: every start x every
intent = **48 per cell**, exhaustively. Only the `random` observer varies with
the seed.

### The headline: the separation does not survive

**There is no formal separation on a genuine GRD instance.** Stated plainly
because the opposite was claimed here five days ago.

Machine-checked over all **18** legal configurations (6 layouts x 3 vantages) x
48 episodes, `python3 gridworld.py --bounds`:

```
best SINGLE fixed configuration   42/48 = 87.5%   <- what a designer actually picks
best per-start ORACLE designer    46/48 = 95.8%   <- told the start before choosing
pooled-trace upper bound          48/48 = 100.0%  <- agree-everywhere argument
--------------------------------------------------
online adaptive                   48/48 = 100.0%
```

Online adaptive **equals** the pooled bound rather than exceeding it. All four
numbers are pinned in `sweep.py verify`; the gate asserts `online <= pooled` and
deliberately **does not** assert a strict inequality any more, because that would
pin a claim the data does not support.

`python3 gridworld.py --witness` now searches for a witness pair and reports:

```
NO WITNESS EXISTS on this map.
Every intent is separated by SOME fixed configuration.
```

Zero pairs are indistinguishable under all 18 legal fixed configurations. On the
old map the witness was start (3,0), `A/direct` vs `A/open_field`; on a map where
design is actually allowed to act, no such pair remains.

> **Correction, 2026-09-14 (later): `wall_hug` and `open_field` were inverted.**
> `best` is a *minimum* over the tie-break key, so `wall_hug` (fewest free
> neighbours) had to key on `+openness` and `open_field` on `-openness`; they
> were the other way round, so each rule behaved as the other one's gloss. Found
> by the vocabulary-proposal harness, which checks that a rule's DSL spec
> reproduces its implementation — the mismatch was invisible to every numeric
> check because it is a pure relabelling of two of the twelve intents.
>
> Aggregates are unchanged for `passive` and `adaptive` and for all three
> bounds, as a relabelling must be. **The `random` rows did move** (online
> 54.2% → 50.7%, design 46.5% → 47.9%, and the misattribute rows likewise),
> because a stochastic observer's RNG stream is consumed in step with trajectory
> lengths, so relabelling shifts which random action lands where. The tables
> below are the corrected set.

### What survives: a heuristic gap, not a formal one

Faithful subject, n = 48 per cell, enumerated:

```
mode                observer   exact ID   |H| final   forced/ep   cost/ep
online probing      passive       37.5%       2.00        0.00     +0.000
online probing      random        50.7%       1.83        1.06    -12.924
online probing      adaptive     100.0%       1.00        1.12     -1.875

environment design  passive       37.5%       2.00        0.00     +0.000
environment design  random        47.9%       1.92        1.10     -2.472
environment design  adaptive      87.5%       1.12        2.15     -2.000
```

Online adaptive beats design adaptive **100% to 87.5%** — a 12.5-point gap. But
87.5% is *exactly* the best-single-fixed-configuration bound, so the design arm
is now provably optimal for its class and the gap is the honest price of
committing before you see anything, not an artifact.

**The design arm was also fixed.** It previously minimised expected |H| and held
the vantage at 1, scoring 39.6% where the best fixed layout scored 47.9% on the
old map. `choose_design` now maximises the same exact-ID objective the bound uses
and selects the vantage, so it lands on its own bound exactly. Comparing an
observer to a baseline scored on a different objective was our error, not a
property of design.

### wcd — speaking the literature's language, and where it diverges from ours

GRD's metric is **wcd**: the longest prefix of an optimal path before the agent's
*goal* becomes clear. Computed over all legal designs (`gridworld.py --bounds`):

```
wcd over every legal design:   [13]
```

**Gate choice does not move wcd at all on this map** — and neither do the two
*illegal* configurations, so this is not the constraint biting. All three
destinations sit in the right-hand column, so optimal paths to them diverge late
regardless of which gate is open.

Yet over the same designs, exact identification of **intents** swings from
**15/48 to 42/48**. That is the mismatch, concretely:

| | GRD's wcd | our exact ID |
|---|---|---|
| what it recognises | the **goal** (3 destinations) | the **intent** (12 destination x routing-rule pairs) |
| across legal designs | constant at 13 | 15/48 → 42/48 |

A design lever that is worthless for goal recognition is decisive for intent
recognition. The two are related but not the same quantity, and a design
minimising one need not minimise the other. Anyone reading our numbers as GRD
results should read this table first.

### The poker result still replicates

Subject declares one intent and faithfully plays another, chosen to be most
pinnable — it never leaves the model.

```
mode                observer   exact ID   |H| final   sound    misID
online probing      passive        0.0%       1.00     0.0%   100.0%
online probing      random         0.0%       1.47     4.2%    76.4%
online probing      adaptive       0.0%       1.00     0.0%   100.0%

environment design  passive        0.0%       1.00     0.0%   100.0%
environment design  random         0.0%       2.11    13.2%    28.5%
environment design  adaptive       0.0%       1.00     0.0%   100.0%
```

Adaptive is misattributed on **100%** of episodes in *both* modes, and random is
again the safest observer because it learns least. The misattribution finding is
untouched by the map rebuild — it is a property of exact elimination over
intents-as-policies, not of any particular environment.

### Environment self-checks (in the gate)

```
reachability   every destination from every start, every legal layout: 72/72
termination    longest episode 14 steps (cap 30); failed to arrive: 0
distinctness   12/12 intents behave distinctly
GRD constraint cost-preserving designs: 6 of 8
```

### Shared code, and what could not be shared

`core.py` holds the per-episode record schema, every metric computed from it,
seed aggregation, Wilson intervals and table rendering — both environments import
it. Not shared: the elimination step and the selection rule, each a few lines
over an environment-specific `policy()` and action set whose recursion must walk
that environment's own tree. `sweep.py verify` holds them honest instead.

## Soft elimination — the likelihood layer (blocking for human data)

`python3 soft.py --verify` &middot; `--sweep` &middot; `--pilot pilot.json`

Exact elimination assumes the subject is inside the model. Real people are not:
the 20-hand pilot scored soundness **9/20** with no deception involved. Under
hard elimination one move off the declared policy either removes the true intent
or empties the set, so human data scored that way measures the subject's
sloppiness, not the observer's capability.

    P(a | intent, card, node) = (1 - eps) * [policy says a] + eps / |legal(node)|

The card is **profiled out (max), not marginalised (sum)** — hard elimination
asks *"is there some card under which this intent explains what we saw"*, which
is an existential. Summing instead weights an intent by how many cards happen to
fit it; that broke the eps = 0 collapse on 1511 of 5400 episodes before the
falsifier caught it. Card facts stay hard; only behaviour is noisy.

### Metric definitions, chosen to collapse exactly at eps = 0

At eps = 0 every consistent hypothesis has likelihood 1 and every inconsistent
one 0, so the posterior is uniform over exactly the hard survivors.

| metric | soft definition | at eps = 0 |
|---|---|---|
| `\|H\| eff` | perplexity, exp(entropy) of the intent posterior | uniform over k → **k** |
| `exact ID` | MAP intent is declared **and** carries mass ≥ τ = 0.9 | mass 1/k ≥ 0.9 iff **k = 1** |
| `misID` | MAP carries mass ≥ τ and is not declared | same argument |
| `sound` | declared intent is in the 95% HPD set | HPD = full support → **survived** |
| `contra` | hard support empty — *no intent explains this without invoking noise* | unchanged |

`contra` deliberately keeps its hard definition at every eps: at eps > 0 nothing
is strictly impossible, so a posterior version would be vacuous, and "refuted
unless the subject slipped" is the quantity we actually want.

**The falsifier passes.** Every metric recomputed through the likelihood layer at
eps = 0, compared per episode against what the hard run recorded, across both
games, all three conditions, every subject type, seeds 1–3:

```
episodes checked:                  9000
episodes where ANY metric differs:    0
```

This now runs inside `sweep.py verify`.

### What noise does (Leduc, adaptive, n = 3000 per cell)

The subject really slips at rate `eps_true`; the observer scores at `eps_model`.

```
eps_true |      scored eps=0 (hard)        |   scored eps=eps_true (matched)
         |  exact  |H|eff   sound  contra  |  exact  |H|eff   sound  contra
    0.00 |  37.0%    2.66  100.0%    0.0%  |  37.0%    2.66  100.0%    0.0%
    0.02 |  36.1%    2.62   97.5%    0.4%  |  36.2%    2.81   98.0%    0.4%
    0.05 |  34.9%    2.57   93.6%    1.1%  |  20.1%    3.00   96.8%    1.1%
    0.10 |  32.6%    2.50   87.8%    2.1%  |   0.7%    3.31   96.8%    2.1%
    0.20 |  27.7%    2.32   75.2%    4.1%  |   0.0%    3.91   96.9%    4.1%
    0.35 |  23.0%    2.09   60.2%    7.2% |   0.0%    4.82   97.1%    7.2%
    0.50 |  18.1%    1.89   46.9%   10.0% |   0.0%    5.62   97.2%   10.0%
```

The `eps_true = 0` row reproduces itself in both columns, which is the sweep's
own falsifier (the replay is verified identical to the original hand in 500/500).

**Read the two columns against each other.** Hard scoring keeps *reporting*
confident identifications as the subject gets noisier — 18.1% exact ID even at
eps = 0.5 — while its soundness collapses to **46.9%**. It is confidently naming
an intent in a fifth of hands while having discarded the true one in half of
them. Matched scoring holds soundness at **~97% at every noise level** and pays
for it in honest uncertainty: exact ID goes to zero and |H| eff rises to 5.62.

That is the trade, and it is the right one for human data. Hard elimination does
not become *uncertain* under noise, it becomes *wrong*.

### Re-scoring the real pilot — 20 human hands

```
   eps    sound    exact   |H| eff   contra   mean mass on declared
  0.00    45.0%     5.0%      1.75     0.0%                   0.242
  0.05    70.0%     5.0%      2.16     0.0%                   0.241
  0.10   100.0%     0.0%      2.49     0.0%                   0.240
  0.20   100.0%     0.0%      3.08     0.0%                   0.237
  0.50   100.0%     0.0%      4.31     0.0%                   0.224
```

**How much of the 9/20 was brittleness? Almost none of it.** Soundness climbs
from 45% to 100% by eps = 0.1 — but look at the last column. The posterior mass
on the intent the person actually declared sits at **≈ 0.24 at every eps**, and
Kuhn's uniform prior over five intents is **0.20**. After a complete hand, the
declared intent carries barely more weight than it started with.

So softening fixes the *metric* — the observer stops claiming the truth was
eliminated — without making the truth any better supported. The signal in this
person's behaviour about their stated intent is close to absent. That is a much
more serious finding for the human pilot than brittleness would have been, and
it is the number to design the study around.

**Caveats, because n is tiny.** 20 hands, one person, and **16 of 20
declarations were `bluff`** — a badly unbalanced sample from a self-chosen menu.
This is suggestive, not conclusive, and it is an argument for assigning intents
rather than letting participants pick.

### Scope limit

The likelihood layer is a **scoring** layer. The observer's action selection
still uses hard elimination, so `--mu` and the probe rule are unchanged and every
committed number is untouched by construction. Posterior-driven *selection* —
choosing probes to minimise expected posterior entropy rather than expected set
size — is the natural next step and is not implemented.

## Vocabulary proposal — held-out routing-rule recovery: MEASURED

`python3 vocab.py --selftest` (no key) &middot;
`python3 run_recovery.py <provider> <model> <n> [style] [prompt-version]`

Hold one routing rule **out** of the observer's vocabulary. The subject uses it,
so the vocabulary is *provably* incomplete — the pilot situation under conditions
where we know exactly what is missing. Show the LLM 36 trajectories, ask for a
new rule, compile it to an executable policy, then:

> **Run the proposal on 36 episodes the LLM never saw. Recovery = predicts
> held-out behaviour exactly.**

Every rule is held out in turn. Responses are cached by
`(provider, model, prompt, nonce, max_tokens)`, so re-runs are free and
reproducible.

> **These numbers could not have been trusted a week ago.** The rule glosses
> shown to the LLM are the *corrected* ones — `wall_hug` and `open_field` were
> implemented backwards until 2026-09-14. Before that fix the experiment would
> have been asking a model to recover a rule whose description was wrong, and
> scoring it against behaviour that contradicted the words. `glosses.py` now
> asserts the glosses against behaviour in the gate.

### Headline: the untuned baseline

gpt-4o-mini, **n = 160 per style** (4 held-out rules x 40 draws).

```
prompt v1 (untuned)   ill_formed  inconsistent  duplicate  novel_valid   RECOVERED
constrained                 0.6%         98.1%       1.2%         0.0%   0/160 =  0.0%  [0.0%,  2.3%]
free-form                  45.0%         20.6%      20.0%        14.4%  23/160 = 14.4%  [9.8%, 20.6%]
```

**Constrained recovery is 0/160.** At n=20 that looked like it might be small
sample; at n=160 the interval is [0.0%, 2.3%] and it is a real result. Handing
the model the exact schema produced *nothing usable*, 160 times.

**Free-form at 14.4% [9.8%, 20.6%]** — and note this is *lower* than the 25%
[11.2%, 46.9%] the n=20 pilot suggested. The wide early interval was optimistic,
which is the argument for n=160.

### The pooled number hides most of the story

```
recovery by held-out rule (v1 free-form)      direct  wall_hug  open_field  evasive
                                               1/40      1/40        6/40    15/40
                                               2.5%      2.5%       15.0%    37.5%
```

Recovery is **concentrated in `evasive`**. Two of the four rules essentially never
recover. Reporting 14.4% alone would have implied a uniform capability that does
not exist.

One cause is ours, not the model's: **`direct` is unrecoverable by the free-form
path by construction.** Its true spec is the *empty* list, and `compile_prose`
requires at least one recognised criterion, so it can never emit it. Any `direct`
"recovery" in the free-form column is an artifact.

### Does the symbolic compiler explain the free-form advantage? Yes, measurably

```
gpt-4o-mini, all draws              CONSTRAINED    FREE-FORM
criteria the model asserted (mean)         4.00    n/a (prose)
criteria surviving compilation             4.00           1.25
spec-length distribution               {4: 160}   {1:67, 2:20, 3:1}
proposals discarded entirely                  1             72
```

True specs are length **0 or 1**. **160 of 160** constrained proposals used
exactly four criteria — the model read "at most 4" as a target — and the
compiler filtered *none* of them, because four valid criterion names are
syntactically perfect. The free-form path averages 1.25 because the compiler can
only emit what it recognises, and it discarded 72 proposals outright.

**The symbolic layer protects the LLM from itself — but only when the LLM is not
handed the schema.** Giving it the schema invites over-specification the
validator has no grounds to reject. That is a measured architecture claim, not
an inferred one.

### Cross-model: the 0% is a property of the model, not the task

gpt-5-mini, **n = 80 per style**, **prompt v1 — the same untuned prompt**
(verified: all 160 cached prompts are v1).

```
prompt v1 (untuned)   ill_formed  inconsistent  duplicate  novel_valid   RECOVERED
gpt-4o-mini  constr.        0.6%         98.1%       1.2%         0.0%   0/160 =  0.0%  [0.0%,  2.3%]
gpt-5-mini   constr.        0.0%         38.8%      23.8%        37.5%  30/80  = 37.5%  [27.7%, 48.5%]

gpt-4o-mini  free-form     45.0%         20.6%      20.0%        14.4%  23/160 = 14.4%  [9.8%, 20.6%]
gpt-5-mini   free-form     33.8%          8.8%      25.0%        32.5%  26/80  = 32.5%  [23.2%, 43.4%]
```

**The constrained 0% was gpt-4o-mini, not the task.** Handed the identical
schema and the identical prompt, gpt-5-mini recovers 37.5%. Whatever the
constrained path is asking for, it is askable — the weaker model simply could not
do it.

This also reframes the whole exercise: for the weak model, the schema was
*harmful* (0.0% constrained vs 14.4% free-form); for the stronger model it is
mildly *helpful* (37.5% vs 32.5%). The architecture claim above holds only in the
regime where the model cannot use the schema properly.

`gpt-5` (207s/call) and `gemini-3.5-flash` (100s/call) were **dropped**: neither
reaches n = 20 in usable wall time, and an n = 3 number cannot support a claim.
gpt-5-mini (61s/call) was chosen as the cross-model comparison for the best
information per unit of wall time.

### Tuning: one iteration, both numbers reported

Two failure modes were visible in v1, and v2 targets both: it says fewer criteria
are better and that most rules need exactly one; it states that the shortest-path
constraint is already enforced so goal-directed criteria explain nothing; and it
asks for the rule in words, naming the direction, before the JSON.

```
                        v1 (untuned)                 v2 (tuned)
constrained    0/160 =  0.0% [0.0%,  2.3%]   52/160 = 32.5% [25.7%, 40.1%]
free-form     23/160 = 14.4% [9.8%, 20.6%]   14/160 =  8.8% [5.3%, 14.2%]
```

**Both failure modes were prompt design, not model capability**, and the
mechanism is visible:

```
constrained, gpt-4o-mini        v1        v2
mean criteria per spec        4.00      1.00
length distribution       {4: 159}  {1: 160}
leads with a goal_* criterion  58%        1%
observer_dist lead = "max"    0/62     80/95
```

Over-specification went from universal to absent. Direction inversion went from
**0/62 correct to 80/95 (84%)**. The same model, with the same information,
produced a 32.5% recovery rate once the prompt stopped inviting both errors.

**Tuning helped constrained and HURT free-form** (14.4% → 8.8%), and the ordering
flipped: v1 had free-form ahead, v2 has constrained ahead. The v2 free-form
instructions push toward naming one distinguishing feature, which appears to
trade recall for a higher duplicate rate (46.2%).

**A trade-off v2 introduced, stated plainly:** by insisting on at least one
criterion, v2 made `direct` — whose true spec is empty — unreachable in the
constrained path too (0/40, all classified duplicate). v2 fixed two failure
modes and created a third. **One iteration only**; we stopped there deliberately,
because the eval is small and tuning until the number looks good is fitting to it.

### Prompting the weak model ≈ using the strong one

```
constrained recovery      gpt-4o-mini            gpt-5-mini
prompt v1 (untuned)    0/160 =  0.0% [0.0, 2.3]   30/80 = 37.5% [27.7, 48.5]
prompt v2 (tuned)     52/160 = 32.5% [25.7,40.1]  not run
```

Fixing the prompt took gpt-4o-mini from 0.0% to **32.5%**; switching to a
stronger model took it from 0.0% to **37.5%** with no prompt change. The
intervals overlap heavily. **Almost the entire gap between the two models on
this task was prompt design** — the weak model was not failing to reason about
routing rules, it was over-specifying and inverting directions because the
prompt invited both.

That is a cheerful result for anyone deploying this, and a cautionary one for
anyone benchmarking: the untuned v1 number would have ranked these two models as
incomparable (0.0% vs 37.5%) when one prompt change closes most of it.

### Graded score: does "recovered" being all-or-nothing understate it?

`python3 rescore.py` — re-scores every cached proposal; no API calls. Binary
recovery is kept unchanged so committed numbers stay comparable.

Agreement is measured by **teacher forcing**: at each state the subject actually
visited under the true rule, ask what the proposal would do there. Rolling the
proposal out instead would let one early mistake cascade into a different route
and score near zero — that measures divergence amplification, not agreement.

**The floor is the whole story.** Most steps have exactly one shortest option, so
every rule agrees on them for free:

```
held-out rule    empty spec, raw   empty spec, contested
direct                   100.0%                  100.0%     <- `direct` IS the empty spec
wall_hug                  85.7%                   62.5%
open_field                90.5%                   78.6%
evasive                   92.1%                   83.1%
```

The empty spec is "shortest path, no rule at all". Raw agreement cannot go below
~86%, so quoting it would turn *learned nothing* into *90% accurate*. Only
**contested** agreement — states with two or more shortest steps, where a routing
rule has any content — discriminates. Everything below is contested.

```
model / prompt / style          n    BINARY   contested mean   contested distribution
gpt-4o-mini v1 constrained    160      0.0%            19.7%   <25:110 25-50:36 50-75:11 75-90:2 100%:0
gpt-4o-mini v1 free-form      160     14.4%            64.2%   <25:10 25-50:14 50-75:28 75-90:9 90-:4 100%:23
gpt-4o-mini v2 constrained    160     32.5%            72.8%   <25:1 25-50:15 50-75:80 75-90:12 90-:0 100%:52
gpt-4o-mini v2 free-form      160      8.8%            55.8%   <25:13 25-50:41 50-75:42 75-90:13 90-:5 100%:14
gpt-5-mini  v1 constrained     80     37.5%            64.5%   <25:12 25-50:18 50-75:10 75-90:7 90-:1 100%:32
gpt-5-mini  v1 free-form       80     32.5%            74.0%   <25:0 25-50:17 50-75:2 75-90:4 90-:4 100%:26
```

**The answer is no — the graded metric does not rescue the component.** Two
things say so.

**1. The distribution is bimodal, not clustered near the top.** Proposals are
either *exactly right* or *substantially wrong*; the 90–99% band is almost empty
everywhere (0, 4, 0, 5, 1, 4 across the six conditions). gpt-5-mini free-form is
the clearest case: 17 proposals in 25–50%, 26 at exactly 100%, and 2 in between.
So the honest statement is **not** "recovers the gist but not the exact ordering".
It is closer to "gets it or does not".

**2. On two of the four rules, proposals are on average WORSE than proposing
nothing.** Against the no-rule floor:

```
                          gpt-4o-mini v2 constr.      gpt-5-mini v1 free-form
rule          floor       contested   vs floor        contested   vs floor
direct       100.0%           64.9%   -35.1           89.6%       -10.4
wall_hug      62.5%           64.3%   +1.8            31.2%       -31.3
open_field    78.6%           71.5%   -7.1            91.1%       +12.5
evasive       83.1%           90.7%   +7.6           100.0%       +16.9
```

`evasive` and `open_field` beat the floor. `direct` and `wall_hug` do not — a
proposal for them is, on average, *less* predictive of the subject's behaviour
than assuming no rule at all. Those are exactly the two rules with near-zero
binary recovery, so the graded view confirms the concentration rather than
softening it.

### Why two rules are dead: it is NOT an informational limit

The hypothesis was that ordering is unrecoverable — that several orderings of a
rule's criteria look identical on the shown episodes and diverge on unseen ones.
**It cannot apply as stated**: the true specs have length 0 or 1, so there is
only one ordering to get right. Tested the correct generalisation instead —
enumerate the whole DSL up to length 2 (**183 specs**) and ask how many are
indistinguishable from the truth on the 36 shown trajectories but diverge on the
36 unseen ones.

```
held-out rule   |true spec|   match SHOWN   of those, match UNSEEN   TRAPS
direct                   0             53                      53       0
wall_hug                 1              5                       5       0
open_field               1              6                       6       0
evasive                  1              5                       5       0
```

**Zero traps, for every rule.** Every spec that fits the training trajectories
also predicts the unseen ones. The 36 shown episodes *fully determine* the rule.

Pooling all 676 compiled proposals across every model, prompt and style:

```
compiled                                676
FITS the 36 trajectories it was SHOWN   150  = 22.2%
generalises to the 36 unseen            152  = 22.5%
fits shown BUT fails unseen               0
```

**Generalisation is free; fitting is the entire bottleneck.** 77.8% of compiled
proposals contradict trajectories the model was *literally shown*. And the
per-rule fit rate reproduces the recovery ordering exactly:

```
rule         compiled   fits shown   generalises
direct            157        10.2%         10.2%
wall_hug          178         5.6%          6.7%
open_field        171        31.6%         31.6%
evasive           170        41.2%         41.2%
```

**So the diagnosis is mundane, not structural.** Two specific causes:

* **`direct` is our framing artifact.** Its true answer is *"there is no further
  rule"* — the empty spec. The prompt asks the model to "propose ONE new routing
  rule that explains this behaviour", and v2 adds that "almost every rule needs
  exactly ONE" criterion. **The question presupposes a rule exists**, so "no
  rule" is not an available answer. The commonest proposals are
  `[openness max]` (x33), a four-criterion goal-directed spec (x27), and
  `[openness min]` (x17) — all contradicted by the shown data. This is fixable
  by letting the answer be empty, and we have not fixed it, because the tuning
  budget was spent.
* **`wall_hug` is confused with its neighbours.** The commonest proposal is
  `[observer_dist max]` — which *is* `evasive`, a rule still in the vocabulary
  (x48, classified duplicate) — followed by `[openness max]`, the exact opposite
  of the truth (x24). On this map, hugging walls often also increases distance
  from the observer, so the salient explanation ("avoiding the watcher") wins
  over the correct one. The shown data distinguishes them; the model does not
  check.

### The suggested connection to misattribution does NOT hold

The brief asked whether this links to confident misattribution — both being
cases where behaviour underdetermines what produced it. **The data says no, and
that is worth stating clearly rather than drawing a pleasing parallel.**

In misattribution, behaviour genuinely underdetermines the label: the observer is
right about the policy and no action history distinguishes "is executing X" from
"believes they are executing Y". That is an informational limit.

Here there is no informational limit at all — 0 traps, 0 fit-but-fail-to-
generalise. The evidence is sufficient and the proposer simply fails to fit it.
These are *different kinds of failure*, and merging them would have manufactured
a unified story the measurements do not support.

### LLM branch: CLOSED

**Verdict: the component does not work well enough to build on, and the reason
is now understood rather than guessed.**

What was measured, across 836 proposals from two models, two prompt styles and
one tuning iteration:

| | |
|---|---|
| best binary recovery | **37.5%** [27.7%, 48.5%] (gpt-5-mini, constrained, untuned) |
| best after tuning a weak model | 32.5% [25.7%, 40.1%] (gpt-4o-mini, constrained, v2) |
| proposals that fit the data they were shown | **22.2%** |
| proposals that fit-but-fail-to-generalise | **0** |
| rules beating the no-rule floor | **2 of 4** |

And the task was made as easy as we could honestly make it: ground truth is a
one-line tie-break rule *we designed to be recoverable*, shown 36 trajectories,
expressible in a 7-criterion vocabulary *we also designed*, with the answer
usually a single criterion.

**The failure is fitting, not generalising.** That is the one genuinely useful
thing this branch produced. Every proposal that fit the evidence also
generalised, so nothing here is about induction or sample size; 77.8% of
proposals simply contradict trajectories the model was handed.

Three findings worth keeping even though the component is shelved:

1. **The symbolic validator is load-bearing, but only in a specific regime.**
   For the weak model the schema was actively harmful (0.0% constrained vs 14.4%
   free-form) because it invited over-specification the validator had no grounds
   to reject — 160/160 proposals used exactly four criteria against true specs of
   length 0–1. For the stronger model the schema helps slightly. "The symbolic
   layer protects the LLM from itself" is true where the model cannot use the
   schema, and not a general property.
2. **Prompt design and model capability were near-substitutes here.** Fixing the
   prompt took gpt-4o-mini from 0.0% to 32.5%; switching model took it from 0.0%
   to 37.5% with no prompt change. Untuned numbers would have ranked the two
   models as incomparable.
3. **Graded scoring does not rescue a weak binary score.** The distribution is
   bimodal — proposals are exactly right or substantially wrong, with the 90–99%
   band nearly empty — and on two of four rules they are on average *worse than
   proposing no rule at all*.

**What we are not doing, and why.** Not tuning further: the eval has four
possible answers and n=80 per cell, so past one honest iteration improvement is
indistinguishable from fitting the test. Not building the repair case (propose a
*sixth* intent when a contradiction fires): it is strictly harder and has no
ground truth, so it could not be measured this way at all.

**Why this branch existed, restated.** The LLM was never the goal. It was one
candidate fix for a problem we have **not confirmed exists** — that our intent
vocabulary does not match real people — and that suspicion rests on a single
20-hand session in which the subject declared `bluff` 16 times out of 20. The
right next move is not a better proposer. It is to find out whether the problem
is real.

**Everything now waits on the pilot.**

## Gridworld human pilot — instrument built, NOT run

`python3 pilot.py --check` &middot; `--irb` &middot; `--dry-run` &middot; `--analyse`

Every number in this document comes from a subject that follows its declared
intent **by construction**. The one human trial we have put posterior mass on
the declared intent at **0.24 against a 0.20 prior**. Nothing here is validated
on people. This is the instrument for fixing that; it has not been run.

### What the poker pilot got wrong, and what changes

| poker pilot | this instrument |
|---|---|
| 20 episodes, 1 participant | 102 episodes x 2–3 participants |
| intents **self-chosen** — 16 of 20 were `bluff` | intents **assigned**, balanced round-robin, order shuffled per participant so intent is not confounded with fatigue |
| **soundness** led the report, and hit 100% under soft scoring while mass stayed at 0.24 | **posterior mass on the assigned intent vs the uniform prior** leads; soundness is logged and explicitly not the headline |
| rescoring needed the session re-run | every decision logged with full state — any eps, prior or metric recomputable offline |

### The design decision that matters most: ties are equiprobable

`gridworld.policy` returns one move by applying a fixed N/E/S/W order to
whatever the routing rule leaves tied. **That order is an implementation detail
no human will reproduce.** Scoring a person against it would manufacture misfit
that is our artifact and then report it as theirs. So the likelihood treats
every move the rule leaves open as equally likely:

```
P(a | intent) = (1 - eps) * [a in acceptable] / |acceptable|  +  eps / |legal|
```

`direct` is excluded from assignment by default for the same reason: "take any
shortest path" means the tie-break carries *all* of its content.

### The destination/rule split, wired through every metric

The specificity hypothesis was refuted and **this replaced it**. Confusions are
**52.6% right-rule-wrong-destination** and **38.9%** the reverse: the two halves
of an intent fail *independently*, so a single intent-level number is dominated
by whichever half is worse and hides the other.

Every metric is therefore reported three ways — joint, destination-only,
rule-only — not just the headline. Marginals are taken over the posterior, and
each is compared against its own prior (intent 1/9, destination 1/3, rule 1/3).

From a dry run (a rule-following subject, so these are instrument numbers, not
human data):

```
posterior mass on the assigned ...      eps=0.0   eps=0.1   eps=0.35   prior
  INTENT                                  0.602     0.585      0.523   0.111
  DESTINATION                             0.778     0.763      0.710   0.333
  RULE                                    0.740     0.734      0.711   0.333

MAP correct at eps = 0.1:   intent 61.1%   destination 66.7%   rule 83.3%
effective support:          |H| 2.53       |D| 1.70            |R| 1.51
```

The rule is recovered far better than the destination (83.3% vs 66.7%), and the
joint number (61.1%) sits below both — exactly the collapsing the split exists
to prevent. Had we reported only the joint figure we would have concluded the
instrument recovers intent poorly, when in fact it recovers *routing rules* well
and *destinations* less well, which are different problems with different fixes.

### Session transcript, for review before recruiting

`docs/pilot-session-transcript.txt` — a full dry-run session exactly as a
participant sees it, generated by

```
python3 pilot.py --dry-run --participant REVIEW --episodes 3
```

Three rounds rather than 102 so the **wording** can be reviewed. Every rule
sentence in it is asserted against actual behaviour by `glosses.py`, which runs
in the gate: a participant told one thing while the code does another would
manufacture exactly the misfit we are trying to measure.

### Instrument self-check (in the gate, no human involved)

```
1. Assignment balanced           102 episodes / 9 intents -> 11-12 each
2. Perfect vs random subject     perfect 0.558, random 0.127, prior 0.111 (eps=0.1)
3. Tie-indifferent rule-follower mean 0.667, worst 0.167, none at or below prior
4. Records round-trip            posterior sums to 1.000000 from the log alone
```

### A prediction the instrument makes before anyone is recruited

Check 3 surfaced something worth stating in advance: a subject obeying a
**permissive** rule is systematically attributed to a more **specific** one. If
the move they freely chose is the only move the specific rule allows, that rule
assigns it probability 1 while the permissive rule spreads mass across its tie
set. This is correct Bayesian behaviour — Occam's razor over the tie set — and it
means **permissive intents will be under-recovered**.

It is also the misattribution finding arriving a third time, now in the pilot
design rather than in a simulation. We are recording it as a pre-registered
expectation rather than discovering it afterwards.

### IRB

`python3 pilot.py --irb` prints the full note. In brief, and **not** as a legal
determination: 2–3 adults, ~25 minutes, no deception, no personal data beyond a
participant code, keystrokes in a game. Most likely **exempt** under 45 CFR
46.104 Category 3(i)(A) (benign behavioural intervention, adults, not readily
identifiable) — but exemption is a determination the IRB makes, not one a
researcher may self-certify, so a request still has to be filed and approved
before recruiting.

Prepare anyway: protocol, an information sheet with click-through consent rather
than a signed form, a data plan (participant codes only — the logger writes
exactly a code, timestamps and moves), and recruitment text noting any
compensation and the recruiter–participant relationship. Deception, audio/video,
identifiers or minors would each lose the exemption; none are in this design.

**Two or three participants is an instrument shakedown, not a study.** At n = 3
we can detect "mass sits at the prior" and nothing about a population.

## The specificity hypothesis — tested and refuted

`python3 specificity.py`

The pilot instrument shipped with a pre-registered prediction: a subject
following a **permissive** intent would be systematically attributed to a more
**specific** one, because the specific intent assigns probability 1 to a move the
permissive one spreads across its tie set. If it held, it would have been the
structural explanation for misattribution that this project has been missing, and
a bias with a known direction might be correctable.

**It does not hold.** Tested against data we already had, before anyone was
recruited.

> **Status: DEAD. Do not re-run this.** Recorded at this length precisely so
> that the next person to look at confident misattribution — including future us
> — does not spend a week rediscovering that it is not an Occam bias. What is
> still open is stated at the end of this section.

**Why it looked likely.** It had the two properties a good structural
explanation needs. It named a *mechanism* rather than a correlation — mass
spread over a tie set — and that mechanism is real arithmetic that does happen
in any likelihood over under-determined policies. And it predicted a *direction*,
which meant it could be corrected rather than merely lamented. It was also the
only candidate that promised to unify poker and gridworld, since both showed the
same misattribution ordering. Those three things together are why it was worth
pre-registering, and equally why it was worth testing before anyone was
recruited on the strength of it.

### Permissiveness, defined before looking

For an intent `i`, `perm(i)` = the mean, over reachable decision states, of the
number of actions the intent leaves open **before** any arbitrary tie-break.
`perm = 1` means the intent fully determines behaviour. Chosen over
action-distribution entropy because it is exactly the quantity the mechanism
names — mass is spread over `|acceptable|` — and because if it is 1 everywhere
the mechanism provably cannot operate, which makes the claim falsifiable.

### (a) The mechanism cannot operate in any committed run

```
                     elimination code   pilot likelihood
gridworld, 12 intents    1.000 (all)      1.127 – 1.431
Leduc, 7 intents         1.000 (all)      n/a
spread                       0.000              0.304
```

Every committed misattribution number — poker and gridworld alike — comes from
code where policies are **total and deterministic**. Every intent has `perm = 1`
exactly, there are no tie sets, and there is nothing for mass to be spread over.
**The hypothesis cannot explain the misattribution finding**, because the
mechanism it names is unavailable there.

### (b) What the committed misattributions do track — not much

Leduc, `out_of_model`, adaptive, 2000 hands, **134 misattributions**. This
subject optimises *concealment*, not pinnability, so any relationship found is
discovered rather than built in (the `misattribute` subject is excluded here
precisely because it *chooses* the most pinnable intent, which would be
circular).

```
named intent MORE a-priori pinnable than declared : 85/134 = 63.4%  95% CI [55.0%, 71.1%]
named intent LESS pinnable                        : 49/134 = 36.6%
```

Above chance, but weak — and **contradicted by its own detail**: the intent
wrongly named most often is `value_bet` (64 of 134), which is one of the *least*
pinnable intents (0.400 against `probe`'s 0.667). Pinnability is not the driver
either.

### (c) Where the mechanism *can* act, it still is not what is happening

Tested in the pilot likelihood, where tie sets are real: 360 episodes, subject
obeys its intent and breaks residual ties at random.

```
MAP correct 265/360 = 73.6%;  wrong 95

named intent MORE SPECIFIC than assigned   63/95 = 66.3%   <- the prediction
named intent MORE PERMISSIVE               32/95 = 33.7%
```

66.3% looks like support until you look at the size of the effect it is
attributing:

```
median |perm difference| in a confusion   0.014   (full range across intents: 0.084)

what the wrong attributions ACTUALLY differ in:
  same ROUTING RULE, wrong destination    50/95 = 52.6%
  same DESTINATION, wrong routing rule    37/95 = 38.9%
  both wrong                               8/95 =  8.4%
```

Confusions happen between intents of **essentially identical permissiveness**.
The commonest are `B/open_field → A/open_field` and `B/evasive → A/evasive`: the
routing rule is recovered correctly and the **destination** is wrong.
Permissiveness has nothing to say about that, and the 66.3% is a near-tie on a
variable that barely varies being read as a trend.

### (d) The pre-registration is revised, before recruiting

**No correction was implemented.** Task (c) of the brief said to test a
specificity prior *if the hypothesis held*. It does not, so building one would be
fitting a correction to a bias that is not there.

The revised prediction, now in `pilot.py`:

> **Destination confusion dominates routing-rule confusion.** Report destination
> accuracy and rule accuracy **separately**; a single intent-level number is
> dominated by the destination component and hides how well the rule was
> recovered.

That is actionable in a way the original was not: it changes what the pilot
reports, not just what we expect to see.

### What this costs the misattribution finding, and what is still open

It stays a negative result about our own method with **no structural
explanation** — we tested the best candidate we had and it failed. The honest
position is unchanged from the headline section: the observer is right about the
policy and wrong only about the label, and no action history separates those.
What we have now additionally ruled out is that the effect is an Occam bias over
tie sets.

**Still open, for whoever picks this up next.** Misattribution is robust across
two environments, a map rebuild and both interaction modes, and we cannot say
why. Candidates not yet tested, in the order we would try them:

1. **It is not one phenomenon.** The committed runs and the pilot likelihood may
   fail for unrelated reasons — the first has no tie sets at all, the second is
   dominated by destination confusion. Treating them as one effect may be the
   error.
2. **Structure of the intent space**, not of any intent: how much of the
   behaviour space each intent occupies, and which intents are near-neighbours
   under the observer's own partition.
3. **The declared/executed gap is irreducible** and there is nothing to explain
   — in which case the finding is a limit on behavioural inference and should be
   stated as a theorem attempt, not chased as a bug.

Anything tested here should be added to this list with its outcome, refuted
included.

## Descriptions vs behaviour — a check for the class of bug numbers cannot catch

`python3 glosses.py`, and in the gate.

`wall_hug` and `open_field` were implemented **backwards for five days**. Every
numeric check in the project passed over it, because swapping two labels is a
pure relabelling and changes no aggregate. It surfaced only when the vocabulary
experiment forced something to assert that a rule's *gloss* matches its
*behaviour*.

That is the shape of the recent failures in this project, and it is worth naming:
**the object-level code is solid; the errors live in the things that describe
it.** The guard's own config listing a stale section set. A bound labelled as
covering more than it did. Two rules named backwards. Numbers cannot catch any of
these, because a wrong description of right behaviour produces right numbers.

So every place a human-readable description sits next to executable behaviour,
the description is now turned into a machine-checkable predicate and asserted:

| where | shown to | checks |
|---|---|---|
| gridworld routing-rule glosses | the LLM, in the vocabulary prompt | 1101 real choices per rule |
| pilot participant instructions | **human subjects** | 576–590 accepted moves per rule |
| Leduc intent glosses | the LLM, in the poker proposer prompt | 7 checkable claims |

Each predicate is deliberately **weaker than the implementation** — it encodes
only what the sentence actually claims. A predicate that restated the code would
pass by construction and catch nothing. `give_up`'s gloss says "check and fold",
so the check is that its action set is a subset of `{check, fold}`; it says
nothing about *when*.

The Leduc checks additionally verify that the quoted fragment is still a
substring of `INTENT_GLOSS`, so editing the prose without editing the claim fails
rather than silently decoupling. That fired on first run — one claim was a
paraphrase rather than a quote.

**The checker was falsified against the real bug.** Re-introducing the exact
`wall_hug`/`open_field` inversion in memory produces:

```
wall_hug     1101 real choices checked -> VIOLATED in 546
open_field   1101 real choices checked -> VIOLATED in 546
```

A check that would not have caught the bug it was written for is not worth
having, so this is verified rather than assumed.

### `must_replace`

`glosses.must_replace(text, old, new, label)` raises when the anchor is absent.
Unanchored `str.replace` no-ops silently on a missed anchor, which is how the
document guard came to report 13/13 while checking a stale list — the edit that
was supposed to register a new section simply did nothing, and nothing said so.
Scripted edits to tracked files go through it.

## Commands

```
python3 kuhn_intent.py  --hands 2000 --condition all --subject both --seeds 1 2 3
python3 leduc_intent.py --hands 2000 --condition all --subject both --seeds 1 2 3
python3 leduc_intent.py --human --hands 10 --subject misfit --misfit-mode in_model --log pilot.json
python3 kuhn_intent.py  --hands 2000 --condition adaptive --subject misfit --lam 0.25 --observer lookahead
python3 leduc_intent.py --hands 2000 --condition adaptive --mu 0.1
python3 sweep.py verify       # gate: Kuhn greedy == lookahead, 36k hands per rule
python3 sweep.py deception    # task 4c: deception-aware observer
python3 sweep.py all          # gate + every table above, seeds 1-3
python3 leduc_intent.py --hands 2000 --condition all --subject both --seeds 1 2 3
```

Per-hand JSON records carry a `subject` tag (`faithful`, `misfit:in_model`,
`misfit:out_of_model`), the full action history, deviation count, and the
surviving intent set after every action.
