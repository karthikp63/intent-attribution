# Results

All runs: 2000 hands per cell, seed 1, same deals across conditions.
Subjects: **faithful** plays its declared policy; **adversarial** declares
honestly (ground truth unchanged), then picks each action to maximise the
observer's expected final intent-set size. The adversary knows the observer's
selection rule and condition but not its card (worst case, as in GRD).
Ties break toward the declared policy, so it deviates only when it gains.

## Kuhn (`kuhn_intent.py`, 5 intents, 10 hypotheses)

```
condition               exact ID  |H| final   sound  misID  reveal  deviate  chips/hand
passive/faithful           6.7%       2.84   100%   0.0%    100%     0.0%      -0.004
random/faithful            9.8%       2.58   100%   0.0%     63%     0.0%      +0.054
adaptive/faithful         21.1%       2.04   100%   0.0%     61%     0.0%      +0.409
passive/adversarial        0.0%       3.32    65%   0.0%    100%    34.5%      -0.009
random/adversarial         0.0%       3.00    61%   0.0%     51%    38.0%      +0.486
adaptive/adversarial       0.0%       2.66    52%   0.0%      0%    44.4%      +1.000
```

Degradation under adversarial play: exact ID 21.1% → 0%, |H| final 2.04 → 2.66,
soundness 100% → 52%. Adaptive still ends with the smallest set. The adversary's
best reply to the adaptive observer is check-then-fold every hand (reveal 0%):
the largest reachable set in Kuhn is the check-fold bucket {bluff, give_up,
value_bet}. It pays 1 chip/hand for that concealment — the adversary has no chip
objective, which is a modelling choice to revisit (a cost-aware adversary is the
natural next variant).

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
set** (0 of 72,000). This is why Kuhn has no `impersonate`/`refute` split: the
two adversaries differ only in how they score an empty final set, an outcome
Kuhn cannot produce. The distinction exists only in Leduc.

Reproduce:

```
python3 sweep.py verify
```

## Leduc (`leduc_intent.py`, 7 intents, 35 hypotheses, two betting rounds)

```
condition                       exact ID  |H| final   sound  misID  contra  reveal  deviate  chips/hand
passive/faithful                  27.4%       3.23   100%   0.0%    0.0%    100%     0.0%      -0.207
random/faithful                   20.3%       3.33   100%   0.0%    0.0%     44%     0.0%      -0.158
adaptive/faithful                 38.4%       2.66   100%   0.0%    0.0%     49%     0.0%      -0.490
passive/adversarial:impersonate     0.0%       4.22    62%   0.0%    0.0%    100%    23.7%      -0.036
random/adversarial:impersonate     0.0%       4.74    69%   0.0%    0.0%     26%    34.1%      +0.752
adaptive/adversarial:impersonate     0.0%       4.65    67%   0.0%    0.0%     19%    36.4%      +0.901
passive/adversarial:refute         0.0%       1.88    27%   0.0%   48.0%    100%    56.0%      -0.216
random/adversarial:refute          1.2%       1.86    27%   6.0%   51.2%     61%    39.8%      +0.525
adaptive/adversarial:refute        1.5%       1.60    23%   6.7%   57.5%     64%    52.7%      +0.279
```

Seeds 1–3, adaptive/faithful: 37.8–39.4% exact ID, |H| 2.62–2.66.

* **Not a Kuhn artifact.** Adaptive beats passive and random on both exact ID
  (38.4% vs 27.4% / 20.3%) and |H| final (2.66 vs 3.23 / 3.33). The observer's
  rule here is exact lookahead over the rest of the hand (in Kuhn that collapses
  to the one-step greedy rule).
* **The information trade now costs chips.** In Kuhn adaptive made money; in
  Leduc it loses 0.49/hand because probing is a 4-chip bet in round 2. Both are
  real; the selection rule ignores payoff by design, and this is what a
  cost-aware rule would need to fix.
* **Two adversaries.** `impersonate` must stay consistent with some intent
  (empty set scores 0 for it); `refute` may play like nobody (empty set scores
  as full concealment). They are qualitatively different opponents:
  impersonate drives |H| up (4.65) with 0% exact ID; refute drives the observer
  into contradiction 50–58% of the time and produces misattribution (6.7% of
  hands end on exactly one intent, the wrong one — worse than ambiguity).
  Which adversary matters is a question for the group; both are reported.
* **Soundness is now a measurement.** 100% by construction for faithful play;
  52–69% for impersonate, 21–27% for refute. This is the number a human pilot
  will fill in for real subjects.

Representative Leduc trace (declared bluff, subject K1, observer J0):

```
check → bet → call → board J1 → check → bet → call     final: [bluff, represent]
  after subject checks, live intents: bluff give_up pot_control represent trap value_bet
  observer check → expected final |intent set| = 3.060
  observer bet   → expected final |intent set| = 3.040   ← chosen
```


## Task 2: cost-aware adversary — how much concealment does a chip buy?

Adversary objective: maximise `E[final |intent set|] - lam * E[chips lost]`.
**Weighted, not a hard budget**, because the adversary is already an
expectation-max recursion over the tree: a weight folds into the terminal
value and every node stays a plain max, whereas a hard constraint on expected
loss needs a Lagrangian (i.e. this `lam`, found by search) or a constrained
search over mixed strategies. `lam=0` is the pure concealer from the first
tables; `lam=1000` is effectively a pure chip maximiser (concealment only
breaks ties, toward the declared policy). Observer: adaptive, `mu=0`.
Cells: mean over seeds 1–3 [min, max], 2000 hands each.
`python3 sweep.py adversary`

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

### Leduc, impersonate
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

### Leduc, refute
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
  hand for 0% exact ID. At `lam=0.25` the adversary is break-even (−0.008)
  and exact ID is still 0%, |H| 2.50 vs 2.06 faithful. At `lam>=1` the
  adversary is a pure chip maximiser: it *beats* the adaptive observer
  (−0.33/hand for the observer, which made +0.41 against a faithful subject),
  and exact ID is still 0%. The observer's identification advantage in Kuhn
  existed only because the subject was faithful; a player who simply plays to
  win conceals intent as a side effect. "Your adversary is unrealistic" does
  not rescue the observer — the realistic one is worse for it.
* **Leduc, impersonate: break-even still hurts.** Break-even sits between
  `lam=0.25` (observer +0.57) and `lam=0.5` (observer −0.74); on both sides
  exact ID is ≤1% and |H| ≥ 3.95 (faithful: 38.5%, 2.65). At `lam>=1` the chip
  maximiser wins 4.3–4.6 chips/hand from the `mu=0` observer — the
  information-only rule is fully exploitable — and produces **41–55%
  misattribution**: the observer collapses to exactly one intent and it is
  the wrong one. Exact ID actually rises to ~11% there, because a profit
  maximiser plays *some* intents faithfully by coincidence.
* **Leduc, refute:** at `lam=0.1` the adversary is break-even (−0.07) with
  59% contradictions and 8.7% misattribution. Above `lam=1` the refute and
  impersonate adversaries converge on the same chip-maximising play.
* At break-even, in every game and against both adversary types, the
  adversary still holds exact ID at ≤1.5%. Concealment does not have to be
  bought; playing well delivers it.


## Task 3: cost-aware observer — how much identification does a chip buy?

Observer objective: minimise `E[final |intent set|] - mu * E[chips]`,
computed by the same exact lookahead (`V`), with the expectation over
hypotheses in the current set under the uniform prior. Parameterised the same
way as the adversary so the two are directly comparable. `mu=0` is the
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
Impersonate:
```
setting                        exact                     H                 sound                 misID                contra               deviate                 chips
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
mu=0                0.0% [0.0%,0.0%]      4.64 [4.63,4.65]   65.7% [64.3%,67.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   36.6% [36.4%,36.8%]+0.903 [+0.901,+0.907]
mu=0.25             0.0% [0.0%,0.0%]      4.67 [4.66,4.68]   66.2% [64.8%,67.5%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   35.8% [35.7%,36.0%]+0.903 [+0.901,+0.907]
mu=1                0.0% [0.0%,0.0%]      4.81 [4.80,4.82]   67.9% [66.4%,69.0%]      0.0% [0.0%,0.0%]      0.0% [0.0%,0.0%]   40.7% [40.4%,41.1%]+0.903 [+0.901,+0.907]
```
Refute:
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
* **Against the concealer, `mu` is not a defence.** Against impersonate the
  chip-aware observer wins the same +0.90 and identifies nothing at any `mu`.
  Against refute it makes things worse for identification (contradictions
  rise to 80% at `mu=1`) but it does remove misattribution (6.4% → 0.5% →
  0.0%): a chip-aware observer probes less, so it commits to a single wrong
  intent less often. That is a reason to run `mu>0` even if the number one
  cares about is misID.


## Hardening the misattribution result

The refute adversary (Leduc, adaptive observer, `mu=lam=0`) makes the
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
  chip-maximising subject than against the pure refuter; 6.4% is the
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
  Where it does see it early it currently just plays passive and loses
  −0.61/hand in those hands; a "fold on refutation" rule is the obvious
  cheap improvement, and it would be measurable on exactly these 397 hands.
* **What the observer sees when it is wrong.** The commonest misattributed
  histories are `bet call board:J bet fold` and `bet call board:K check
  check` — the subject bets round 1 (looking like value_bet/bluff/probe/
  pot_control) and then plays round 2 in a way only one of them would. Every
  intent gets misread as `value_bet` or `probe` most often: those are the two
  policies that bet round 1 with the widest range of cards, so they are the
  easiest to impersonate. This is the Paris-neuron point in our own data —
  a bet looks like value only until you ask what else produces that bet.

## Commands

```
python3 kuhn_intent.py  --hands 2000 --condition all --subject both --seed 1
python3 leduc_intent.py --hands 2000 --condition all --subject both --seed 1
python3 leduc_intent.py --human --hands 10 --subject adversarial --adversary impersonate --log pilot.json
python3 kuhn_intent.py  --hands 2000 --condition adaptive --subject adversarial --lam 0.25 --observer lookahead
python3 leduc_intent.py --hands 2000 --condition adaptive --mu 0.1
python3 sweep.py all          # ~1 min; every table above, seeds 1-3
```

Per-hand JSON records carry a `subject` tag (`faithful`, `adversarial:impersonate`,
`adversarial:refute`), the full action history, deviation count, and the
surviving intent set after every action.
