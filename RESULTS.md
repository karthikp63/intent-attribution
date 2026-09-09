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

## Task 3: LLM as proposer — the rejection rate

`python3 proposer.py --backend fixture`. The architecture commitment: **the LLM
proposes, never decides, and is never in the trust path.** It does not
eliminate hypotheses, rank them, or select probes — those stay symbolic. Every
proposal either compiles and validates, or is rejected.

Pipeline: serialise the observed history → LLM proposes → **compile** to the
internal policy representation (a total mapping over all **36** decision cells:
2 rounds x 3 situations x 3 card ranks, with a board rank in round 2) →
**validate** (total? legal actions? consistent with what was observed?) →
classify. All 7 built-in intents compile to total, legal 36-cell tables, so
proposals and incumbents are checked as like for like.

> **Read this before the numbers.** The intent space here is already
> enumerable, so on *this* setup the LLM adds nothing we could not hand-write.
> That is deliberate: it is the only setting where the rejection rate can be
> measured against known ground truth. The ordering is **measure the proposer
> where it can be verified, then deploy it where it is needed** — and "where it
> is needed" is the repair case at the end of this section.

### Provenance — and a bias risk to state plainly

The proposals were generated by **Claude Opus 5** inside a Claude Code session
on 2026-09-09, responding to the verbatim prompts in `proposer.py`, recorded in
`fixtures/proposals.json`, and **committed before the classifier was run** so
the rate could not be tuned by editing them afterwards.

They were nonetheless *self-generated by an author who knew a measurement would
follow*, which biases toward care. **These numbers should be replicated with
`--backend api` before being reported as a property of LLMs in general**, and
n is small (8 / 10 / 4). The API path is implemented (stdlib `urllib`, needs
`ANTHROPIC_API_KEY`) and writes its own fixture.

A second caveat in the other direction: the free-form compiler is a hand-written
symbolic parser, and its first version had real bugs — it did not recognise
"round one" (only "round 1"), and first-verb-wins made *"facing a bet I raise"*
compile as *bet*. Those were fixed before reporting, along with two generic
gaps (the determiner "any" in condition phrases; comma-joined independent
clauses). It was **not** tuned per proposal, and the remaining failures are
listed individually below so a reader can judge which are the prose's fault and
which are the parser's strictness.

### CONSTRAINED — exact schema, model fills a 36-cell table (n = 8)

```
category          count     rate        95% Wilson CI
ill_formed            0     0.0%   [  0.0%,  32.4%]
inconsistent          0     0.0%   [  0.0%,  32.4%]
duplicate             1    12.5%   [  2.2%,  47.1%]
novel_valid           7    87.5%   [ 52.9%,  97.8%]
REJECTED              1    12.5%   [  2.2%,  47.1%]
```

The single rejection is `fold_everything`, semantically **identical to the
built-in `give_up`** in all 36 cells. Nothing was ill-formed. As anticipated,
this strategy is reliable and close to a lookup: the model is filling in a table
we designed, and barely reasoning.

### FREE-FORM — prose, compiled by a deterministic parser (n = 10)

The compiler is symbolic **on purpose**: a second LLM pass would put the model
back in the trust path, which the architecture forbids. So the compile-failure
rate measures how far free-form description sits from the formal
representation — which is the number we actually want.

```
category          count     rate        95% Wilson CI
ill_formed            6    60.0%   [ 31.3%,  83.2%]
inconsistent          0     0.0%   [  0.0%,  27.8%]
duplicate             0     0.0%   [  0.0%,  27.8%]
novel_valid           4    40.0%   [ 16.8%,  68.7%]
REJECTED              6    60.0%   [ 31.3%,  83.2%]
```

**The failure distribution is bimodal, and that is the finding.** Cells left
undetermined, out of 36:

```
board_paranoid          1  |
min_defence             2  ||
pot_odds                6  ||||||
escalate                9  |||||||||
read_dependent         24  ||||||||||||||||||||||||
mirror                 36  ||||||||||||||||||||||||||||||||||||
```

Two different things are being called "ill-formed":

* **A policy with a gap** (1–9 cells). `board_paranoid` specifies 35 of 36 cells
  and forgets what to do opening with a K. `min_defence` reasons carefully about
  defence frequencies and never says what to do when opening with J or Q. These
  are recognisably intents; they are just not *total*, and totality is exactly
  what makes an intent a counterfactual object rather than a description.
* **Not a policy at all** (24–36 cells). `mirror` — "whatever my opponent does,
  I do the same back" — determines **zero** of 36 cells: it is a function of the
  opponent's action, not of (situation, card, board), so it is not the right
  *type* of object. `read_dependent` — "I watch how they have been playing over
  the session and adapt" — conditions on session history that is not in the
  state space at all.

The second kind is the more damning, and it is not a formatting failure. Asked
for a complete policy, a capable model produced fluent, plausible, poker-literate
prose that **cannot be a policy in this game** — and it produced it in the same
register as the proposals that worked. Nothing in the text signals which is
which. Only compilation does.

### REPAIR — propose a new intent to explain a contradiction (n = 4)

The case where proposal earns its place, and the one that connects to the misfit
reframing: the observed play is explained by **no** intent in the vocabulary, so
the vocabulary is *by definition* incomplete. The observations are real
contradiction histories from `misfit:out_of_model` at seed 1 (1149 of 2000 hands
contradict; these are among the most common).

Here `novel_valid` is a strictly harder bar: the proposal must compile to a
total policy **and** actually generate the observed actions with the revealed
card.

```
category          count     rate        95% Wilson CI
ill_formed            3    75.0%   [ 30.1%,  95.4%]
inconsistent          0     0.0%   [  0.0%,  49.0%]
duplicate             0     0.0%   [  0.0%,  49.0%]
novel_valid           1    25.0%   [  4.6%,  69.9%]
REJECTED              3    75.0%   [ 30.1%,  95.4%]

COVERAGE: 1/4 proposed intents actually explain the behaviour that refuted
          the model.   cells undetermined: relentless 3, stab_then_catch 3,
          never_fold_second 9
```

The one that works, `double_barrel` — *"bet round 1 with any hand and bet round
2 with any hand as well, regardless of what the board brings"* — is precisely the
gap in the vocabulary. The most common contradiction at seed 1 is a subject
betting Q in round 1 and betting again on a Q board: `probe` is the only
built-in intent that bets round 1 with a Q, and `probe` checks in round 2. No
intent sustains aggression across both rounds. The LLM found that, in plain
language, from the serialised history.

Two of the six authored repair observations turned out not to be terminal
contradicting histories and were **excluded** rather than quietly reshaped,
leaving n = 4. That is itself a small datum about hand-authored fixtures.

### Reading

* **The rejection rate is real and it is high where it matters.** 12.5%
  constrained, 60% free-form, 75% repair. The rate rises exactly as the task
  moves from filling in our schema toward doing something we could not do
  ourselves.
* **Constrained buys reliability by removing the reasoning.** 0% ill-formed —
  and 1 of 8 proposals was an existing intent under a new name. If the schema is
  given, the model is a table-filler; that is safe and nearly pointless.
* **Free-form is where the interesting failure lives.** The bimodal
  distribution says the useful question is not "did it compile" but "was it the
  right kind of object at all". A proposal missing 1 cell is a repairable
  intent. `mirror` is a category error dressed in fluent poker prose.
* **This is the argument for the architecture, in one number.** 60–75% of
  proposals do not survive the symbolic layer, and **none** of them announce
  that in their text. If the LLM were allowed to eliminate hypotheses or select
  probes directly, those rejections would instead be silent corruptions of the
  hypothesis set. The validator is not a formality; it is doing most of the work.
* **What to do next.** Replicate via `--backend api` for an unbiased rate at
  larger n; then the repair loop is the deployment case — run it live whenever a
  contradiction fires, and measure coverage over many hands rather than 4.

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
