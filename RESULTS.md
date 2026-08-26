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

### Greedy vs. lookahead observer (sanity check)

Kuhn's original adaptive rule is one-step greedy (`expected_posterior_size`);
Leduc's is exact lookahead over the rest of the hand. `kuhn_intent.py` now has
an independent history-based implementation of the lookahead rule
(`--observer lookahead`, same code shape as Leduc's `V`). Per-hand JSON logs
were diffed record by record: seeds 1-3, all three conditions, both subject
types, 36,000 hands -- **0 differing records**. The Leduc observer is a strict
generalisation of the Kuhn one. Reproduce:

```
python3 kuhn_intent.py --hands 2000 --condition all --subject both --seed 1 --observer greedy    --log g.json
python3 kuhn_intent.py --hands 2000 --condition all --subject both --seed 1 --observer lookahead --log l.json
# then compare g.json and l.json ignoring the "observer" tag
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

## Commands

```
python3 kuhn_intent.py  --hands 2000 --condition all --subject both --seed 1
python3 leduc_intent.py --hands 2000 --condition all --subject both --seed 1
python3 leduc_intent.py --human --hands 10 --subject adversarial --adversary impersonate --log pilot.json
```

Per-hand JSON records carry a `subject` tag (`faithful`, `adversarial:impersonate`,
`adversarial:refute`), the full action history, deviation count, and the
surviving intent set after every action.
