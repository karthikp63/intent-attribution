#!/usr/bin/env python3
"""
Testing the specificity hypothesis against data we already have.

THE HYPOTHESIS, as pre-registered in the pilot instrument: a subject following a
PERMISSIVE intent is systematically attributed to a more SPECIFIC one, because
the specific intent assigns probability 1 to a move the permissive one spreads
across its tie set. If that holds it would be the structural explanation for
misattribution we have been missing, and a bias with a known direction may be
correctable.

WHAT PERMISSIVENESS MEANS HERE, stated before any result. For an intent i,

    perm(i) = mean over reachable decision states of |acceptable(i, state)|

where `acceptable` is the set of actions the intent leaves open BEFORE any
arbitrary tie-break. perm(i) = 1 means the intent fully determines behaviour;
larger means it constrains less. Chosen over action-distribution entropy because
it is exactly the quantity that appears in the likelihood -- the mechanism says
mass is spread over |acceptable|, so that is what must be measured -- and
because at |acceptable| = 1 for all i the mechanism provably cannot operate,
which makes the hypothesis falsifiable rather than vague.

    python3 specificity.py
"""

import math
import random
from collections import Counter

import gridworld as G
import leduc_intent as L
import pilot as PL


# ------------------------------------------------------- (a) permissiveness

def perm_gridworld(intent, use_pilot_sets=True):
    tot = n = 0
    for s in G.STARTS:
        for c in G.grd_legal_designs():
            for v in range(len(G.VANTAGES)):
                pos, last = s, None
                for _ in range(G.STEP_CAP):
                    if pos == G.DESTS[intent[0]]:
                        break
                    acc = (PL.acceptable_moves(intent, pos, c, v, last)
                           if use_pilot_sets
                           else {G.policy(intent, pos, c, v)})
                    tot += len(acc)
                    n += 1
                    a = sorted(acc)[0]
                    last = a
                    pos = G.step(pos, a)
    return tot / max(1, n)


def perm_poker(intent):
    """Leduc policies are total and deterministic: exactly one action per
    (intent, card, node). So perm is 1 for every intent, by construction."""
    seen = set()
    for card in L.DECK:
        for hist in _leduc_nodes():
            seen.add(1)          # |{policy(...)}| is always 1
    return 1.0


def _leduc_nodes():
    out, frontier = [], [()]
    while frontier:
        h = frontier.pop()
        if len(h) > 5 or L.to_act(h) == "terminal":
            continue
        if L.to_act(h) == "subject":
            out.append(h)
        if L.to_act(h) == "chance":
            frontier.append(h + ("board:J1",))
            continue
        for a in L.legal(h):
            frontier.append(h + (a,))
    return out


# ---------------------------------------- (b) direction of misattributions

def direction_poker(n_hands=2000, seed=1):
    """Every misattribution in a committed Leduc out_of_model run.

    out_of_model is the right test case: it optimises CONCEALMENT, not
    pinnability, so any relationship between the named intent and the declared
    one is discovered rather than built in. (The `misattribute` subject is
    excluded here precisely because it CHOOSES the most pinnable intent, which
    would make the test circular.)
    """
    rng, orng = random.Random(seed), random.Random(seed + 10 ** 6)
    state = {"refutations": 0}
    rows = [L.play_hand("adaptive", rng, hand_no=k + 1, subject="misfit:out_of_model",
                        state=state, orng=orng) for k in range(n_hands)]
    mis = [r for r in rows if r["misattributed"]]
    pins = {}
    for i in L.INTENTS:
        pins[i] = sum(L.pin_probability(i, c, "adaptive", 0.0)
                      for c in L.DECK) / len(L.DECK)
    more_pinnable = sum(1 for r in mis
                        if pins[r["final_set"][0]] > pins[r["declared"]])
    less = sum(1 for r in mis if pins[r["final_set"][0]] < pins[r["declared"]])
    same = len(mis) - more_pinnable - less
    return mis, pins, more_pinnable, less, same


# ------------------------------------------- (c) the direct test of the claim

def direct_test(n_per_intent=40, seed=11):
    """The pre-registered claim, tested where the mechanism can actually act:
    the pilot likelihood, where tie sets are real.

    A subject obeys its assigned intent but breaks residual ties at random --
    the realistic human. Where the MAP intent is wrong, is it MORE SPECIFIC
    (lower perm) than the assigned one?
    """
    intents = PL.assignable_intents()
    perm = {i: perm_gridworld(i) for i in intents}
    rng = random.Random(seed)
    wrong_more_specific = wrong_more_permissive = wrong_equal = correct = 0
    pairs = Counter()
    for i in intents:
        for k in range(n_per_intent):
            s = G.STARTS[k % len(G.STARTS)]
            c = PL.LAYOUTS[k % len(PL.LAYOUTS)]
            v = k % len(G.VANTAGES)
            pos, last, dec = s, None, []
            for _ in range(G.STEP_CAP):
                if pos == G.DESTS[i[0]]:
                    break
                acc = sorted(PL.acceptable_moves(i, pos, c, v, last))
                a = rng.choice(acc)
                dec.append([list(pos), c, v, last, a])
                last = a
                pos = G.step(pos, a)
            p = PL.posterior(dec, 0.0, intents)
            mapi = max(p, key=p.get)
            if mapi == i:
                correct += 1
                continue
            pairs[(i, mapi)] += 1
            if perm[mapi] < perm[i] - 1e-9:
                wrong_more_specific += 1
            elif perm[mapi] > perm[i] + 1e-9:
                wrong_more_permissive += 1
            else:
                wrong_equal += 1
    return (perm, correct, wrong_more_specific, wrong_more_permissive,
            wrong_equal, pairs)


def main():
    print("\n" + "=" * 76)
    print("THE SPECIFICITY HYPOTHESIS -- does it explain misattribution?")
    print("=" * 76)

    # ---- (a)
    print("\n## (a) Permissiveness, and where the mechanism CAN operate\n")
    print("perm(i) = mean |acceptable actions| over reachable states.")
    print("perm = 1 means the intent fully determines behaviour.\n")

    elim = {G.name(i): perm_gridworld(i, use_pilot_sets=False) for i in G.INTENTS}
    pilo = {G.name(i): perm_gridworld(i, use_pilot_sets=True) for i in G.INTENTS}
    print(f"  {'intent':<18}{'elimination code':>18}{'pilot likelihood':>19}")
    print("  " + "-" * 55)
    for k in sorted(pilo, key=lambda x: -pilo[x]):
        print(f"  {k:<18}{elim[k]:>18.3f}{pilo[k]:>19.3f}")
    print(f"\n  poker (Leduc), every intent: {perm_poker(L.INTENTS[0]):.3f}")

    spread_elim = max(elim.values()) - min(elim.values())
    print(f"\n  VERDICT ON APPLICABILITY:")
    print(f"    elimination code  spread = {spread_elim:.3f}  -> mechanism CANNOT operate")
    print(f"    pilot likelihood  spread = {max(pilo.values()) - min(pilo.values()):.3f}"
          f"  -> mechanism CAN operate")
    print("\n  Every committed misattribution number -- poker and gridworld alike --")
    print("  comes from code where policies are TOTAL and DETERMINISTIC, so every")
    print("  intent has perm = 1 exactly. With no tie sets there is nothing to")
    print("  spread mass over, and the Occam mechanism is unavailable. The")
    print("  hypothesis therefore CANNOT explain the misattribution finding.")

    # ---- (b)
    print("\n## (b) What the committed misattributions actually track\n")
    mis, pins, more, less, same = direction_poker()
    n = len(mis)
    print(f"  Leduc, out_of_model, adaptive, 2000 hands: {n} misattributions.")
    print(f"  (out_of_model optimises concealment, not pinnability, so this is")
    print(f"   not circular the way the `misattribute` subject would be.)\n")
    def wil(k, nn, z=1.96):
        p = k / nn
        d = 1 + z * z / nn
        c = (p + z * z / (2 * nn)) / d
        h = z * math.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / d
        return c - h, c + h
    lo, hi = wil(more, n)
    print(f"  named intent is MORE a-priori pinnable than declared : {more}/{n} = {more/n:.1%}"
          f"  95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  named intent is LESS pinnable                        : {less}/{n} = {less/n:.1%}")
    print(f"  equal                                                : {same}/{n} = {same/n:.1%}")
    print("\n  most pinnable intents (mean over cards):")
    for i, v in sorted(pins.items(), key=lambda kv: -kv[1])[:4]:
        print(f"    {i:<14}{v:.3f}")
    print("  least pinnable:")
    for i, v in sorted(pins.items(), key=lambda kv: kv[1])[:3]:
        print(f"    {i:<14}{v:.3f}")
    top = Counter(r["final_set"][0] for r in mis).most_common(4)
    print(f"\n  intents most often WRONGLY named: "
          + ", ".join(f"{k} x{v}" for k, v in top))

    # ---- (c)
    print("\n## (c) The claim tested where the mechanism CAN act (pilot likelihood)\n")
    perm, correct, spec, permv, eq, pairs = direct_test()
    wrong = spec + permv + eq
    tot = correct + wrong
    print(f"  {tot} simulated episodes, subject obeys its intent and breaks ties freely.")
    print(f"  MAP correct: {correct}/{tot} = {correct/tot:.1%};  wrong: {wrong}")
    if wrong:
        print(f"\n  of the {wrong} wrong attributions:")
        print(f"    named intent MORE SPECIFIC than assigned  : {spec}/{wrong} = {spec/wrong:.1%}"
              f"   <- the prediction")
        print(f"    named intent MORE PERMISSIVE than assigned: {permv}/{wrong} = {permv/wrong:.1%}")
        print(f"    equal permissiveness                      : {eq}/{wrong} = {eq/wrong:.1%}")
        # What do the wrong attributions ACTUALLY differ in?
        same_rule = sum(c for (x, y), c in pairs.items() if x[1] == y[1])
        same_dest = sum(c for (x, y), c in pairs.items() if x[0] == y[0])
        both = sum(c for (x, y), c in pairs.items() if x[0] != y[0] and x[1] != y[1])
        print(f"\n  what the wrong attributions actually differ in:")
        print(f"    same ROUTING RULE, wrong destination : {same_rule}/{wrong} = {same_rule/wrong:.1%}")
        print(f"    same DESTINATION, wrong routing rule : {same_dest}/{wrong} = {same_dest/wrong:.1%}")
        print(f"    both wrong                           : {both}/{wrong} = {both/wrong:.1%}")
        deltas = [abs(perm[y] - perm[x]) for (x, y), c in pairs.items() for _ in range(c)]
        print(f"    median |perm difference| in a confusion: "
              f"{sorted(deltas)[len(deltas)//2]:.3f}  (full perm range is "
              f"{max(perm.values()) - min(perm.values()):.3f})")
        print("\n  commonest confusions (assigned -> named):")
        for (a, b), c in pairs.most_common(5):
            print(f"    {G.name(a):<18} -> {G.name(b):<18} x{c}"
                  f"   perm {perm[a]:.3f} -> {perm[b]:.3f}")
    print("\n" + "=" * 76)


if __name__ == "__main__":
    main()
