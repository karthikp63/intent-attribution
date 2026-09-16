#!/usr/bin/env python3
"""
Assert that DESCRIPTIONS match BEHAVIOUR.

WHY THIS FILE EXISTS. `wall_hug` and `open_field` were implemented backwards for
five days. Every numeric check in the project passed over it, because swapping
two labels is a pure relabelling and changes no aggregate. It surfaced only when
the vocabulary experiment forced something to assert that a rule's *gloss*
matches its *behaviour*.

That is the shape of our recent failures: the object-level code is solid, and the
errors live in the things that DESCRIBE it -- the guard's own config, a bound's
label, a rule's name. Numbers cannot catch those, because a wrong description of
right behaviour produces right numbers.

So: anywhere a human-readable description sits next to executable behaviour, the
description is turned into a machine-checkable predicate and asserted. Three
places currently qualify, and each is a place a human or an LLM is shown the
words and expected to rely on them:

  1. gridworld routing-rule glosses  -> shown to the LLM in the vocabulary prompt
  2. pilot participant instructions  -> shown to human subjects
  3. Leduc intent glosses            -> shown in the poker proposer prompt

A predicate here is deliberately WEAKER than the implementation: it encodes only
what the sentence actually claims. If it restated the code it would pass by
construction and catch nothing.

    python3 glosses.py
"""

import gridworld as G
import leduc_intent as L
import pilot as PL


def must_replace(text, old, new, label=""):
    """String replacement that FAILS LOUDLY when the anchor does not match.

    Unanchored `str.replace` no-ops silently on a missed anchor. That is how the
    document guard came to report 13/13 while checking a stale list: the edit
    that was supposed to register a new section simply did nothing, and nothing
    said so. Use this for any scripted edit to a tracked file.
    """
    if old not in text:
        raise AssertionError(f"anchor not found{' for ' + label if label else ''}: "
                             f"{old[:80]!r}")
    return text.replace(old, new)


# ------------------------------------------- 1. gridworld routing-rule glosses
#
# The claim each gloss makes, as a predicate over the move the rule picks versus
# the moves it passed over. `openness(cell)` is how many neighbours are free.

def _progressing(intent, pos, closed):
    dest = G.DESTS[intent[0]]
    d = G.dist_field(dest, closed)
    here = d.get(pos)
    if pos == dest or here is None:
        return []
    return [(m, n) for m, n in G.neighbours(pos, closed)
            if d.get(n, 1 << 20) == here - 1]


GRID_CLAIMS = {
    # rule: (the sentence, predicate(chosen_cell, alternatives, closed, vantage))
    "wall_hug": (
        "keep to cells with the MOST walls around them",
        lambda c, alts, cl, v: all(G.openness(c, cl) <= G.openness(a, cl) for a in alts)),
    "open_field": (
        "keep to the MOST OPEN cells",
        lambda c, alts, cl, v: all(G.openness(c, cl) >= G.openness(a, cl) for a in alts)),
    "evasive": (
        "maximise distance from the observer",
        lambda c, alts, cl, v: all(
            abs(c[0] - G.VANTAGES[v][0]) + abs(c[1] - G.VANTAGES[v][1]) >=
            abs(a[0] - G.VANTAGES[v][0]) + abs(a[1] - G.VANTAGES[v][1]) for a in alts)),
    "direct": (
        "any shortest path; no further preference",
        lambda c, alts, cl, v: True),
}


def check_gridworld():
    print("\n1. gridworld routing-rule glosses vs. policy()\n")
    ok = True
    for rule, (sentence, claim) in GRID_CLAIMS.items():
        checked = bad = 0
        for dest in sorted(G.DESTS):
            for closed in G.grd_legal_designs():
                for v in range(len(G.VANTAGES)):
                    for r in range(G.R):
                        for c in range(G.C):
                            pos = (r, c)
                            if G.blocked(pos, closed):
                                continue
                            steps = _progressing((dest, rule), pos, closed)
                            if len(steps) < 2:
                                continue      # no choice: the gloss claims nothing
                            chosen = G.policy((dest, rule), pos, closed, v)
                            cell = dict(steps).get(chosen)
                            if cell is None:
                                bad += 1
                                continue
                            alts = [n for m, n in steps if m != chosen]
                            checked += 1
                            if not claim(cell, alts, closed, v):
                                bad += 1
        status = "OK" if not bad else f"VIOLATED in {bad}"
        print(f"   {rule:<12} \"{sentence}\"")
        print(f"   {'':<12} {checked} real choices checked -> {status}")
        ok &= bad == 0
    return ok


# ----------------------------------------- 2. pilot instructions vs behaviour
#
# These sentences are shown to human participants. If they do not describe what
# the scoring model accepts, we grade people against a rule we never gave them.

PILOT_CLAIMS = {
    "wall_hug":   lambda c, alts, cl, v: all(G.openness(c, cl) <= G.openness(a, cl) for a in alts),
    "open_field": lambda c, alts, cl, v: all(G.openness(c, cl) >= G.openness(a, cl) for a in alts),
    "evasive":    lambda c, alts, cl, v: all(
        abs(c[0] - G.VANTAGES[v][0]) + abs(c[1] - G.VANTAGES[v][1]) >=
        abs(a[0] - G.VANTAGES[v][0]) + abs(a[1] - G.VANTAGES[v][1]) for a in alts),
}


def check_pilot():
    print("\n2. pilot participant instructions vs. acceptable_moves()\n")
    ok = True
    for rule in PL.ASSIGNABLE_RULES:
        claim = PILOT_CLAIMS[rule]
        checked = bad = 0
        for dest in sorted(G.DESTS):
            for closed in PL.LAYOUTS:
                for v in range(len(G.VANTAGES)):
                    for r in range(G.R):
                        for c in range(G.C):
                            pos = (r, c)
                            if G.blocked(pos, closed):
                                continue
                            steps = _progressing((dest, rule), pos, closed)
                            if len(steps) < 2:
                                continue
                            acc = PL.acceptable_moves((dest, rule), pos, closed, v, None)
                            by_move = dict(steps)
                            for m in acc:
                                if m not in by_move:
                                    continue
                                alts = [n for mm, n in steps if mm != m]
                                checked += 1
                                if not claim(by_move[m], alts, closed, v):
                                    bad += 1
        words = PL.INSTRUCTIONS[rule].split(".")[1].strip()[:56]
        print(f"   {rule:<12} \"{words}...\"")
        print(f"   {'':<12} {checked} accepted moves checked -> "
              + ("OK" if not bad else f"VIOLATED in {bad}"))
        ok &= bad == 0
    return ok


# ------------------------------------------- 3. Leduc intent glosses
#
# Only the checkable fragments of each sentence. Deliberately partial: the point
# is to encode what the WORDS claim, not to restate the policy.

def _leduc_open(intent, rank, board=None):
    hist = () if board is None else ("check", "check", "board:" + board)
    return L.policy(intent, rank + "0", hist)


def _leduc_all_actions(intent):
    out = set()
    for card in L.DECK:
        for hist in [(), ("bet",), ("bet", "raise"),
                     ("check", "check", "board:J1"), ("check", "check", "board:J1", "bet"),
                     ("check", "check", "board:K1"), ("check", "check", "board:K1", "bet")]:
            out.add(L.policy(intent, card, hist))
    return out


LEDUC_CLAIMS = [
    ("give_up", "check and fold; conceding the pot",
     lambda: _leduc_all_actions("give_up") <= {"check", "fold"}),
    ("probe", "bet the first round with anything",
     lambda: all(_leduc_open("probe", r) == "bet" for r in L.RANKS)),
    ("trap", "check strong hands to induce a bet",
     lambda: _leduc_open("trap", "K") == "check"),
    ("represent", "check round 1, then bet a Q/K board regardless of holding",
     lambda: all(_leduc_open("represent", r) == "check" for r in L.RANKS)
     and all(_leduc_open("represent", r, b) == "bet" for r in L.RANKS for b in ("Q", "K"))),
    ("pot_control", "bet K early, then check round 2",
     lambda: _leduc_open("pot_control", "K") == "bet"
     and all(_leduc_open("pot_control", r, b) == "check" for r in L.RANKS for b in L.RANKS)),
    ("bluff", "bet J to represent strength",
     lambda: _leduc_open("bluff", "J") == "bet"),
    ("value_bet", "bet/raise strong hands (K, pairs) for value",
     lambda: _leduc_open("value_bet", "K") == "bet"),
]


def check_leduc():
    print("\n3. Leduc intent glosses vs. policy()\n")
    ok = True
    for intent, fragment, claim in LEDUC_CLAIMS:
        held = claim()
        gloss = L.INTENT_GLOSS[intent]
        if fragment.split(";")[0].split(" regardless")[0][:12] not in gloss:
            print(f"   {intent:<12} GLOSS CHANGED -- the checked fragment "
                  f"{fragment[:40]!r} is no longer in INTENT_GLOSS; update the claim")
            ok = False
            continue
        print(f"   {intent:<12} \"{fragment}\" -> " + ("OK" if held else "VIOLATED"))
        ok &= held
    return ok


def check():
    print("=" * 72)
    print("DESCRIPTIONS vs BEHAVIOUR")
    print("=" * 72)
    ok = check_gridworld() & check_pilot() & check_leduc()
    print("\n" + ("  ALL DESCRIPTIONS MATCH BEHAVIOUR"
                  if ok else "  MISMATCH -- a description is lying about the code"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
