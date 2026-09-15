#!/usr/bin/env python3
"""
Gridworld navigation -- the second environment.

Why a second environment at all: the claim is about a MECHANISM (hypothesis set
over intents-as-policies, eliminate what behaviour rules out, act to shrink the
rest). One environment cannot distinguish "the mechanism works" from "poker's
tree happens to suit it". Two can.

Why a grid specifically:
  1. A probe is legible in one sentence -- close a corridor and see which way
     they turn. In poker you have to explain a betting tree first.
  2. It is the setting the goal-recognition-design literature actually uses
     (Keren, Gal & Karpas), so this is a direct comparison rather than an
     analogy.
  3. It lets us do the thing we have been citing but NOT doing. GRD redesigns
     the environment IN ADVANCE so that whichever goal the agent holds becomes
     evident sooner. Everything we had before was online inquiry inside a fixed
     environment. Here we implement both, and report them separately:
         --mode online   observer acts during the episode
         --mode design   observer fixes the layout before the subject starts
     They are not the same thing and are never averaged together.

Everything is exact. The environment is deterministic given (start, intent), so
episodes are ENUMERATED -- all starts x all intents -- rather than sampled. No
seeds are needed except for the `random` observer, which is the only stochastic
element.

    python3 gridworld.py --mode online --condition all
    python3 gridworld.py --mode design --condition all
    python3 gridworld.py --map
"""

import argparse
import itertools
import random
from collections import deque
from functools import lru_cache

import core

# ------------------------------------------------------------------ the map
#
#      0 1 2 3 4 5 6 7
#   0  . . . # . . . .
#   1  S . . G . . # .      G = gate (observer may open/close)
#   2  . . . # . . # .      # = wall
#   3  S # . # . . . .      S = possible subject start
#   4  . . . G . # . A      A/B/C = candidate destinations
#   5  S . . # . # . .
#   6  . . . G . . . B
#   7  S . . # . . # C
#
# The wall down column 3 is the point: every route from the left half to any
# destination must pass one of three gates, so closing one is a real question
# and the answer is a turn.

R = C = 8
WALLS = frozenset([
    (0, 3), (2, 3), (3, 3), (5, 3), (7, 3),          # column-3 wall
    (1, 6), (2, 6), (4, 5), (5, 5), (7, 6), (3, 1),  # interior clutter
])
GATES = ((1, 3), (4, 3), (6, 3))                     # indices 0, 1, 2
STARTS = ((1, 0), (3, 0), (5, 0), (7, 0))
DESTS = {"A": (4, 7), "B": (6, 7), "C": (7, 7)}
VANTAGES = ((0, 5), (4, 4), (7, 4))                  # where the observer may stand

MOVES = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
MOVE_ORDER = ("N", "E", "S", "W")                    # deterministic final tie-break
STEP_CAP = 30

# Routing rules: HOW the subject goes, not WHERE. Two intents sharing a
# destination but routing differently are the interesting case -- the direct
# analogue of trap vs value_bet in poker, which differ only in how they get
# to the same place.
RULES = ("direct", "wall_hug", "open_field", "evasive")
RULE_GLOSS = {
    "direct":     "any shortest path; fixed tie-break",
    "wall_hug":   "among shortest steps, keep to cells with the most walls around them",
    "open_field": "among shortest steps, keep to the most open cells",
    "evasive":    "among shortest steps, maximise distance from the observer",
}
INTENTS = tuple((d, r) for d in sorted(DESTS) for r in RULES)   # 3 x 4 = 12


def name(intent):
    return f"{intent[0]}/{intent[1]}"


# ------------------------------------------------------------- grid queries

def blocked(cell, closed):
    """closed is a bitmask over GATES."""
    if cell in WALLS:
        return True
    for i, g in enumerate(GATES):
        if cell == g and (closed >> i) & 1:
            return True
    return False


def in_bounds(cell):
    return 0 <= cell[0] < R and 0 <= cell[1] < C


def neighbours(cell, closed):
    out = []
    for m, (dr, dc) in MOVES.items():
        n = (cell[0] + dr, cell[1] + dc)
        if in_bounds(n) and not blocked(n, closed):
            out.append((m, n))
    return out


@lru_cache(maxsize=None)
def dist_field(dest, closed):
    """BFS distance to `dest` on the grid with `closed` gates shut."""
    d = {dest: 0}
    q = deque([dest])
    while q:
        cur = q.popleft()
        for _, n in neighbours(cur, closed):
            if n not in d:
                d[n] = d[cur] + 1
                q.append(n)
    return d


@lru_cache(maxsize=None)
def openness(cell, closed):
    """How many of the four neighbours are free -- the wall_hug / open_field key."""
    return len(neighbours(cell, closed))


# ------------------------------------------------------------- the policies
#
# An intent is a TOTAL policy: for every (position, layout, observer position)
# it says what the subject would do, including in situations that never arose.
# Same commitment as poker -- that is what makes an intent counterfactual
# rather than a label.

@lru_cache(maxsize=None)
def policy(intent, pos, closed, vantage):
    dest_key, rule = intent
    dest = DESTS[dest_key]
    if pos == dest:
        return "stop"
    d = dist_field(dest, closed)
    here = d.get(pos)
    if here is None:
        return "stop"                     # unreachable; cannot happen (see legal_close)
    steps = [(m, n) for m, n in neighbours(pos, closed) if d.get(n, 1 << 20) == here - 1]
    if not steps:
        return "stop"
    if rule == "direct":
        key = lambda mn: 0
    elif rule == "wall_hug":
        key = lambda mn: -openness(mn[1], closed)
    elif rule == "open_field":
        key = lambda mn: openness(mn[1], closed)
    else:                                  # evasive
        v = VANTAGES[vantage]
        key = lambda mn: -(abs(mn[1][0] - v[0]) + abs(mn[1][1] - v[1]))
    best = min(key(s) for s in steps)
    return min((m for m, n in steps if key((m, n)) == best), key=MOVE_ORDER.index)


def step(pos, move):
    dr, dc = MOVES[move]
    return (pos[0] + dr, pos[1] + dc)


# ---------------------------------------------------------- observer actions

ACTIONS = ("wait",) + tuple(f"close{i}" for i in range(len(GATES))) \
                    + tuple(f"open{i}" for i in range(len(GATES))) \
                    + tuple(f"watch{i}" for i in range(len(VANTAGES)))
COST = {"wait": 0.0}
COST.update({f"close{i}": 2.0 for i in range(len(GATES))})
COST.update({f"open{i}": 1.0 for i in range(len(GATES))})
COST.update({f"watch{i}": 1.0 for i in range(len(VANTAGES))})

ALL_OPEN = 0
ALL_CLOSED = (1 << len(GATES)) - 1


def legal_close(closed):
    """Gate configurations the observer is allowed to produce.

    THE CONSTRAINT IS GRD'S OWN. Goal recognition design minimises worst-case
    distinctiveness *subject to not preventing agents from achieving their
    goals*. Here that is literally "you may not shut the last open gate" --
    every destination must stay reachable from every start. Without it the
    observer would trivially "identify" intent by making the task impossible.
    """
    return closed != ALL_CLOSED


def apply_action(a, closed, vantage):
    if a == "wait":
        return closed, vantage
    if a.startswith("close"):
        return closed | (1 << int(a[5:])), vantage
    if a.startswith("open"):
        return closed & ~(1 << int(a[4:])), vantage
    return closed, int(a[5:])


def legal_actions(closed, vantage):
    out = []
    for a in ACTIONS:
        c2, v2 = apply_action(a, closed, vantage)
        if not legal_close(c2):
            continue
        if (c2, v2) == (closed, vantage) and a != "wait":
            continue                       # no-op dressed as an action
        out.append(a)
    return out


# --------------------------------------------------------------- the rule
#
# Identical in shape to the poker observer: partition the live hypothesis set by
# the outcome each action would induce, score by expected surviving intent-set
# size, take the minimiser. mu buys cost-awareness exactly as in poker.

@lru_cache(maxsize=None)
def V(H, pos, closed, vantage, t, mu):
    """Expected (final |intent set| + mu * remaining observer cost) under
    optimal observer play from here, faithful subject, uniform prior over H."""
    if len(H) <= 1 or t >= STEP_CAP:
        return float(len(H))
    best = None
    for a in legal_actions(closed, vantage):
        c2, v2 = apply_action(a, closed, vantage)
        buckets = {}
        for i in H:
            buckets.setdefault(policy(i, pos, c2, v2), []).append(i)
        val = mu * COST[a]
        for m, Hb in buckets.items():
            w = len(Hb) / len(H)
            if m == "stop":
                val += w * len(Hb)
            else:
                val += w * V(tuple(Hb), step(pos, m), c2, v2, t + 1, mu)
        best = val if best is None else min(best, val)
    return best


def choose_action(H, pos, closed, vantage, t, condition, rng, mu=0.0):
    acts = legal_actions(closed, vantage)
    if condition == "passive":
        return "wait"
    if condition == "random":
        return rng.choice(acts)
    if len(H) <= 1:
        return "wait"                      # nothing left to learn; stop paying
    scored = []
    for a in acts:
        c2, v2 = apply_action(a, closed, vantage)
        buckets = {}
        for i in H:
            buckets.setdefault(policy(i, pos, c2, v2), []).append(i)
        val = mu * COST[a]
        for m, Hb in buckets.items():
            w = len(Hb) / len(H)
            val += w * (len(Hb) if m == "stop"
                        else V(tuple(Hb), step(pos, m), c2, v2, t + 1, mu))
        scored.append((val, COST[a], ACTIONS.index(a), a))
    return min(scored)[3]                  # ties -> cheaper, then fixed order


# ------------------------------------------------- environment design (GRD)

@lru_cache(maxsize=None)
def design_value(closed, mu):
    """Expected final |intent set| if the layout is fixed to `closed` before the
    episode starts and the observer then does nothing -- averaged over starts
    and over intents under a uniform prior. This is the GRD objective."""
    tot = 0.0
    for start in STARTS:
        H = INTENTS
        tot += _passive_run_value(tuple(H), start, closed)
    return tot / len(STARTS) + mu * sum(COST[f"close{i}"]
                                        for i in range(len(GATES)) if (closed >> i) & 1)


@lru_cache(maxsize=None)
def _passive_run_value(H, pos, closed):
    """Expected surviving |H| with a silent observer on a fixed layout."""
    if len(H) <= 1:
        return float(len(H))
    buckets = {}
    for i in H:
        buckets.setdefault(policy(i, pos, closed, 1), []).append(i)
    val = 0.0
    for m, Hb in buckets.items():
        w = len(Hb) / len(H)
        val += w * (len(Hb) if m == "stop"
                    else _passive_run_value(tuple(Hb), step(pos, m), closed))
    return val


def choose_design(condition, rng, mu=0.0):
    configs = [c for c in range(1 << len(GATES)) if legal_close(c)]
    if condition == "passive":
        return ALL_OPEN                    # leave the world alone
    if condition == "random":
        return rng.choice(configs)
    return min(configs, key=lambda c: (design_value(c, mu), bin(c).count("1"), c))


# ------------------------------------------------------------- one episode

def run_episode(condition, start, declared, rng, mode="online", mu=0.0, play_as=None):
    played = play_as or declared
    closed, vantage = ALL_OPEN, 1
    cost = 0.0
    design = None

    if mode == "design":
        design = choose_design(condition, rng, mu)
        cost += sum(COST[f"close{i}"] for i in range(len(GATES)) if (design >> i) & 1)
        closed = design

    H = tuple(INTENTS)
    pos, t = start, 0
    subject_actions = forced = deviations = 0
    path = [pos]

    while t < STEP_CAP:
        if mode == "online":
            a = choose_action(H, pos, closed, vantage, t, condition, rng, mu)
            cost += COST[a]
            closed, vantage = apply_action(a, closed, vantage)

        m = policy(played, pos, closed, vantage)
        # A step is FORCED if the observer's interventions changed what the
        # subject does from this very square -- the same position compared
        # under the untouched layout. That is the probe actually landing.
        if policy(played, pos, ALL_OPEN, 1) != m:
            forced += 1
        if m != policy(declared, pos, closed, vantage):
            deviations += 1
        subject_actions += 1

        H = tuple(i for i in H if policy(i, pos, closed, vantage) == m)
        if m == "stop":
            break
        pos = step(pos, m)
        path.append(pos)
        t += 1

    final = sorted(name(i) for i in H)
    dn = name(declared)
    return {
        "condition": condition, "mode": mode, "start": start,
        "declared": dn, "played": name(played),
        "final_set": final, "final_size": len(final),
        "sound": dn in final,
        "exact": final == [dn],
        "misattributed": len(final) == 1 and final != [dn],
        "contradiction": len(final) == 0,
        "subject_actions": subject_actions,
        "forced_decisions": forced,
        "deviations": deviations,
        "observer_cost": -cost,            # negative: cost is a loss, as chips are
        "design": design, "path": path,
    }


# ------------------------------------------------- the consistent misattributor
#
# The poker result that mattered most was that a subject who stays strictly
# inside the model -- declares one intent, faithfully plays ANOTHER -- drives
# the observer to a confident wrong answer, and that probing harder makes it
# worse. If that is a property of the mechanism rather than of poker, it must
# reappear here. Same construction: play the most pinnable intent that is not
# the declared one. The environment is deterministic, so "most pinnable" is a
# simulation, not an estimate.

def misattributing_intent(start, declared, condition, mode, mu=0.0):
    best = None
    for j in INTENTS:
        if j == declared:
            continue
        r = run_episode(condition, start, declared, random.Random(0), mode, mu, play_as=j)
        score = (1 if r["misattributed"] else 0, -r["final_size"])
        if best is None or score > best[0]:
            best = (score, j)
    return best[1]


def all_episodes(condition, mode, rng, mu=0.0, play_as_fn=None):
    """Exhaustive: every start x every declared intent. No sampling."""
    out = []
    for start in STARTS:
        for declared in INTENTS:
            pa = play_as_fn(start, declared) if play_as_fn else None
            out.append(run_episode(condition, start, declared, rng, mode, mu, pa))
    return out


# ------------------------------------------------------------------ report

KEYS = ["exact", "H", "sound", "misID", "forced", "cost"]


def grd_legal_designs():
    """Configurations a FAITHFUL GRD designer may produce.

    Keren, Gal & Karpas (ICAPS 2014) do not merely require goals to stay
    reachable. Their design problem is

        minimize_{A-} (wcd(D_{A\\A-}), |A-|)
        subject to  for every goal G:  C*_D(G) = C*_{D\\A-}(G)

    -- "as a way of maintaining 'user comfort' in the model we require the
    solution to preserve the original optimal solution length of all goals."
    So a removal that merely lengthens a route is ILLEGAL, not just costly.

    Our `legal_close` ("do not shut the last gate") is a strictly weaker
    relaxation of that. This function reports what the real constraint allows.
    """
    base = {(s, k): dist_field(d, ALL_OPEN)[s]
            for s in STARTS for k, d in DESTS.items()}
    out = []
    for c in range(1 << len(GATES)):
        ok = True
        for s in STARTS:
            for k, d in DESTS.items():
                f = dist_field(d, c)
                if s not in f or f[s] != base[(s, k)]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append(c)
    return out


def _passive_episode(start, intent, closed, vantage):
    """One episode on a layout fixed before it starts, observer silent."""
    H = tuple(INTENTS)
    pos, t = start, 0
    while t < STEP_CAP:
        m = policy(intent, pos, closed, vantage)
        H = tuple(i for i in H if policy(i, pos, closed, vantage) == m)
        if m == "stop":
            break
        pos = step(pos, m)
        t += 1
    return H


def static_bound():
    """MACHINE-CHECKED upper bounds on exact ID for a fixed layout.

    Brute force, not argument. Returns
      (best_single, best_per_start, pooled, n, n_configs)
    with
      best_single   the best exact-ID achievable by ONE layout+vantage held
                    fixed for every episode -- what a designer actually picks;
      best_per_start a strictly MORE permissive designer that may pick a
                    different layout for each start (an oracle: it is told the
                    start before choosing). Reported to make the bound
                    adversarial rather than convenient;
      pooled        the weaker argument: two intents that agree under EVERY
                    fixed configuration cannot be separated by any one of them.

    All three bound FIXED layouts only. None of them bounds an observer that
    reconfigures mid-episode.
    """
    configs = [(c, v) for c in range(1 << len(GATES)) if legal_close(c)
               for v in range(len(VANTAGES))]
    n = len(STARTS) * len(INTENTS)

    best_single = 0
    for c, v in configs:
        hits = sum(1 for s in STARTS for i in INTENTS
                   if [x for x in _passive_episode(s, i, c, v)] == [i])
        best_single = max(best_single, hits)

    best_per_start = sum(
        max(sum(1 for i in INTENTS if [x for x in _passive_episode(s, i, c, v)] == [i])
            for c, v in configs)
        for s in STARTS)

    pooled, _ = static_ceiling()
    return best_single, best_per_start, pooled, n, len(configs)


def witness():
    """The concrete pair behind the separation, regenerated from scratch.

        python3 gridworld.py --witness
    """
    s, i1, i2 = (3, 0), ("A", "direct"), ("A", "open_field")
    print(f"\n  WITNESS: start {s}, {name(i1)} vs {name(i2)}\n")
    same = True
    for c in range(1 << len(GATES)):
        if not legal_close(c):
            continue
        for v in range(len(VANTAGES)):
            t1 = _trace(i1, s, c, v)
            t2 = _trace(i2, s, c, v)
            if t1 != t2:
                same = False
                print(f"    differ under fixed layout {bin(c)} vantage {v}")
    print(f"    identical under ALL {len([c for c in range(1 << len(GATES)) if legal_close(c)]) * len(VANTAGES)}"
          f" fixed configurations: {same}")

    print("\n  Now with an observer that reconfigures mid-episode:\n")
    print(f"    {'t':>2}  {'observer':<9} {'gates':>6} {'pos':<8} "
          f"{name(i1):<14} {name(i2):<14}")
    closed, v, t, pos = ALL_OPEN, 1, 0, s
    H = tuple(INTENTS)
    while t < STEP_CAP:
        a = choose_action(H, pos, closed, v, t, "adaptive", None, 0.0)
        closed, v = apply_action(a, closed, v)
        m1, m2 = policy(i1, pos, closed, v), policy(i2, pos, closed, v)
        flag = "" if m1 == m2 else "   <-- SEPARATED"
        print(f"    {t:>2}  {a:<9} {bin(closed)[2:]:>6} {str(pos):<8} "
              f"{m1:<14} {m2:<14}{flag}")
        if m1 != m2 or m1 == "stop":
            break
        H = tuple(x for x in H if policy(x, pos, closed, v) == m1)
        pos = step(pos, m1)
        t += 1


def _trace(intent, start, closed, vantage):
    out, pos, t = [], start, 0
    while t < STEP_CAP:
        m = policy(intent, pos, closed, vantage)
        out.append(m)
        if m == "stop":
            break
        pos = step(pos, m)
        t += 1
    return out


def static_ceiling():
    """Upper bound on exact ID for ANY observer whose layout is fixed for the
    whole episode -- including the best possible environment design.

    Pools the subject's behaviour under EVERY legal layout and vantage: two
    intents that agree everywhere in that pool cannot be told apart by any
    single fixed configuration, since one configuration sees strictly less than
    the pool. Online probing is NOT bounded by this, because it can change the
    layout mid-episode; that is the whole difference between the two modes."""
    uniq = 0
    for s in STARTS:
        sig = {}
        for i in INTENTS:
            tr = []
            for closed in range(1 << len(GATES)):
                if not legal_close(closed):
                    continue
                for v in range(len(VANTAGES)):
                    pos, t = s, 0
                    while t < STEP_CAP:
                        m = policy(i, pos, closed, v)
                        tr.append(m)
                        if m == "stop":
                            break
                        pos = step(pos, m)
                        t += 1
            sig.setdefault(tuple(tr), []).append(i)
        uniq += sum(1 for v in sig.values() if len(v) == 1)
    return uniq, len(STARTS) * len(INTENTS)


def print_map():
    print("\n  gridworld  (# wall, G gate, S start, A/B/C destinations, v vantage)\n")
    print("      " + " ".join(str(c) for c in range(C)))
    for r in range(R):
        row = []
        for c in range(C):
            cell = (r, c)
            ch = "."
            if cell in WALLS:
                ch = "#"
            if cell in GATES:
                ch = "G"
            if cell in STARTS:
                ch = "S"
            if cell in VANTAGES:
                ch = "v"
            for k, d in DESTS.items():
                if cell == d:
                    ch = k
            row.append(ch)
        print(f"   {r}  " + " ".join(row))
    print(f"\n  intents: {len(INTENTS)} = {len(DESTS)} destinations x {len(RULES)} routing rules")
    for r in RULES:
        print(f"    {r:<11} {RULE_GLOSS[r]}")
    print(f"  episodes per cell: {len(STARTS)} starts x {len(INTENTS)} intents = "
          f"{len(STARTS) * len(INTENTS)}, enumerated exhaustively")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="online", choices=["online", "design", "both"])
    ap.add_argument("--condition", default="all",
                    choices=["passive", "random", "adaptive", "all"])
    ap.add_argument("--subject", default="faithful",
                    choices=["faithful", "misattribute"],
                    help="faithful: plays the declared intent; misattribute: declares one "
                         "intent and faithfully plays another, chosen to be most pinnable")
    ap.add_argument("--mu", type=float, default=0.0,
                    help="observer cost weight: minimise E[|H|] + mu * E[cost]")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--map", action="store_true", help="print the grid and exit")
    ap.add_argument("--witness", action="store_true",
                    help="regenerate the separation witness pair and exit")
    ap.add_argument("--bounds", action="store_true",
                    help="machine-check the static-layout bounds and exit")
    args = ap.parse_args()

    if args.map:
        print_map()
        return
    if args.witness:
        witness()
        return
    if args.bounds:
        bs, bp, pl, n, nc = static_bound()
        print(f"\n  configurations enumerated (layout x vantage): {nc}")
        print(f"  episodes per configuration:                   {n}")
        print(f"  best SINGLE fixed configuration : {bs}/{n} = {bs / n:.1%}")
        print(f"  best per-start (oracle designer): {bp}/{n} = {bp / n:.1%}")
        print(f"  pooled-trace upper bound        : {pl}/{n} = {pl / n:.1%}")
        print(f"  GRD-legal designs (cost-preserving): {grd_legal_designs()}")
        return

    conds = ["passive", "random", "adaptive"] if args.condition == "all" else [args.condition]
    modes = ["online", "design"] if args.mode == "both" else [args.mode]

    for mode in modes:
        results = {}
        for cond in conds:
            per_seed = []
            # passive and adaptive are deterministic: the seed changes nothing,
            # so one pass is the whole population. Only `random` needs seeds.
            seeds = args.seeds if cond == "random" else args.seeds[:1]
            fn = (None if args.subject == "faithful" else
                  (lambda st, d, c=cond, m=mode: misattributing_intent(st, d, c, m, args.mu)))
            for s in seeds:
                rows = all_episodes(cond, mode, random.Random(s), args.mu, fn)
                per_seed.append(core.summarise(rows, KEYS))
            results[cond] = core.aggregate(per_seed)
        title = ("ONLINE PROBING -- observer acts during the episode"
                 if mode == "online" else
                 "ENVIRONMENT DESIGN (GRD) -- layout fixed before the subject starts")
        title += f"   [subject: {args.subject}]"
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        core.print_table(results, KEYS, args.seeds, len(STARTS) * len(INTENTS), "observer", total_n=len(STARTS) * len(INTENTS), notes=[
            "cost/ep    = observer's cost, negated so more-negative is worse (as chips are)",
            "forced/ep  = subject steps the observer's interventions actually changed",
            "n is EXHAUSTIVE (all starts x all intents), NOT sampled, so for passive",
            "  and adaptive -- which are deterministic -- [min, max] is degenerate by",
            "  construction and n is the whole population. Only `random` actually",
            "  varies with the seed.",
        ])


if __name__ == "__main__":
    main()
