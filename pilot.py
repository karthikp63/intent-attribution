#!/usr/bin/env python3
"""
Gridworld human pilot -- INSTRUMENT ONLY. Built to be reviewed before it is run.

WHY THIS EXISTS. Every number in this project comes from a subject that follows
its declared intent by construction. The one human trial we have ran on poker and
put posterior mass on the declared intent at 0.24 against a 0.20 uniform prior.
Nothing here is validated on people.

WHAT THE POKER PILOT GOT WRONG, and this must not repeat:
  * 20 episodes, one participant -- no power to say anything
  * intents SELF-CHOSEN from a menu, so 16 of 20 were the same one
  * soundness was reported as the headline, and it went to 100% under soft
    scoring while posterior mass stayed at 0.24. It is the metric that HID the
    failure.

DESIGN DECISIONS, and the reasoning, so they are reviewable:

1. INTENTS ARE ASSIGNED AND BALANCED. The participant is told which intent to
   act on, round-robin over the intent set, deterministic per participant. Never
   a menu.

2. THE HEADLINE IS POSTERIOR MASS ON THE ASSIGNED INTENT, against the uniform
   prior. Soundness is logged but never led with.

3. TIES ARE EQUIPROBABLE, NOT ARBITRARY. Our routing rules end in a fixed
   N/E/S/W tie-break -- an implementation detail no human would reproduce. So
   the likelihood treats every move the rule leaves open as equally likely:

       P(a | intent) = (1 - eps) * [a in acceptable] / |acceptable|
                     + eps / |legal|

   Scoring a human against the arbitrary tie-break would manufacture misfit that
   is our artifact, not their behaviour. This is the single most important
   difference from how the poker pilot scored.

4. EVERYTHING IS LOGGED TO RESCORE OFFLINE. Each record carries the full state
   at every decision, so any epsilon, prior or metric can be recomputed later
   without bringing anyone back.

    python3 pilot.py --check                     # instrument self-check, no human
    python3 pilot.py --dry-run --participant P1  # see what a session looks like
    python3 pilot.py --run --participant P1 --episodes 100
    python3 pilot.py --analyse runs/P1.jsonl
"""

import argparse
import json
import math
import os
import random
import sys
import time

import gridworld as G

# Rules a participant can actually be asked to follow, in plain language.
# `direct` is EXCLUDED by default: it says "take any shortest path", so the
# implementation's tie-break carries all of its content and a human cannot
# reproduce that. Including it would measure our arbitrary move ordering.
INSTRUCTIONS = {
    "wall_hug":   "Take a shortest route. When you have a real choice, step to the "
                  "square with MORE walls around it -- stay close to walls.",
    "open_field": "Take a shortest route. When you have a real choice, step to the "
                  "square with FEWER walls around it -- stay in the open.",
    "evasive":    "Take a shortest route. When you have a real choice, step to the "
                  "square FURTHER from the observer (shown as O).",
    "direct":     "Take a shortest route. When you have a choice, it does not matter "
                  "which you take.",
}
ASSIGNABLE_RULES = ("wall_hug", "open_field", "evasive")
LAYOUTS = (0, 4)


def assignable_intents(include_direct=False):
    rules = G.RULES if include_direct else ASSIGNABLE_RULES
    return [(d, r) for d in sorted(G.DESTS) for r in rules]


# --------------------------------------------------------- the noise model

def acceptable_moves(intent, pos, closed, vantage, last_move):
    """Every move the rule leaves open -- BEFORE the arbitrary final tie-break.

    This is the heart of the instrument. `gridworld.policy` returns one move by
    applying a fixed N/E/S/W order to whatever the rule leaves tied. A person
    will not reproduce that order, so scoring them against it invents misfit.
    Here the rule narrows, and whatever it does not decide stays open.
    """
    dest_key, rule = intent
    dest = G.DESTS[dest_key]
    if pos == dest:
        return {"stop"}
    d = G.dist_field(dest, closed)
    here = d.get(pos)
    if here is None:
        return {"stop"}
    steps = [(m, n) for m, n in G.neighbours(pos, closed)
             if d.get(n, 1 << 20) == here - 1]
    if not steps:
        return {"stop"}
    if rule == "direct":
        return {m for m, _ in steps}
    if rule == "wall_hug":
        key = lambda mn: G.openness(mn[1], closed)
    elif rule == "open_field":
        key = lambda mn: -G.openness(mn[1], closed)
    else:                                            # evasive
        v = G.VANTAGES[vantage]
        key = lambda mn: -(abs(mn[1][0] - v[0]) + abs(mn[1][1] - v[1]))
    best = min(key(s) for s in steps)
    return {m for m, n in steps if key((m, n)) == best}


def legal_moves(pos, closed):
    return [m for m, _ in G.neighbours(pos, closed)] + ["stop"]


def posterior(decisions, eps, intents):
    """decisions: list of (pos, closed, vantage, last_move, action taken).
    Uniform prior over `intents`; returns a normalised dict."""
    w = {}
    for i in intents:
        lik = 1.0
        for pos, closed, vantage, last, a in decisions:
            acc = acceptable_moves(i, tuple(pos), closed, vantage, last)
            legal = legal_moves(tuple(pos), closed)
            p = eps / len(legal)
            if a in acc:
                p += (1.0 - eps) / len(acc)
            lik *= p
        w[i] = lik
    tot = sum(w.values())
    if tot <= 0:
        return {i: 1.0 / len(intents) for i in intents}
    return {i: v / tot for i, v in w.items()}


def eff_size(p):
    h = -sum(v * math.log(v) for v in p.values() if v > 0)
    return math.exp(h)


# ------------------------------------------------------------- the session

def session_plan(participant, n_episodes, include_direct=False, seed=None):
    """Balanced, assigned, deterministic per participant.

    Round-robin over the intent set so every intent is used equally often, then
    the ORDER is shuffled with a per-participant seed so intent is not confounded
    with practice or fatigue.
    """
    intents = assignable_intents(include_direct)
    plan = [intents[k % len(intents)] for k in range(n_episodes)]
    rng = random.Random(seed if seed is not None else abs(hash(participant)) % (2 ** 31))
    rng.shuffle(plan)
    out = []
    for k, intent in enumerate(plan):
        out.append({
            "episode": k + 1,
            "intent": list(intent),
            "start": list(G.STARTS[k % len(G.STARTS)]),
            "closed": LAYOUTS[k % len(LAYOUTS)],
            "vantage": k % len(G.VANTAGES),
        })
    return out


def render(pos, closed, vantage, dest_key):
    lines = ["     " + " ".join(str(c) for c in range(G.C))]
    for r in range(G.R):
        row = []
        for c in range(G.C):
            cell = (r, c)
            ch = "."
            if G.blocked(cell, closed):
                ch = "#"
            elif cell in G.GATES:
                ch = "+"
            if cell == G.VANTAGES[vantage]:
                ch = "O"
            if cell == G.DESTS[dest_key]:
                ch = dest_key
            if cell == pos:
                ch = "@"
            row.append(ch)
        lines.append(f"  {r}  " + " ".join(row))
    return "\n".join(lines)


BRIEF = """\
You are the @ on the grid. Walk to the letter shown, using the SHORTEST route.

  #  wall (cannot enter)      +  open gate
  O  the observer             A/B/C  destinations

Type one of: n s e w   (or q to quit the session)

Some steps have only one shortest option. When there is a real choice, follow
the rule you are given for that round. Do not try to be unpredictable, and do
not try to help -- just follow the rule.
"""


def run_session(participant, n_episodes, out_path, include_direct=False, dry_run=False):
    plan = session_plan(participant, n_episodes, include_direct)
    intents = assignable_intents(include_direct)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(BRIEF)
    print(f"Participant {participant}: {n_episodes} rounds, "
          f"{len(intents)} intents, each used {n_episodes // len(intents)}x.\n")

    with open(out_path, "a") as fh:
        for spec in plan:
            intent = tuple(spec["intent"])
            dest_key, rule = intent
            pos = tuple(spec["start"])
            closed, vantage = spec["closed"], spec["vantage"]
            decisions, last = [], None
            print("=" * 58)
            print(f"ROUND {spec['episode']}   go to {dest_key}")
            print(f"RULE: {INSTRUCTIONS[rule]}\n")
            while True:
                print(render(pos, closed, vantage, dest_key))
                if pos == G.DESTS[dest_key]:
                    print("  arrived.\n")
                    break
                if dry_run:
                    a = sorted(acceptable_moves(intent, pos, closed, vantage, last))[0]
                    print(f"  [dry-run plays {a}]")
                else:
                    raw = input("  move> ").strip().lower()
                    if raw in ("q", "quit"):
                        print("session ended early")
                        return
                    a = {"n": "N", "s": "S", "e": "E", "w": "W"}.get(raw)
                    if a is None or a not in legal_moves(pos, closed):
                        print("  not a legal move here; try again")
                        continue
                decisions.append([list(pos), closed, vantage, last, a])
                last = a
                pos = G.step(pos, a)
            rec = {
                "participant": participant, "episode": spec["episode"],
                "ts": time.time(),
                "assigned_intent": list(intent),
                "start": spec["start"], "closed": closed, "vantage": vantage,
                "decisions": decisions,
                "n_intents": len(intents), "include_direct": include_direct,
                "map_version": "2026-09-14",
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    print(f"\nwrote {out_path}")


# ---------------------------------------------------------------- analysis

EPS_GRID = (0.0, 0.05, 0.1, 0.2, 0.35)


def analyse(paths, eps_grid=EPS_GRID):
    recs = []
    for p in paths:
        with open(p) as fh:
            recs += [json.loads(ln) for ln in fh if ln.strip()]
    if not recs:
        print("no records")
        return
    by_p = {}
    for r in recs:
        by_p.setdefault(r["participant"], []).append(r)
    n_int = recs[0]["n_intents"]
    prior = 1.0 / n_int
    intents = assignable_intents(recs[0]["include_direct"])

    print(f"\n## Gridworld pilot -- {len(recs)} episodes, "
          f"{len(by_p)} participant(s), {n_int} intents\n")
    print(f"HEADLINE METRIC: posterior mass on the ASSIGNED intent.")
    print(f"Uniform prior = 1/{n_int} = {prior:.3f}. Mass at or near the prior means")
    print(f"the behaviour carried no information about the intent the person was given.\n")

    hdr = f"{'participant':<14}{'n':>5}" + "".join(f"{f'eps={e}':>11}" for e in eps_grid)
    print(hdr)
    print("-" * len(hdr))
    for who in sorted(by_p) + (["POOLED"] if len(by_p) > 1 else []):
        rows = recs if who == "POOLED" else by_p[who]
        cells = []
        for eps in eps_grid:
            m = sum(posterior(r["decisions"], eps, intents)[tuple(r["assigned_intent"])]
                    for r in rows) / len(rows)
            cells.append(f"{m:>11.3f}")
        print(f"{who:<14}{len(rows):>5}" + "".join(cells))
    print(f"{'(prior)':<14}{'':>5}" + "".join(f"{prior:>11.3f}" for _ in eps_grid))

    print("\nSupporting (never the headline):")
    print(f"{'participant':<14}{'MAP correct':>13}{'|H| eff':>10}{'in 95% HPD':>13}")
    print("-" * 50)
    for who in sorted(by_p) + (["POOLED"] if len(by_p) > 1 else []):
        rows = recs if who == "POOLED" else by_p[who]
        mp = hp = 0
        hs = 0.0
        for r in rows:
            p = posterior(r["decisions"], 0.1, intents)
            a = tuple(r["assigned_intent"])
            mp += max(p, key=p.get) == a
            hs += eff_size(p)
            acc, tot = [], 0.0
            for i, v in sorted(p.items(), key=lambda kv: -kv[1]):
                acc.append(i)
                tot += v
                if tot >= 0.95:
                    break
            hp += a in acc
        n = len(rows)
        print(f"{who:<14}{mp / n:>12.1%}{hs / n:>10.2f}{hp / n:>12.1%}")
    print("  (at eps = 0.1; soundness/HPD is reported because it is comparable to")
    print("   the poker pilot, NOT because it is the number to judge this by)")


# ------------------------------------------------------------ self-check

def check():
    """The instrument must work before anyone is recruited."""
    print("\n## Pilot instrument self-check\n")
    ok = True
    intents = assignable_intents()
    prior = 1.0 / len(intents)

    print("1. Assignment is BALANCED and never self-chosen.")
    plan = session_plan("P1", 102)
    counts = {}
    for spec in plan:
        counts[tuple(spec["intent"])] = counts.get(tuple(spec["intent"]), 0) + 1
    lo, hi = min(counts.values()), max(counts.values())
    print(f"   102 episodes over {len(intents)} intents: each used {lo}-{hi} times")
    ok &= hi - lo <= 1 and len(counts) == len(intents)

    print("\n2. A PERFECT participant must be identified; a RANDOM one must not.")
    for eps in (0.0, 0.1):
        perfect = rand = 0.0
        rng = random.Random(7)
        for spec in plan[:36]:
            intent = tuple(spec["intent"])
            for mode in ("perfect", "random"):
                pos, last, dec = tuple(spec["start"]), None, []
                for _ in range(G.STEP_CAP):
                    if pos == G.DESTS[intent[0]]:
                        break
                    acc = acceptable_moves(intent, pos, spec["closed"], spec["vantage"], last)
                    legal = [m for m in legal_moves(pos, spec["closed"]) if m != "stop"]
                    a = sorted(acc)[0] if mode == "perfect" else rng.choice(legal)
                    dec.append([list(pos), spec["closed"], spec["vantage"], last, a])
                    last = a
                    pos = G.step(pos, a)
                p = posterior(dec, eps, intents)[intent]
                if mode == "perfect":
                    perfect += p
                else:
                    rand += p
        n = 36
        print(f"   eps={eps}: perfect {perfect / n:.3f}   random {rand / n:.3f}   "
              f"prior {1 / len(intents):.3f}")
        ok &= perfect / n > rand / n

    print("\n3. A rule-following but TIE-INDIFFERENT subject -- the realistic human.")
    print("   They obey the rule but break its residual ties however they like.")
    masses, below = [], 0
    rng = random.Random(3)
    for spec in plan[:48]:
        intent = tuple(spec["intent"])
        pos, last, dec = tuple(spec["start"]), None, []
        for _ in range(G.STEP_CAP):
            if pos == G.DESTS[intent[0]]:
                break
            acc = sorted(acceptable_moves(intent, pos, spec["closed"], spec["vantage"], last))
            a = rng.choice(acc)
            dec.append([list(pos), spec["closed"], spec["vantage"], last, a])
            last = a
            pos = G.step(pos, a)
        m = posterior(dec, 0.0, intents)[intent]
        masses.append(m)
        below += m <= prior + 1e-9
    mean = sum(masses) / len(masses)
    print(f"   mean mass {mean:.3f} vs prior {prior:.3f}   "
          f"(worst {min(masses):.3f}, at-or-below prior in {below}/{len(masses)})")
    ok &= mean > prior

    print("\n   PRE-REGISTERED PREDICTION -- REVISED 2026-09-15 after testing it.")
    print("   The original prediction was an Occam bias: permissive intents would")
    print("   be systematically attributed to more SPECIFIC ones. `specificity.py`")
    print("   tested that and it does NOT hold. Confusions occur between intents of")
    print("   essentially identical permissiveness (median |perm difference| 0.014")
    print("   against a full range of 0.084), so permissiveness is not the driver.")
    print("   ")
    print("   What the data actually shows, and what we now predict instead:")
    print("     DESTINATION confusion dominates ROUTING-RULE confusion.")
    print("     52.6% of wrong attributions name the right rule and the wrong")
    print("     destination; only 38.9% name the right destination and the wrong")
    print("     rule. So report destination accuracy and rule accuracy SEPARATELY;")
    print("     a single intent-level number will be dominated by the destination")
    print("     component and will hide how well the rule was recovered.")

    print("\n4. Records round-trip: logged decisions rescore without the session.")
    rec = {"decisions": [[list(G.STARTS[0]), 0, 1, None, "S"]]}
    p = posterior(rec["decisions"], 0.1, intents)
    ok &= abs(sum(p.values()) - 1.0) < 1e-9
    print(f"   posterior over {len(intents)} intents sums to {sum(p.values()):.6f}")

    print("\n  " + ("INSTRUMENT OK" if ok else "INSTRUMENT BROKEN"))
    return ok


IRB = """
## IRB — what this would likely require

NOT LEGAL ADVICE. This is what to take to Yale's IRB, not a determination.

WHAT THE STUDY IS: 2-3 adults play ~100 rounds of a grid navigation game, about
20-30 minutes. No deception. No personal data beyond a participant code. No
sensitive categories. Data collected is keystrokes in a game.

LIKELY DETERMINATION: **exempt**, under the US Common Rule (45 CFR 46.104)
Category 3(i)(A) -- benign behavioural interventions with adults, where
information is recorded so that subjects cannot readily be identified. Category 2
(educational tests / observation of public behaviour) may also be argued. Exempt
status is a determination the IRB makes, NOT one the researcher may self-certify
at most institutions, so an exemption request still has to be filed and approved
before recruiting.

WHAT TO PREPARE ANYWAY:
  * Protocol describing the task, the ~25 minute duration, and the intent
    assignment procedure.
  * Consent: for exempt minimal-risk work this is usually an information sheet
    plus verbal or click-through agreement rather than a signed form. Say
    plainly that we record moves and timing, that participation is voluntary,
    and that they may stop at any time.
  * Data plan: participant codes only (P1, P2, ...); no names, emails or IP
    addresses in the logs; state where files live and who has access. The
    current logger writes exactly a code, timestamps and moves -- keep it that
    way and the privacy analysis stays trivial.
  * Recruitment text, and whether anyone is compensated. Compensating lab
    members or students raises undue-influence questions; note the relationship
    between recruiter and participant.

WHAT WOULD LOSE THE EXEMPTION: deceiving participants about the purpose,
recording audio or video, collecting identifiers, enrolling minors, or any
stress/discomfort element. None of that is in this design, and none of it should
be added casually.

SAMPLE SIZE: 2-3 participants is a pilot for instrument shakedown, not a study.
Say so in the protocol and in any write-up. At n = 3 we can detect "mass is at
the prior" but not estimate a population effect.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--irb", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyse", nargs="+", metavar="LOG")
    ap.add_argument("--participant", default="P1")
    ap.add_argument("--episodes", type=int, default=102)
    ap.add_argument("--include-direct", action="store_true",
                    help="also assign the `direct` rule (see the note in the source)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.irb:
        print(IRB)
        return
    if a.analyse:
        analyse(a.analyse)
        return
    if a.run or a.dry_run:
        out = a.out or f"runs/{a.participant}.jsonl"
        run_session(a.participant, a.episodes, out, a.include_direct, a.dry_run)
        return
    check()
    print(IRB)


if __name__ == "__main__":
    main()
