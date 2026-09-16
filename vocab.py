#!/usr/bin/env python3
"""
LLM as VOCABULARY proposer -- held-out routing-rule recovery.

WHY THIS AND NOT THE POKER VERSION. Proposing intents from a set we already
enumerated adds nothing; we can write those by hand. The live problem is the one
the pilot exposed: posterior mass on the intent a real person declared was 0.24
against a 0.20 uniform prior. Our vocabulary does not describe what the person
was doing. That is a vocabulary problem, and no amount of better elimination
fixes it.

WHY THE GRIDWORLD. A routing rule is INSPECTABLE. A human can read "keeps to
cells with walls around them" and judge it. A 36-cell poker policy table cannot
be eyeballed, so a proposal there can only be checked mechanically.

THE EXPERIMENT. Hold one routing rule out of the observer's vocabulary. The
subject uses it, so the vocabulary is provably incomplete -- the pilot situation
under conditions where we know exactly what is missing. Show the LLM some
trajectories, ask for a new routing rule, compile it, and then:

    RUN THE PROPOSED RULE ON EPISODES THE LLM NEVER SAW.

"Is the proposal plausible" is unfalsifiable. "Does it predict held-out
behaviour" is a real generalisation test, and it is only possible because we hid
the rule ourselves. Every rule is held out in turn so the result is not an
artifact of which one we picked.

The LLM stays a proposer: it never eliminates, never ranks, never selects a
probe. Everything it emits compiles and validates or is discarded.

    python3 vocab.py --selftest              # harness check, no key needed
    python3 vocab.py --backend live --proposer anthropic --n 3
"""

import argparse
import json
import random
import re

import gridworld as G
import proposer as P

# --------------------------------------------------------------------- DSL
#
# A routing rule is a TIE-BREAK over the moves that make progress: every rule
# takes a shortest path, and they differ only in which shortest path. So a rule
# compiles to an ordered list of (criterion, direction) pairs, applied in order,
# with the grid's fixed move order as the final tie-break. That makes a proposal
# three things at once: executable, total by construction, and readable by a
# human as a sentence.

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


CRITERIA = {
    # name              (cell, closed, vantage, dest, move, last_move) -> number
    "openness":        lambda c, cl, v, d, m, lm: G.openness(c, cl),
    "observer_dist":   lambda c, cl, v, d, m, lm: _manhattan(c, G.VANTAGES[v]),
    "momentum":        lambda c, cl, v, d, m, lm: 1 if m == lm else 0,
    "row":             lambda c, cl, v, d, m, lm: c[0],
    "col":             lambda c, cl, v, d, m, lm: c[1],
    "goal_row_align":  lambda c, cl, v, d, m, lm: -abs(c[0] - d[0]),
    "goal_col_align":  lambda c, cl, v, d, m, lm: -abs(c[1] - d[1]),
}
DIRECTIONS = ("max", "min")

CRITERION_GLOSS = {
    "openness":       "how many of the four neighbours of the square are free",
    "observer_dist":  "how far the square is from the observer, in steps",
    "momentum":       "whether the step continues in the same direction as the last one",
    "row":            "how far down the grid the square is",
    "col":            "how far right the square is",
    "goal_row_align": "how close the square's row already is to the destination's row",
    "goal_col_align": "how close the square's column already is to the destination's column",
}

# The four built-in rules, written in the DSL. `selftest` checks that these
# reproduce gridworld.policy EXACTLY -- if the DSL could not express the rule we
# hold out, recovery would be impossible by construction and the whole
# experiment would be measuring nothing.
BUILTIN_SPEC = {
    "direct":     [],
    "wall_hug":   [("openness", "min")],
    "open_field": [("openness", "max")],
    "evasive":    [("observer_dist", "max")],
}


def compile_spec(spec):
    """-> (normalised spec, error). Total by construction once it compiles."""
    if not isinstance(spec, list):
        return None, f"expected a list of criteria, got {type(spec).__name__}"
    out = []
    for item in spec:
        if isinstance(item, dict):
            crit, direction = item.get("criterion"), item.get("direction")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            crit, direction = item
        else:
            return None, f"cannot read criterion entry {item!r}"
        crit = str(crit).strip().lower()
        direction = str(direction).strip().lower()
        if crit not in CRITERIA:
            return None, f"unknown criterion {crit!r} (known: {sorted(CRITERIA)})"
        if direction not in DIRECTIONS:
            return None, f"direction must be max or min, got {direction!r}"
        out.append((crit, direction))
    if len(out) > 4:
        return None, f"too many criteria ({len(out)}); at most 4"
    return out, None


def rule_move(spec, pos, closed, vantage, dest, last_move):
    """Apply a compiled rule. Always returns a legal action -- totality."""
    if pos == dest:
        return "stop"
    d = G.dist_field(dest, closed)
    here = d.get(pos)
    if here is None:
        return "stop"
    steps = [(m, n) for m, n in G.neighbours(pos, closed)
             if d.get(n, 1 << 20) == here - 1]
    if not steps:
        return "stop"
    best = steps
    for crit, direction in spec:
        f = CRITERIA[crit]
        vals = [f(n, closed, vantage, dest, m, last_move) for m, n in best]
        target = max(vals) if direction == "max" else min(vals)
        best = [s for s, v in zip(best, vals) if v == target]
    return min((m for m, _ in best), key=G.MOVE_ORDER.index)


def trajectory_spec(spec, start, dest_key, closed, vantage):
    dest = G.DESTS[dest_key]
    pos, last, out = start, None, []
    for _ in range(G.STEP_CAP):
        m = rule_move(spec, pos, closed, vantage, dest, last)
        out.append(m)
        if m == "stop":
            break
        pos = G.step(pos, m)
        last = m
    return tuple(out)


def trajectory_builtin(rule, start, dest_key, closed, vantage):
    return tuple(G._trace((dest_key, rule), start, closed, vantage))


# ------------------------------------------------------------ the episodes
#
# Shown / held-back split. The LLM sees trajectories from two starts; the test
# is the other two starts, under the same two layouts. It never sees them.

SHOWN_STARTS = (0, 1)
HELD_STARTS = (2, 3)
LAYOUTS = (0, 4)          # all gates open, and gate 2 closed (both GRD-legal)
VANTS = (0, 1, 2)         # the observer's position must vary, or `evasive` and
                          # `direct` coincide and the experiment cannot tell
                          # them apart -- checked in the self-test.


def episodes(idx_starts):
    return [(G.STARTS[i], d, c, v)
            for i in idx_starts for d in sorted(G.DESTS)
            for c in LAYOUTS for v in VANTS]


# ------------------------------------------------------------ serialisation

def describe_grid():
    rows = []
    for r in range(G.R):
        line = []
        for c in range(G.C):
            cell = (r, c)
            ch = "#" if cell in G.WALLS else "."
            if cell in G.GATES:
                ch = "G"
            for k, d in G.DESTS.items():
                if cell == d:
                    ch = k
            line.append(ch)
        rows.append(f"  row {r}: " + " ".join(line))
    return "\n".join(rows)


def serialise_episode(start, dest_key, closed, vantage, moves):
    shut = [i for i in range(len(G.GATES)) if (closed >> i) & 1]
    gates = ("gates all open" if not shut
             else "gate rows shut: " + ",".join(str(G.GATES[i][0]) for i in shut))
    seq = " ".join(m for m in moves if m != "stop")
    return (f"  from {start} to {dest_key}{G.DESTS[dest_key]}, {gates}, "
            f"observer at {G.VANTAGES[vantage]}:  {seq}")


# Prompt versions. v1 is the UNTUNED baseline and is never edited: the reported
# numbers for it must stay reproducible. v2 targets the two failure modes v1
# produced, both measured rather than guessed:
#   (a) OVER-SPECIFICATION -- 160/160 constrained proposals used exactly 4
#       criteria against true specs of length 0 or 1. "At most 4" read as a
#       target, and the lead criterion was almost always a goal-directed one
#       already implied by the shortest-path constraint.
#   (b) INVERTED DIRECTION -- `evasive` proposed as observer_dist MIN.
def build_prompt(rule, known_rules, constrained, version=1):
    shown = episodes(SHOWN_STARTS)
    body = "\n".join(serialise_episode(s, d, c, v,
                                       trajectory_builtin(rule, s, d, c, v))
                     for s, d, c, v in shown)
    head = f"""\
A subject is navigating an 8x8 grid to one of three destinations. The grid (# is
a wall, G is a gate that can be open or closed, A/B/C are destinations):

{describe_grid()}

Moves are N (up), S (down), E (right), W (left).

The subject always takes a SHORTEST path to its destination. What differs
between subjects is WHICH shortest path they take when several are tied -- their
routing rule. The observer stands still at a stated square during each run.

Routing rules already in our vocabulary:
{chr(10).join(f"  - {r}: {G.RULE_GLOSS[r]}" for r in known_rules)}

None of those explains the trajectories below. Here is what this subject did:

{body}

Propose ONE new routing rule that explains this behaviour.
"""
    if version == 2:
        return head + (_V2_CONSTRAINED if constrained else _V2_FREEFORM)

    if constrained:
        opts = "\n".join(f"  - {k}: {v}" for k, v in CRITERION_GLOSS.items())
        return head + f"""
Express it as an ordered list of tie-break criteria. When several shortest steps
are tied, the first criterion is applied, then the second among whatever is still
tied, and so on.

Available criteria:
{opts}

Return ONLY a JSON list, for example:
  [{{"criterion": "openness", "direction": "min"}}]
"criterion" must be one of the names above; "direction" is "max" or "min".
At most 4 entries.
"""
    return head + """
Describe the rule in one or two plain English sentences -- the way you would
explain it to another person. Do not use JSON.
"""


_V2_CONSTRAINED = """
Express the rule as an ordered list of tie-break criteria. When several shortest
steps are tied, the first criterion is applied, then the second among whatever is
still tied, and so on.

Available criteria:
  - openness: how many of the four neighbours of the square are free
  - observer_dist: how far the square is from the observer, in steps
  - momentum: whether the step continues in the same direction as the last one
  - row: how far down the grid the square is
  - col: how far right the square is
  - goal_row_align: how close the square's row already is to the destination's row
  - goal_col_align: how close the square's column already is to the destination's column
"direction" is "max" or "min".

THREE THINGS THAT MATTER:

1. USE AS FEW CRITERIA AS POSSIBLE. Almost every rule of this kind needs exactly
   ONE. Adding criteria you do not need makes the rule WRONG, because each extra
   one changes which step is taken whenever the earlier ones tie. Do not pad the
   list.

2. THE SHORTEST PATH IS ALREADY ENFORCED. The subject is guaranteed to take a
   shortest route; you are only explaining how it chooses BETWEEN equally short
   steps. So criteria about getting closer to the destination
   (goal_row_align, goal_col_align) explain nothing and will make your answer
   wrong. Do not use them unless the behaviour cannot be explained otherwise.

3. GET THE DIRECTION RIGHT. "max" picks the LARGEST value, "min" the SMALLEST.
   Check yourself against one of the trajectories above before answering.

First write ONE sentence stating the rule in plain words, naming the direction
explicitly (for example: "prefers the square with FEWER free neighbours").
Then, on a new line, output ONLY the JSON list, for example:
  [{"criterion": "openness", "direction": "min"}]
"""

_V2_FREEFORM = """
Describe the rule in one or two plain English sentences -- the way you would
explain it to another person. Do not use JSON.

TWO THINGS THAT MATTER:

1. The subject is ALREADY guaranteed to take a shortest route. You are only
   explaining how it chooses BETWEEN equally short steps, so do not describe it
   as "heading toward the goal" -- that is true of every subject and explains
   nothing. Name the ONE thing that distinguishes this subject.

2. State the DIRECTION explicitly and unambiguously -- whether it prefers MORE
   or LESS of whatever it is you have identified. Check your sentence against
   one of the trajectories above before answering.
"""


# ---------------------------------------------------- free-form compiler
#
# Symbolic on purpose. A second LLM pass would put the model back in the trust
# path, which the architecture forbids, so the compile-failure rate measures how
# far prose sits from the formal representation -- which is the number we want.

_PHRASES = [
    (r"\b(hug|hugs|hugging|against|along|next to|close to|near)\b[^.;]{0,30}\b(wall|walls|edge|edges)\b",
     ("openness", "min")),
    (r"\b(narrow|enclosed|confined|tight|cramped|sheltered)\b", ("openness", "min")),
    (r"\b(open|openest|spacious|wide|roomy|uncluttered|centre|center|middle)\b",
     ("openness", "max")),
    (r"\b(away from|avoid|avoids|avoiding|far from|distance from|evade|evades|flee)\b"
     r"[^.;]{0,30}\b(observer|watcher|opponent|guard)\b", ("observer_dist", "max")),
    (r"\b(toward|towards|near|approach|close to)\b[^.;]{0,30}\b(observer|watcher|guard)\b",
     ("observer_dist", "min")),
    (r"\b(same direction|straight|keeps? going|continues?|momentum|without turning|"
     r"minimi[sz]e turns|fewest turns)\b", ("momentum", "max")),
    (r"\b(zigzag|zig-zag|alternates?|changes direction|turns often)\b", ("momentum", "min")),
    (r"\b(downwards?|southwards?|toward the bottom|lower rows?)\b", ("row", "max")),
    (r"\b(upwards?|northwards?|toward the top|higher rows?|upper rows?)\b", ("row", "min")),
    (r"\b(rightwards?|eastwards?|toward the right)\b", ("col", "max")),
    (r"\b(leftwards?|westwards?|toward the left)\b", ("col", "min")),
    (r"\b(match(es|ing)? the (destination|goal|target)'?s? row|correct row|right row first|"
     r"vertical(ly)? first|row first)\b", ("goal_row_align", "max")),
    (r"\b(match(es|ing)? the (destination|goal|target)'?s? column|correct column|"
     r"horizontal(ly)? first|column first)\b", ("goal_col_align", "max")),
]


def compile_prose(text):
    """Prose -> DSL spec. Order of first mention is the tie-break order."""
    t = " " + text.lower().replace("\n", " ") + " "
    hits = []
    for pat, pair in _PHRASES:
        m = re.search(pat, t)
        if m and pair not in [h[1] for h in hits]:
            hits.append((m.start(), pair))
    if not hits:
        return None, ("no tie-break criterion could be extracted from the prose "
                      "(no recognised phrase for any known criterion)")
    spec = [p for _, p in sorted(hits)][:4]
    return compile_spec([list(p) for p in spec])


# -------------------------------------------------------------- validation

def classify(spec, err, rule, known_rules):
    """ill_formed | inconsistent | duplicate | novel_valid, plus recovery."""
    if err:
        return "ill_formed", err, False
    shown = episodes(SHOWN_STARTS)
    held = episodes(HELD_STARTS)

    # duplicate: behaves identically to a rule already in the vocabulary
    for kr in known_rules:
        if all(trajectory_spec(spec, s, d, c, v) == trajectory_builtin(kr, s, d, c, v)
               for s, d, c, v in shown + held):
            return "duplicate", f"identical to known rule {kr!r}", False

    # inconsistent: does not reproduce what the LLM was actually shown
    bad = [e for e in shown
           if trajectory_spec(spec, *e) != trajectory_builtin(rule, *e)]
    if bad:
        return "inconsistent", f"fails {len(bad)}/{len(shown)} shown trajectories", False

    # THE TEST: episodes the LLM never saw
    miss = [e for e in held
            if trajectory_spec(spec, *e) != trajectory_builtin(rule, *e)]
    recovered = not miss
    note = ("recovers the held-out rule on all "
            f"{len(held)} unseen episodes" if recovered
            else f"consistent with what it saw, but fails {len(miss)}/{len(held)} unseen")
    return "novel_valid", note, recovered


# ------------------------------------------------------------------ runner

def run(ask, label, n=1, rules=None, styles=(True, False), version=1):
    """`ask(prompt, nonce) -> (text, truncated)`. Holds each rule out in turn.

    `n` is the number of independent draws per (held-out rule, prompt style).
    A single draw is not a measurement: these models sample, so the spread
    across draws is part of the result."""
    rules = rules or list(G.RULES)
    rows = []
    for rule in rules:
        known = [r for r in G.RULES if r != rule]
        for constrained in styles:
            for k in range(n):
                prompt = build_prompt(rule, known, constrained, version)
                text, truncated = ask(prompt, nonce=(version, k))
                if truncated:
                    rows.append((rule, constrained, "ill_formed",
                                 "response truncated", False))
                    continue
                if constrained:
                    m = re.search(r"\[.*\]", text, re.S)
                    if not m:
                        rows.append((rule, constrained, "ill_formed",
                                     "no JSON list in response", False))
                        continue
                    try:
                        spec, err = compile_spec(json.loads(m.group(0)))
                    except json.JSONDecodeError as e:
                        spec, err = None, f"JSON did not parse: {e}"
                else:
                    spec, err = compile_prose(text)
                cat, note, rec = classify(spec, err, rule, known)
                rows.append((rule, constrained, cat, note, rec))
    report(rows, label)
    return rows


CATS = ["ill_formed", "inconsistent", "duplicate", "novel_valid"]


def report(rows, label):
    print(f"\n{'=' * 76}\nHELD-OUT ROUTING-RULE RECOVERY -- {label}\n{'=' * 76}")
    for constrained in (True, False):
        sub = [r for r in rows if r[1] is constrained]
        if not sub:
            continue
        style = "CONSTRAINED (emit the DSL)" if constrained else "FREE-FORM (prose, symbolic compile)"
        n = len(sub)
        print(f"\n### {style}   (n = {n})\n")
        print(f"{'category':<16}{'count':>7}{'rate':>9}   {'95% Wilson CI':>18}")
        print("-" * 52)
        for cat in CATS:
            k = sum(1 for r in sub if r[2] == cat)
            p, lo, hi = P.wilson(k, n)
            print(f"{cat:<16}{k:>7}{p:>9.1%}   [{lo:>6.1%}, {hi:>6.1%}]")
        rec = sum(1 for r in sub if r[4])
        p, lo, hi = P.wilson(rec, n)
        print("-" * 52)
        print(f"{'RECOVERED':<16}{rec:>7}{p:>9.1%}   [{lo:>6.1%}, {hi:>6.1%}]"
              f"   <- predicts UNSEEN behaviour")
        print("\n  per held-out rule:")
        for rule in G.RULES:
            rr = [r for r in sub if r[0] == rule]
            if rr:
                print(f"    {rule:<12} {sum(1 for r in rr if r[4])}/{len(rr)} recovered"
                      f"   {rr[0][2]:<13} {rr[0][3][:52]}")


# ------------------------------------------------------------------ selftest
#
# The harness must be checked without an LLM, and separately from any claim
# about one. These are HARNESS TESTS, not results.

def selftest():
    print("\n## Harness self-test (no model involved)\n")
    ok = True

    print("1. Can the DSL express the built-in rules EXACTLY?")
    print("   If not, the held-out rule is unrecoverable by construction and the")
    print("   experiment measures nothing.\n")
    allep = episodes(SHOWN_STARTS) + episodes(HELD_STARTS)
    for rule, spec in BUILTIN_SPEC.items():
        bad = sum(1 for e in allep
                  if trajectory_spec(spec, *e) != trajectory_builtin(rule, *e))
        print(f"   {rule:<12} {len(allep) - bad}/{len(allep)} trajectories match"
              + ("" if not bad else "   MISMATCH"))
        ok &= bad == 0

    print("\n2. Do the episodes DISTINGUISH the four rules?")
    print("   If two rules produce identical trajectories on these episodes, no")
    print("   proposer could separate them and recovery would be unmeasurable.\n")
    for lab, eps in (("shown", episodes(SHOWN_STARTS)), ("held-back", episodes(HELD_STARTS))):
        sig = {}
        for r in G.RULES:
            sig.setdefault(tuple(t for e in eps for t in trajectory_builtin(r, *e)),
                           []).append(r)
        dup = [v for v in sig.values() if len(v) > 1]
        print(f"   {lab:<10} {len(sig)}/{len(G.RULES)} rules distinct over {len(eps)} episodes"
              + ("" if not dup else f"   INDISTINGUISHABLE: {dup}"))
        ok &= not dup

    print("\n3. An ORACLE proposer (emits the correct spec) must score 100% recovered.")
    for rule in G.RULES:
        known = [r for r in G.RULES if r != rule]
        cat, note, rec = classify(BUILTIN_SPEC[rule], None, rule, known)
        print(f"   {rule:<12} {cat:<12} recovered={rec}   {note}")
        ok &= rec and cat == "novel_valid"

    print("\n4. A NULL proposer (emits an empty rule) must NOT score as recovery,")
    print("   except where the empty rule genuinely is the held-out one.")
    for rule in G.RULES:
        known = [r for r in G.RULES if r != rule]
        cat, note, rec = classify([], None, rule, known)
        expected = (rule == "direct")
        flag = "" if rec == expected else "   UNEXPECTED"
        print(f"   {rule:<12} {cat:<12} recovered={rec}{flag}")
        ok &= rec == expected

    print("\n5. A MALFORMED proposal must be rejected, not silently accepted.")
    for bad in ("not a list", [{"criterion": "wingspan", "direction": "max"}],
                [{"criterion": "openness", "direction": "sideways"}]):
        spec, err = compile_spec(bad)
        cat, note, rec = classify(spec, err, "wall_hug", ["direct"])
        print(f"   {str(bad)[:44]:<46} -> {cat}")
        ok &= cat == "ill_formed"

    print("\n6. Free-form compiler on prose describing each rule. INFORMATIONAL:")
    print("   this measures the prose->DSL compiler, which is part of what the")
    print("   experiment reports, so it is not a pass/fail gate here.")
    prose = {
        "wall_hug":   "It hugs the walls, keeping to narrow squares wherever it can.",
        "open_field": "It stays in the most open part of the grid, avoiding tight corners.",
        "evasive":    "It keeps as far away from the observer as it can.",
        "direct":     "It just walks straight there with no particular preference.",
    }
    for rule, text in prose.items():
        spec, err = compile_prose(text)
        match = spec == BUILTIN_SPEC[rule] if spec is not None else False
        print(f"   {rule:<12} -> {str(spec):<34} "
              + ("matches" if match else f"NO ({err or 'different spec'})"))

    print("\n  " + ("HARNESS OK" if ok else "HARNESS BROKEN -- do not run the experiment"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--backend", default="none", choices=["none", "live"])
    ap.add_argument("--proposer", default="anthropic", choices=sorted(P.PROVIDERS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=1, help="proposals per (rule, style)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    if args.show_prompt:
        print(build_prompt("wall_hug", [r for r in G.RULES if r != "wall_hug"], True))
        return
    if args.selftest or args.backend == "none":
        selftest()
        if args.backend == "none":
            print("\nNo model run. To measure recovery with a real proposer:")
            print("  export ANTHROPIC_API_KEY=...   # or GEMINI_API_KEY; ollama needs none")
            print("  python3 vocab.py --backend live --proposer anthropic --n 3")
            print("  python3 vocab.py --backend live --proposer ollama --model llama3.1 --n 3")
            print("\nNO RECOVERY RATE IS REPORTED HERE. The harness is validated above;")
            print("the proposer is not yet measured.")
        return

    prov = P.PROVIDERS[args.proposer](args.model or P.DEFAULT_MODEL[args.proposer],
                                      use_cache=not args.no_cache)
    run(prov.ask, f"{args.proposer}/{prov.model}", args.n)


if __name__ == "__main__":
    main()
