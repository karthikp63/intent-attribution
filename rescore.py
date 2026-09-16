#!/usr/bin/env python3
"""
Re-score every cached proposal with a graded metric alongside the binary one.

WHY. "Recovered" means the proposed rule predicts EVERY move across 36 unseen
episodes exactly. One wrong move out of hundreds scores as total failure, so a
33% recovery rate cannot distinguish "clueless two thirds of the time" from
"close nearly always, exact a third of the time". Those are different findings.

The binary metric is NOT replaced -- both are reported, so the committed numbers
stay comparable.

THE FLOOR MATTERS MORE THAN THE SCORE. Most steps have exactly one shortest
option, so every rule agrees on them for free. The EMPTY spec -- no rule at all
-- already scores 85.7-92.1% raw agreement. Any graded number must be read
against that, which is why CONTESTED agreement (only states with two or more
shortest steps, where a routing rule has any content) is the one to look at.

Uses cached responses only; makes no API calls.

    python3 rescore.py
"""

import glob
import json
import re
import statistics as st

import gridworld as G
import vocab as V

HELD = V.episodes(V.HELD_STARTS)


def load():
    """-> rows of (model, version, style, rule, spec_or_None, category)."""
    out = []
    for f in sorted(glob.glob("fixtures/cache/*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        p, model = d.get("prompt", ""), d.get("model", "?")
        if "routing rule" not in p:
            continue
        constrained = ("JSON list" in p) or ("ordered list of tie-break" in p)
        version = 2 if ("USE AS FEW CRITERIA AS POSSIBLE" in p
                        or "TWO THINGS THAT MATTER" in p) else 1
        held = [r for r in G.RULES if f"  - {r}:" not in p]
        if len(held) != 1:
            continue
        rule = held[0]
        if d.get("truncated"):
            out.append((model, version, "constrained" if constrained else "freeform",
                        rule, None, "ill_formed"))
            continue
        if constrained:
            m = re.search(r"\[.*\]", d["text"], re.S)
            if not m:
                spec, err = None, "no JSON list"
            else:
                try:
                    spec, err = V.compile_spec(json.loads(m.group(0)))
                except json.JSONDecodeError as e:
                    spec, err = None, str(e)
        else:
            spec, err = V.compile_prose(d["text"])
        cat, _, _ = V.classify(spec, err, rule,
                               [r for r in G.RULES if r != rule])
        out.append((model, version, "constrained" if constrained else "freeform",
                    rule, None if err else spec, cat))
    return out


def graded(spec, rule):
    if spec is None:
        return None
    h, t, ch, ct = V.move_agreement(spec, rule, HELD)
    return h / t, (ch / ct if ct else 1.0)


def bucket(xs, edges=(0.25, 0.5, 0.75, 0.9, 0.999)):
    labs = ["<25%", "25-50", "50-75", "75-90", "90-<100", "100%"]
    c = [0] * len(labs)
    for x in xs:
        for i, e in enumerate(edges):
            if x < e:
                c[i] += 1
                break
        else:
            c[-1] += 1
    return labs, c


def report(rows):
    print("=" * 78)
    print("GRADED RE-SCORE -- binary recovery alongside per-move agreement")
    print("=" * 78)

    print("\n## Calibration: what a NON-answer already scores\n")
    print(f"  {'held-out rule':<14}{'empty spec, raw':>18}{'empty spec, contested':>24}")
    for r in G.RULES:
        h, t, ch, ct = V.move_agreement([], r, HELD)
        print(f"  {r:<14}{h/t:>17.1%}{ch/ct:>24.1%}")
    print("\n  The empty spec is 'shortest path, no rule at all'. Raw agreement")
    print("  cannot go below ~86%, so only CONTESTED agreement discriminates.")
    print("  (`direct` IS the empty spec, so its floor is 100% by definition and")
    print("   its graded score carries no information.)")

    keys = sorted({(m, v, s) for m, v, s, _, _, _ in rows})
    for model, ver, style in keys:
        sub = [r for r in rows if (r[0], r[1], r[2]) == (model, ver, style)]
        if len(sub) < 20:
            continue
        gs = [(r[3], graded(r[4], r[3]), r[5]) for r in sub]
        comp = [(rule, g, c) for rule, g, c in gs if g is not None]
        rec = sum(1 for r in sub if r[5] == "novel_valid")
        print(f"\n## {model}  prompt-v{ver}  {style}   n = {len(sub)}\n")
        print(f"  BINARY recovery          {rec}/{len(sub)} = {rec/len(sub):.1%}")
        if not comp:
            print("  (no proposal compiled; no graded score)")
            continue
        raw = [g[0] for _, g, _ in comp]
        con = [g[1] for _, g, _ in comp]
        print(f"  compiled                 {len(comp)}/{len(sub)}")
        print(f"  GRADED raw agreement     mean {st.mean(raw):.1%}  "
              f"median {st.median(raw):.1%}")
        print(f"  GRADED contested         mean {st.mean(con):.1%}  "
              f"median {st.median(con):.1%}")
        labs, c = bucket(con)
        print("  contested distribution   " +
              "  ".join(f"{l}:{n}" for l, n in zip(labs, c)))
        print(f"\n  {'rule':<12}{'n':>4}{'binary':>9}{'contested mean':>17}{'floor':>8}")
        for r in G.RULES:
            rr = [x for x in sub if x[3] == r]
            cc = [g[1] for rule, g, _ in comp if rule == r]
            if not rr:
                continue
            b = sum(1 for x in rr if x[5] == "novel_valid") / len(rr)
            _, _, fh, ft = V.move_agreement([], r, HELD)
            print(f"  {r:<12}{len(rr):>4}{b:>9.1%}"
                  + (f"{st.mean(cc):>17.1%}" if cc else f"{'-':>17}")
                  + f"{fh/ft:>8.1%}")


if __name__ == "__main__":
    report(load())
