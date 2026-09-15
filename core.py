#!/usr/bin/env python3
"""
Shared machinery for every environment.

The claim the project makes is about a MECHANISM -- maintain an explicit
hypothesis set over intents-as-policies, eliminate what the observed behaviour
rules out, and choose your own actions to shrink what is left. If that claim is
real it should not care whether the environment is a poker tree or a grid. So
the metric definitions, the seed aggregation and the table rendering live here
once, and each environment supplies only its own game logic.

What is genuinely shared: the per-episode record schema, every metric computed
from it, Wilson intervals, seed aggregation, table rendering.

What is NOT shared, and why: the elimination step and the selection rule. Both
are one-liners over an environment-specific policy function and action set
(`policy(intent, state) -> action`, `legal(state)`), and the recursion that
scores actions has to walk the environment's own tree. Abstracting those behind
a common interface would have meant passing four callbacks into a generic
recursion -- more indirection than the six lines it would save, and it would
have obscured the one thing a reader needs to check, which is that the two
environments really do apply the same rule. They are written out in each
environment and are three lines apart in shape; `sweep.py verify` is what
holds them honest.
"""

import math

# --------------------------------------------------------------- the record
#
# Every environment emits one of these per episode (per hand, per navigation
# run). Field names are shared so the metrics below are environment-agnostic.
#
#   declared          the intent the subject declared, before acting
#   final_set         sorted list of intents still standing at the end
#   final_size        len(final_set)
#   sound             declared intent was never wrongly eliminated
#   exact             final_set == [declared]
#   misattributed     final_set is exactly one intent, and it is the wrong one
#   contradiction     final_set is empty: no intent explains the behaviour
#   subject_actions   decisions the subject made
#   forced_decisions  decisions the subject made ONLY because the observer
#                     acted -- the probe count; 0 for a passive observer
#   deviations        subject moves that departed from the declared policy
#   observer_cost     observer's net cost (chips lost / actions paid for);
#                     sign convention is per environment, stated in its table

METRICS = {
    "exact":    ("exact ID", "pct", lambda r: r.get("report_exact", r["exact"])),
    "H":        ("|H| final", "num", lambda r: r["final_size"]),
    "sound":    ("sound", "pct", lambda r: r["sound"]),
    "misID":    ("misID", "pct", lambda r: r.get("report_misID", r["misattributed"])),
    "contra":   ("contra", "pct", lambda r: r.get("contradiction", False)),
    "forced":   ("forced/ep", "num", lambda r: r.get("forced_decisions", 0)),
    "cost":     ("cost/ep", "signed", lambda r: r.get("observer_cost", r.get("observer_chips", 0))),
}


def summarise(rows, keys=None):
    """Mean of each requested metric over a list of episode records."""
    keys = keys or list(METRICS)
    n = len(rows)
    return {k: sum(METRICS[k][2](r) for r in rows) / n for k in keys}


def aggregate(per_seed):
    """mean, min, max of each metric across seeds."""
    return {k: (sum(p[k] for p in per_seed) / len(per_seed),
                min(p[k] for p in per_seed), max(p[k] for p in per_seed))
            for k in per_seed[0]}


def fmt_cell(v, kind, multi):
    m, lo, hi = v
    f = (lambda x: f"{x:.1%}") if kind == "pct" else \
        (lambda x: f"{x:.2f}") if kind == "num" else (lambda x: f"{x:+.3f}")
    return f"{f(m)} [{f(lo)},{f(hi)}]" if multi else f(m)


def print_table(results, keys, seeds, episodes, label="condition", notes=(), total_n=None):
    """results: {row label -> {metric -> (mean, min, max)}}.

    `total_n` overrides the default episodes x seeds, for environments whose
    episode set is enumerated exhaustively rather than sampled per seed."""
    multi = len(seeds) > 1
    w = 24 if multi else 11
    lw = max(len(label), max((len(k) for k in results), default=8)) + 2
    hdr = f"{label:<{lw}} {'n':>7} " + " ".join(f"{METRICS[k][0]:>{w}}" for k in keys)
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, s in results.items():
        n = total_n if total_n is not None else episodes * len(seeds)
        print(f"{name:<{lw}} {n:>7} " +
              " ".join(f"{fmt_cell(s[k], METRICS[k][1], multi):>{w}}" for k in keys))
    print()
    if total_n is not None:
        print(f"n          = {total_n} episodes per cell, enumerated exhaustively"
              + ("; cells show mean [min, max] across seeds " + str(seeds) if multi else ""))
    else:
        print(f"n          = episodes per cell ({episodes} x {len(seeds)} seed(s): "
              f"{', '.join(map(str, seeds))})"
              + ("; cells show mean [min, max] across seeds" if multi else ""))
    for line in notes:
        print(line)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h
