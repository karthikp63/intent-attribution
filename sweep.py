#!/usr/bin/env python3
"""
Sweeps and hardening for the cost-aware adversary / observer.

    python3 sweep.py verify        # task 1: Kuhn greedy == lookahead, every metric
    python3 sweep.py adversary     # task 2: concealment bought per chip (lam sweep)
    python3 sweep.py observer      # task 3: identification kept per chip (mu sweep)
    python3 sweep.py harden        # misID confidence interval, contradiction timing
    python3 sweep.py all

Every cell: 2000 hands per seed, seeds 1-3; table shows the seed mean and,
in brackets, the min-max range. Stdlib only.
"""

import math
import random
import sys

import kuhn_intent as K
import leduc_intent as L

HANDS = 2000
SEEDS = [1, 2, 3]
GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1000.0]   # 1000 ~ "chips only"
GAMES = {"kuhn": K, "leduc": L}


def run(game, condition, subject, seed, mu=0.0, lam=0.0):
    rng = random.Random(seed)
    kw = {"observer": "lookahead"} if game == "kuhn" else {}
    mod = GAMES[game]
    return [mod.play_hand(condition, rng, hand_no=k + 1, subject=subject, mu=mu, lam=lam, **kw)
            for k in range(HANDS)]


def stats(rows):
    n = len(rows)
    acts = sum(r["subject_actions"] for r in rows)
    return {
        "exact": sum(r["exact"] for r in rows) / n,
        "H": sum(r["final_size"] for r in rows) / n,
        "sound": sum(r["sound"] for r in rows) / n,
        "misID": sum(r["misattributed"] for r in rows) / n,
        "contra": sum(r.get("contradiction", False) for r in rows) / n,
        "deviate": sum(r["deviations"] for r in rows) / acts,
        "chips": sum(r["observer_chips"] for r in rows) / n,
    }


def over_seeds(game, condition, subject, mu=0.0, lam=0.0):
    per = [stats(run(game, condition, subject, s, mu, lam)) for s in SEEDS]
    return {k: (sum(p[k] for p in per) / len(per), min(p[k] for p in per), max(p[k] for p in per))
            for k in per[0]}


def fmt(v, pct=True, signed=False):
    m, lo, hi = v
    if pct:
        return f"{m:6.1%} [{lo:.1%},{hi:.1%}]"
    f = "{:+.3f}" if signed else "{:.2f}"
    return f"{f.format(m):>6} [{f.format(lo)},{f.format(hi)}]"


def table(title, rows):
    cols = ["exact", "H", "sound", "misID", "contra", "deviate", "chips"]
    print(f"\n### {title}\n")
    print(f"{'setting':<14}" + "".join(f"{c:>22}" for c in cols))
    print("-" * (14 + 22 * len(cols)))
    for label, s in rows:
        line = f"{label:<14}"
        for c in cols:
            line += f"{fmt(s[c], pct=c not in ('H', 'chips'), signed=c == 'chips'):>22}"
        print(line)


# ------------------------------------------------------------------ task 2

def sweep_adversary():
    print("\n## Task 2: cost-aware adversary vs. the adaptive observer (mu = 0)")
    print("adversary maximises E[final |intent set|] - lam * E[chips lost]; "
          "chips = observer's profit per hand")
    for game, subjects in [("kuhn", ["adversarial"]),
                           ("leduc", ["adversarial:impersonate", "adversarial:refute"])]:
        for subj in subjects:
            rows = [("faithful", over_seeds(game, "adaptive", "faithful"))]
            for lam in GRID:
                rows.append((f"lam={lam:g}", over_seeds(game, "adaptive", subj, lam=lam)))
            table(f"{game} / {subj}", rows)


# ------------------------------------------------------------------ task 3

def sweep_observer():
    print("\n## Task 3: cost-aware observer vs. the faithful subject")
    print("observer minimises E[final |intent set|] - mu * E[chips]")
    for game in GAMES:
        rows = [("passive", over_seeds(game, "passive", "faithful"))]
        for mu in GRID:
            rows.append((f"mu={mu:g}", over_seeds(game, "adaptive", "faithful", mu=mu)))
        table(f"{game} / faithful subject", rows)
    print("\n### cross-check: cost-aware observer vs. the pure concealer (lam = 0), leduc")
    for subj in ["adversarial:impersonate", "adversarial:refute"]:
        rows = [(f"mu={mu:g}", over_seeds("leduc", "adaptive", subj, mu=mu))
                for mu in [0.0, 0.25, 1.0]]
        table(f"leduc / {subj}", rows)


# ---------------------------------------------------------------- hardening

# --------------------------------------------------------------- task 1: gate
#
# Kuhn has ONE betting round, so there is no "later" for the lookahead observer
# to look ahead to: at every observer decision the remaining subtree is at most
# one subject reply plus the showdown, which is exactly what the one-step greedy
# rule already scores. The two rules are therefore the same computation and must
# agree hand for hand. This is a gate, not a nicety -- every result below assumes
# both observers are correct, and the Leduc observer is only a generalisation of
# the Kuhn one if this passes.
#
# Compared per hand, not just in aggregate: two different rules can produce
# identical means over 2000 hands while disagreeing on individual hands, so an
# aggregate-only check would not catch a real divergence.

# Every field of a per-hand record except the tag naming which rule produced it.
_IGNORE = {"observer"}

METRICS = ["exact", "H", "sound", "misID", "contra", "deviate", "chips", "reveal"]


def _metrics(rows):
    s = stats(rows)
    s["reveal"] = sum(r["card_revealed"] for r in rows) / len(rows)
    return s


def verify():
    print("\n## Task 1 gate: Kuhn greedy vs. exact lookahead\n")
    print("One betting round => no lookahead horizon => the two rules are the same")
    print("computation. Compared per hand (not just in aggregate) over every cell.\n")

    hands = 0
    bad_records, bad_metrics = [], []
    empty_sets = 0

    hdr = f"{'cell':<34} {'hands':>6} {'records differing':>18} {'metrics differing':>18}"
    print(hdr)
    print("-" * len(hdr))

    for subject in ["faithful", "adversarial"]:
        for cond in ["passive", "random", "adaptive"]:
            for seed in SEEDS:
                g = [K.play_hand(cond, random.Random(seed), hand_no=k + 1,
                                 subject=subject, observer="greedy")
                     for k in range(HANDS)]
                l = [K.play_hand(cond, random.Random(seed), hand_no=k + 1,
                                 subject=subject, observer="lookahead")
                     for k in range(HANDS)]

                diffs = 0
                for a, b in zip(g, l):
                    fa = {k: v for k, v in a.items() if k not in _IGNORE}
                    fb = {k: v for k, v in b.items() if k not in _IGNORE}
                    if fa != fb:
                        diffs += 1
                        if len(bad_records) < 5:
                            bad_records.append((cond, subject, seed, fa, fb))

                ma, mb = _metrics(g), _metrics(l)
                mdiff = [k for k in METRICS if abs(ma[k] - mb[k]) > 1e-12]
                if mdiff:
                    bad_metrics.append((cond, subject, seed, mdiff, ma, mb))

                # Kuhn should never refute the model: justifies the absence of
                # the impersonate/refute split that Leduc needs.
                empty_sets += sum(1 for r in g + l if r["final_size"] == 0)

                hands += len(g)
                cell = f"{cond}/{subject} seed {seed}"
                print(f"{cell:<34} {len(g):>6} {diffs:>18} {len(mdiff):>18}")

    print(f"\ntotal hands compared: {hands} per rule ({2 * hands} played)")
    print(f"per-hand records differing: {sum(1 for _ in bad_records) if bad_records else 0}"
          f"{' (first 5 shown below)' if bad_records else ''}")
    print(f"aggregate metrics differing: {len(bad_metrics)} cells")
    print(f"hands ending with an EMPTY hypothesis set: {empty_sets} "
          f"(Kuhn cannot refute the model, so impersonate/refute do not arise -- "
          f"the split exists only in Leduc)")

    for cond, subject, seed, fa, fb in bad_records:
        print(f"\n  DIVERGENCE {cond}/{subject} seed {seed}")
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k) != fb.get(k):
                print(f"    {k}: greedy={fa.get(k)!r}  lookahead={fb.get(k)!r}")
    for cond, subject, seed, mdiff, ma, mb in bad_metrics:
        print(f"\n  METRIC DIVERGENCE {cond}/{subject} seed {seed}")
        for k in mdiff:
            print(f"    {k}: greedy={ma[k]!r}  lookahead={mb[k]!r}")

    ok = not bad_records and not bad_metrics
    print("\nRESULT: " + ("PASS -- identical hand for hand and metric for metric."
                          if ok else
                          "FAIL -- divergence found. STOP; one implementation is wrong."))
    return ok


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def harden():
    print("\n## Hardening: misattribution under the refute adversary (leduc, adaptive, mu=lam=0)\n")
    pooled = []
    for s in SEEDS:
        rows = run("leduc", "adaptive", "adversarial:refute", s)
        pooled += rows
        k = sum(r["misattributed"] for r in rows)
        p, lo, hi = wilson(k, len(rows))
        print(f"seed {s}: misID {k}/{len(rows)} = {p:.1%}   95% Wilson CI [{lo:.1%}, {hi:.1%}]")
    k = sum(r["misattributed"] for r in pooled)
    p, lo, hi = wilson(k, len(pooled))
    print(f"pooled: misID {k}/{len(pooled)} = {p:.1%}   95% Wilson CI [{lo:.1%}, {hi:.1%}]")
    kc = sum(r["contradiction"] for r in pooled)
    p, lo, hi = wilson(kc, len(pooled))
    print(f"pooled: contradiction {kc}/{len(pooled)} = {p:.1%}   95% Wilson CI [{lo:.1%}, {hi:.1%}]")

    print("\n### Do contradiction and misattribution co-occur within a hand?")
    both = [r for r in pooled if r["contradiction"] and r["misattributed"]]
    print(f"hands with both: {len(both)}  (structurally impossible: the hypothesis set only "
          f"shrinks, so a hand that reaches the empty set ends empty; misID means it never did)")

    print("\n### Is the contradiction visible before the observer commits?")
    contra = [r for r in pooled if r["contradiction"]]
    early = [r for r in contra if r["observer_moves_after_contradiction"] > 0]
    print(f"contradiction hands: {len(contra)}")
    print(f"  H became empty BEFORE at least one observer decision: {len(early)} "
          f"({len(early) / len(contra):.1%})")
    at_show = [r for r in contra if r["contradiction_at"] == "showdown"]
    print(f"  H became empty at the subject's last move (no observer decision left): "
          f"{len(contra) - len(early) - len(at_show)}")
    print(f"  H became empty only at showdown (play fit an intent; the card fit none):   "
          f"{len(at_show)} ({len(at_show) / len(contra):.1%})")
    at = {}
    for r in contra:
        at[str(r["contradiction_at"])] = at.get(str(r["contradiction_at"]), 0) + 1
    print("  where H emptied (history index or showdown) -> count: " +
          ", ".join(f"{i}:{c}" for i, c in sorted(at.items())))
    chips_after = sum(r["observer_chips"] for r in early) / max(1, len(early))
    print(f"  observer chips/hand in those {len(early)} hands (it currently just plays passive "
          f"after the model is refuted): {chips_after:+.3f}")

    print("\n### Misattributed hands: what did the observer see?")
    mis = [r for r in pooled if r["misattributed"]]
    pairs = {}
    for r in mis:
        key = (r["declared"], r["final_set"][0])
        pairs[key] = pairs.get(key, 0) + 1
    for (d, f), c in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print(f"  declared {d:<12} -> observer concluded {f:<12} x{c}")
    hists = {}
    for r in mis:
        hists[tuple(r["history"])] = hists.get(tuple(r["history"]), 0) + 1
    print("  most common histories:")
    for h, c in sorted(hists.items(), key=lambda kv: -kv[1])[:5]:
        print(f"    {' '.join(h):<60} x{c}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("verify", "all"):
        if not verify() and what == "all":
            sys.exit("gate failed; not running the sweeps")
    if what in ("adversary", "all"):
        sweep_adversary()
    if what in ("observer", "all"):
        sweep_observer()
    if what in ("harden", "all"):
        harden()
