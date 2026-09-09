#!/usr/bin/env python3
"""
Sweeps and hardening for the cost-aware adversary / observer.

    python3 sweep.py verify        # task 1: Kuhn greedy == lookahead, every metric
    python3 sweep.py adversary     # task 2: concealment bought per chip (lam sweep)
    python3 sweep.py observer      # task 3: identification kept per chip (mu sweep)
    python3 sweep.py harden        # misID confidence interval, contradiction timing
    python3 sweep.py deception     # task 4c: deception-aware observer, and what it costs
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


# ------------------------------------------------- task 4c: deception-aware
#
# Two mechanisms, each with a measurement that would show it is NOT working.
#
#   M1  play for chips on refutation. When the hypothesis set empties there is
#       no model left to plan against, so fall back to exact chip-optimal play
#       under a no-model assumption.
#       Falsifier: chips/hand on exactly the hands that refute before an
#       observer decision remains must improve. If it does not, M1 is inert.
#       This falsifier has already earned its keep: the first version of M1
#       FOLDED on refutation, and the measurement caught it losing 6.1 chips a
#       hand. See RESULTS.md.
#
#   M2  cross-hand abstention. An empty set is PROOF that the subject is not
#       faithful to any intent -- which is the premise every elimination
#       conclusion rests on. After k such proofs, stop issuing single-intent
#       conclusions and report "contradicted" instead.
#       Falsifier: against a FAITHFUL subject the flag must never fire, so
#       every faithful number must be unchanged for every k. If a faithful
#       number moves, the trigger is reading something it should not.
#
# Note what M2 is NOT. Within a single hand, contradiction and misattribution
# are disjoint by construction (the set only shrinks, so a hand that empties
# ends empty; misID means it never emptied). So no within-hand rule can convert
# a misattribution into an abstention -- the signal simply is not there in the
# hand that goes wrong. It is there ACROSS hands, and that is the only place
# the conversion can come from.

def _run_state(game, condition, subject, seed, mu=0.0, lam=0.0, deception_aware=0):
    """Like run(), but threads refutation state across the hands of a subject."""
    rng = random.Random(seed)
    mod = GAMES[game]
    state = {"refutations": 0}
    return [mod.play_hand(condition, rng, hand_no=k + 1, subject=subject, mu=mu, lam=lam,
                          deception_aware=deception_aware, state=state)
            for k in range(HANDS)]


def _report_stats(rows):
    n = len(rows)
    return {
        "exact": sum(r["report_exact"] for r in rows) / n,
        "misID": sum(r["report_misID"] for r in rows) / n,
        "abstain": sum(r["reported"] == "contradicted" for r in rows) / n,
        "withheld": sum(r["abstained"] for r in rows) / n,
        "H": sum(r["final_size"] for r in rows) / n,
        "chips": sum(r["observer_chips"] for r in rows) / n,
    }


def _mean_range(per, keys):
    return {k: (sum(p[k] for p in per) / len(per), min(p[k] for p in per), max(p[k] for p in per))
            for k in keys}


def deception():
    print("\n## Task 4c: deception-aware observer\n")

    # ---------------------------------------------------------------- M1
    print("### M1: play for chips on refutation")
    print("Measured on exactly the hands where the set empties while an observer")
    print("decision still remains. Everything before that point is identical, so")
    print("the same hands qualify under both observers -- a paired comparison.\n")
    hdr = f"{'subject / lam':<28} {'hands':>7} {'chips/hand passive':>20} {'chips/hand no-model':>21} {'delta':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, subj, lam in [("refute lam=0", "adversarial:refute", 0.0),
                             ("refute lam=2", "adversarial:refute", 2.0),
                             ("impersonate lam=2", "adversarial:impersonate", 2.0)]:
        off_n = off_c = on_n = on_c = 0
        for s in SEEDS:
            off = _run_state("leduc", "adaptive", subj, s, lam=lam, deception_aware=0)
            on = _run_state("leduc", "adaptive", subj, s, lam=lam, deception_aware=1)
            o = [r for r in off if r["observer_moves_after_contradiction"] > 0]
            n_ = [r for r in on if r["observer_moves_after_contradiction"] > 0]
            off_n += len(o); off_c += sum(r["observer_chips"] for r in o)
            on_n += len(n_); on_c += sum(r["observer_chips"] for r in n_)
        a = off_c / max(1, off_n)
        b = on_c / max(1, on_n)
        flag = "" if off_n == on_n else f"  (!! hand counts differ: {off_n} vs {on_n})"
        print(f"{label:<28} {off_n:>7} {a:>+20.3f} {b:>+21.3f} {b - a:>+8.3f}{flag}")

    # ---------------------------------------------------------------- M2
    print("\n### M2: cross-hand abstention -- the confident-wrong / known-unknown trade")
    print("k = report 'contradicted' instead of a single intent once the subject has")
    print("refuted the model in k prior hands. k=0 is the plain observer.")
    print("Cells: mean over seeds 1-3 [min, max], n = 2000 hands per seed (6000 total).\n")

    KS = [0, 1, 2, 3, 5]
    for label, subj, lam in [("faithful (control)", "faithful", 0.0),
                             ("refute, lam=0", "adversarial:refute", 0.0),
                             ("chip maximiser, lam=2", "adversarial:refute", 2.0),
                             ("impersonate, lam=2", "adversarial:impersonate", 2.0)]:
        print(f"#### {label}   (n = 6000)")
        hdr = (f"{'k':<5} {'exact ID (report)':>26} {'misID (report)':>26} "
               f"{'abstain':>26} {'chips/hand':>24}")
        print(hdr)
        print("-" * len(hdr))
        base = None
        for k in KS:
            per = [_report_stats(_run_state("leduc", "adaptive", subj, s, lam=lam,
                                            deception_aware=k)) for s in SEEDS]
            m = _mean_range(per, ["exact", "misID", "abstain", "chips"])
            if k == 0:
                base = m
            print(f"{k:<5} {fmt(m['exact']):>26} {fmt(m['misID']):>26} "
                  f"{fmt(m['abstain']):>26} {fmt(m['chips'], pct=False, signed=True):>24}")
        d_mis = base["misID"][0] - m["misID"][0]
        d_ex = base["exact"][0] - m["exact"][0]
        ratio = (d_mis / d_ex) if d_ex > 1e-9 else float("inf")
        print(f"      k=0 -> k=5: misID {base['misID'][0]:.1%} -> {m['misID'][0]:.1%} "
              f"(-{d_mis:.1%}), exact ID {base['exact'][0]:.1%} -> {m['exact'][0]:.1%} "
              f"(-{d_ex:.1%});  " +
              (f"{ratio:.1f} confident-wrong removed per confident-right given up"
               if ratio != float("inf") else "no confident-right given up"))
        print()

    # ------------------------------------------------- how fast does it fire
    print("### How fast does the flag fire? (hands until the subject first refutes)")
    hdr = f"{'subject / lam':<28} {'median':>8} {'mean':>8} {'never refuted (of 3 seeds)':>28}"
    print(hdr)
    print("-" * len(hdr))
    for label, subj, lam in [("faithful", "faithful", 0.0),
                             ("refute lam=0", "adversarial:refute", 0.0),
                             ("chip maximiser lam=2", "adversarial:refute", 2.0),
                             ("impersonate lam=2", "adversarial:impersonate", 2.0)]:
        firsts, never = [], 0
        for s in SEEDS:
            rows = _run_state("leduc", "adaptive", subj, s, lam=lam, deception_aware=1)
            idx = next((i + 1 for i, r in enumerate(rows) if r["contradiction"]), None)
            if idx is None:
                never += 1
            else:
                firsts.append(idx)
        if firsts:
            srt = sorted(firsts)
            med = srt[len(srt) // 2]
            print(f"{label:<28} {med:>8} {sum(firsts) / len(firsts):>8.1f} {never:>28}")
        else:
            print(f"{label:<28} {'-':>8} {'-':>8} {never:>28}")


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
    if what in ("deception", "all"):
        deception()
