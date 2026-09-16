#!/usr/bin/env python3
"""
Sweeps and hardening for the cost-aware misfit generator / observer.

    python3 sweep.py verify        # task 1: Kuhn greedy == lookahead, every metric
    python3 sweep.py misfit        # task 2: concealment bought per chip (lam sweep)
    python3 sweep.py observer      # task 3: identification kept per chip (mu sweep)
    python3 sweep.py harden        # misID confidence interval, contradiction timing
    python3 sweep.py deception     # task 4c: deception-aware observer, and what it costs
    python3 sweep.py misattribute  # the consistent misattributor: stays in-model, still wrong
    python3 sweep.py grid          # gridworld: online probing vs environment design (GRD)
    python3 soft.py --sweep        # soft elimination: noisy subjects
    python3 sweep.py all

Every cell: 2000 hands per seed, seeds 1-3; table shows the seed mean and,
in brackets, the min-max range. Stdlib only.
"""

import math
import random
import sys

import core
import gridworld as GW
import soft
import glosses
import pilot
import vocab
import kuhn_intent as K
import leduc_intent as L

HANDS = 2000
SEEDS = [1, 2, 3]
GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1000.0]   # 1000 ~ "chips only"
GAMES = {"kuhn": K, "leduc": L}


def run(game, condition, subject, seed, mu=0.0, lam=0.0):
    rng = random.Random(seed)                   # deals + declarations
    orng = random.Random(seed + 1_000_000)      # observer's own randomness
    kw = {"observer": "lookahead"} if game == "kuhn" else {}
    mod = GAMES[game]
    return [mod.play_hand(condition, rng, hand_no=k + 1, subject=subject, mu=mu, lam=lam,
                          orng=orng, **kw)
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
    n = HANDS * len(SEEDS)
    print(f"\n### {title}   (n = {n} per row: {HANDS} hands x {len(SEEDS)} seeds "
          f"{tuple(SEEDS)}; cells are mean [min, max])\n")
    print(f"{'setting':<14}" + "".join(f"{c:>22}" for c in cols))
    print("-" * (14 + 22 * len(cols)))
    for label, s in rows:
        line = f"{label:<14}"
        for c in cols:
            line += f"{fmt(s[c], pct=c not in ('H', 'chips'), signed=c == 'chips'):>22}"
        print(line)


# ------------------------------------------------------------------ task 2

def sweep_misfit():
    print("\n## Task 2: cost-aware misfit generator vs. the adaptive observer (mu = 0)")
    print("misfit generator maximises E[final |intent set|] - lam * E[chips lost]; "
          "chips = observer's profit per hand")
    for game, subjects in [("kuhn", ["misfit"]),
                           ("leduc", ["misfit:in_model", "misfit:out_of_model"])]:
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
    for subj in ["misfit:in_model", "misfit:out_of_model"]:
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

    for subject in ["faithful", "misfit"]:
        for cond in ["passive", "random", "adaptive"]:
            for seed in SEEDS:
                rg, og = random.Random(seed), random.Random(seed + 1_000_000)
                g = [K.play_hand(cond, rg, hand_no=k + 1, subject=subject,
                                 observer="greedy", orng=og) for k in range(HANDS)]
                rl, ol = random.Random(seed), random.Random(seed + 1_000_000)
                l = [K.play_hand(cond, rl, hand_no=k + 1, subject=subject,
                                 observer="lookahead", orng=ol) for k in range(HANDS)]

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
                # the in_model / out_of_model split that Leduc needs.
                empty_sets += sum(1 for r in g + l if r["final_size"] == 0)

                hands += len(g)
                cell = f"{cond}/{subject} seed {seed}"
                print(f"{cell:<34} {len(g):>6} {diffs:>18} {len(mdiff):>18}")

    print(f"\ntotal hands compared: {hands} per rule ({2 * hands} played)")
    print(f"per-hand records differing: {sum(1 for _ in bad_records) if bad_records else 0}"
          f"{' (first 5 shown below)' if bad_records else ''}")
    print(f"aggregate metrics differing: {len(bad_metrics)} cells")
    print(f"hands ending with an EMPTY hypothesis set: {empty_sets} "
          f"(sampled; see the exhaustive proof below)")

    for cond, subject, seed, fa, fb in bad_records:
        print(f"\n  DIVERGENCE {cond}/{subject} seed {seed}")
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k) != fb.get(k):
                print(f"    {k}: greedy={fa.get(k)!r}  lookahead={fb.get(k)!r}")
    for cond, subject, seed, mdiff, ma, mb in bad_metrics:
        print(f"\n  METRIC DIVERGENCE {cond}/{subject} seed {seed}")
        for k in mdiff:
            print(f"    {k}: greedy={ma[k]!r}  lookahead={mb[k]!r}")

    ok = ((not bad_records) and (not bad_metrics) and kuhn_coverage()
          and grid_selfcheck() and soft.verify(hands=120)
          and vocab.selftest() and pilot.check() and glosses.check()
          and document_check())
    print("\nRESULT: " + ("PASS -- identical hand for hand and metric for metric."
                          if ok else
                          "FAIL -- divergence found. STOP; one implementation is wrong."))
    return ok


def kuhn_coverage():
    """EXHAUSTIVE proof that Kuhn has no out_of_model behaviour.

    Enumerate every COMPLETE history in the Kuhn tree (all subject and observer
    action combinations) paired with every subject card, and check that some
    intent generates the subject's decisions in it with that card. Showdown is
    the hardest case: there the observer filters its set down to the true card,
    so emptiness there is exactly "no intent explains this card's play". A
    no-showdown history keeps other cards alive too, so it is strictly easier.

    If every pair is covered, the hypothesis set can never empty in Kuhn, so
    `out_of_model` is unreachable there and coincides with `in_model`. That is
    a statement about the GAME -- stronger than "0 of 72,000 sampled hands".
    """
    print("\n### Exhaustive: can any Kuhn behaviour leave the intent model?\n")

    def histories(hist):
        if K.to_act(hist) == "terminal":
            return [hist]
        out = []
        for a in K.legal(hist):
            out += histories(hist + (a,))
        return out

    total = uncovered = 0
    for card in K.CARDS:
        for hist in histories(()):
            # Intents that generate every subject decision in this history.
            ok = []
            for i in K.INTENTS:
                prefix, good = (), True
                for t in hist:
                    if K.to_act(prefix) == "subject" and K.subject_policy(i, card, prefix) != t:
                        good = False
                        break
                    prefix += (t,)
                if good:
                    ok.append(i)
            total += 1
            if not ok:
                uncovered += 1
                print(f"  UNCOVERED: card {card}, history {' '.join(hist)}")
    print(f"  (card, complete history) pairs checked: {total}")
    print(f"  explained by at least one intent: {total - uncovered}")
    print(f"  explained by NO intent:           {uncovered}")
    print("  => out_of_model is UNREACHABLE in Kuhn: the two misfit modes coincide there,\n"
          "     which is why --misfit-mode exists only in Leduc."
          if not uncovered else
          "  => out_of_model IS reachable in Kuhn; the modes differ. Investigate.")
    return uncovered == 0


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def harden():
    print("\n## Hardening: misattribution under out_of_model misfit (leduc, adaptive, mu=lam=0)\n")
    pooled = []
    for s in SEEDS:
        rows = run("leduc", "adaptive", "misfit:out_of_model", s)
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
    print(f"  observer chips/hand in those {len(early)} hands, plain observer "
          f"(plays passive after the model is refuted): {chips_after:+.3f}")
    print(f"  -> task 4c M1 acts on exactly these hands; see `python3 sweep.py deception`")

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
#       Falsifier: chips/hand on exactly the hands that refute the model before an
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
    orng = random.Random(seed + 1_000_000)
    mod = GAMES[game]
    state = {"refutations": 0}
    return [mod.play_hand(condition, rng, hand_no=k + 1, subject=subject, mu=mu, lam=lam,
                          deception_aware=deception_aware, state=state, orng=orng)
            for k in range(HANDS)]


def _report_stats(rows):
    n = len(rows)
    return {
        "exact": sum(r["report_exact"] for r in rows) / n,
        "misID": sum(r.get("report_misID", r["misattributed"]) for r in rows) / n,
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
    for label, subj, lam in [("out_of_model lam=0", "misfit:out_of_model", 0.0),
                             ("out_of_model lam=2", "misfit:out_of_model", 2.0),
                             ("in_model lam=2", "misfit:in_model", 2.0)]:
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
                             ("out_of_model, lam=0", "misfit:out_of_model", 0.0),
                             ("chip maximiser, lam=2", "misfit:out_of_model", 2.0),
                             ("in_model, lam=2", "misfit:in_model", 2.0)]:
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
                             ("out_of_model lam=0", "misfit:out_of_model", 0.0),
                             ("chip maximiser lam=2", "misfit:out_of_model", 2.0),
                             ("in_model lam=2", "misfit:in_model", 2.0)]:
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


# --------------------------------------- the consistent misattributor
#
# Does M2's guarantee have a hole? M2 abstains on CONTRADICTION, so it defends
# only against behaviour that LEAVES the intent model. This asks whether a
# subject can stay strictly inside the model and still drive the observer to a
# confident WRONG single intent.
#
# Verifications, not assumptions:
#   - contradiction rate must be EXACTLY 0 (that is what "stays in-model" means;
#     if it is not 0 the construction is broken, not merely weak)
#   - exact ID must be EXACTLY 0 (structural: the played intent always survives
#     and is not the declared one, so the set can never be exactly {declared})
#   - the played intent must be in the final set in EVERY hand
#   - M2 on and M2 off must produce identical numbers (no contradiction ever
#     fires, so the flag can never be raised). If they differ, either the
#     construction leaks or M2 is reading something other than contradictions.

def _mis_stats(rows):
    n = len(rows)
    return {
        "exact": sum(r.get("report_exact", r["exact"]) for r in rows) / n,
        "H": sum(r["final_size"] for r in rows) / n,
        "sound": sum(r["sound"] for r in rows) / n,
        "misID": sum(r.get("report_misID", r["misattributed"]) for r in rows) / n,
        "contra": sum(r.get("contradiction", False) for r in rows) / n,
        "chips": sum(r["observer_chips"] for r in rows) / n,
    }


def misattribute():
    print("\n## The consistent misattributor: in-model, and still confidently wrong\n")
    print("Declares honestly, then plays ANOTHER intent's policy faithfully -- the most")
    print("pinnable one available for its card. Never contradicts, by construction.")
    print(f"Cells: mean over seeds {tuple(SEEDS)} [min, max], n = {HANDS * len(SEEDS)} per row.\n")

    KEYS = ["exact", "H", "sound", "misID", "contra", "chips"]
    hdr = (f"{'condition':<14} {'k(M2)':>6} {'exact ID':>20} {'|H| final':>20} {'sound':>20} "
           f"{'misID':>20} {'contra':>20} {'chips/hand':>22}")
    print(hdr)
    print("-" * len(hdr))

    baselines = {}
    for cond in ["passive", "random", "adaptive"]:
        for k in [0, 1]:
            per = [_mis_stats(_run_state("leduc", cond, "misfit:misattribute", s,
                                         deception_aware=k)) for s in SEEDS]
            m = _mean_range(per, KEYS)
            baselines[(cond, k)] = m
            print(f"{cond:<14} {k:>6} " +
                  f"{fmt(m['exact']):>20} {fmt(m['H'], pct=False):>20} {fmt(m['sound']):>20} "
                  f"{fmt(m['misID']):>20} {fmt(m['contra']):>20} "
                  f"{fmt(m['chips'], pct=False, signed=True):>22}")

    print("\nKuhn -- where EVERY behaviour is in-model, so the constraint is free:")
    for cond in ["passive", "random", "adaptive"]:
        per = [_mis_stats(run("kuhn", cond, "misfit:misattribute", s)) for s in SEEDS]
        m = _mean_range(per, KEYS)
        print(f"{cond:<14} {'-':>6} " +
              f"{fmt(m['exact']):>20} {fmt(m['H'], pct=False):>20} {fmt(m['sound']):>20} "
              f"{fmt(m['misID']):>20} {fmt(m['contra']):>20} "
              f"{fmt(m['chips'], pct=False, signed=True):>22}")

    print("\nFor reference, the same observer against other subjects (adaptive, k=0):")
    for subj, label in [("faithful", "faithful"),
                        ("misfit:in_model", "misfit:in_model"),
                        ("misfit:out_of_model", "misfit:out_of_model")]:
        per = [_mis_stats(_run_state("leduc", "adaptive", subj, s)) for s in SEEDS]
        m = _mean_range(per, KEYS)
        print(f"{label:<14} {0:>6} " +
              f"{fmt(m['exact']):>20} {fmt(m['H'], pct=False):>20} {fmt(m['sound']):>20} "
              f"{fmt(m['misID']):>20} {fmt(m['contra']):>20} "
              f"{fmt(m['chips'], pct=False, signed=True):>22}")

    # ------------------------------------------------------------ verifications
    print("\n### Verifications (these would show the construction is broken)\n")
    ok = True
    for cond in ["passive", "random", "adaptive"]:
        rows = [r for s in SEEDS
                for r in _run_state("leduc", cond, "misfit:misattribute", s, deception_aware=0)]
        contra = sum(r["contradiction"] for r in rows)
        exact = sum(r["report_exact"] for r in rows)
        played_gone = sum(1 for r in rows if r["played_as"] not in r["final_set"])
        declared_played = sum(1 for r in rows if r["played_as"] == r["declared"])
        print(f"  {cond:<9} n={len(rows)}  contradictions={contra}  exact ID={exact}  "
              f"played intent eliminated={played_gone}  played==declared={declared_played}")
        ok &= (contra == 0 and exact == 0 and played_gone == 0 and declared_played == 0)

    print("\n  M2 on vs off (must be identical -- no contradiction ever fires):")
    for cond in ["passive", "random", "adaptive"]:
        a, b = baselines[(cond, 0)], baselines[(cond, 1)]
        same = all(abs(a[k][0] - b[k][0]) < 1e-12 for k in KEYS)
        print(f"  {cond:<9} identical on every metric: {same}")
        ok &= same
    print("\n  RESULT: " + ("construction verified" if ok else
                             "VERIFICATION FAILED -- do not report these numbers"))

    # ---------------------------------------------------- what it does at the table
    print("\n### What the attack actually does\n")
    print("Pin probability -- P(observer's final set is exactly {i}) if the subject plays i:\n")
    print(f"  {'card':<6}" + "".join(f"{i[:11]:>13}" for i in L.INTENTS))
    for c in L.DECK:
        print(f"  {c:<6}" + "".join(f"{L.pin_probability(i, c, 'adaptive', 0.0):>13.3f}"
                                    for i in L.INTENTS))
    print("\n  Every card has at least one intent the observer pins with probability 1.000")
    print("  (J: bluff or probe; Q: probe; K: value_bet). So whatever the subject declares,")
    print("  an alternative that gets pinned with CERTAINTY is almost always available.")
    choices = {}
    for c in L.DECK:
        for d in L.INTENTS:
            choices[(c, d)] = L.misattributing_intent(c, d, "adaptive", 0.0)
    from collections import Counter
    print("\n  intent actually played, over all (card, declared) pairs: " +
          ", ".join(f"{i} x{n}" for i, n in Counter(choices.values()).most_common()))


# ------------------------------------------- gridworld: the second environment

def grid_selfcheck():
    """The environment's own falsifiers. If any of these fail, no gridworld
    number below means anything."""
    print("\n### Gridworld environment checks\n")
    ok = True

    bad = tot = 0
    for closed in range(1 << len(GW.GATES)):
        if not GW.legal_close(closed):
            continue
        for d in GW.DESTS.values():
            f = GW.dist_field(d, closed)
            for s in GW.STARTS:
                tot += 1
                bad += s not in f
    print(f"  reachability   every destination from every start, every legal layout: "
          f"{tot - bad}/{tot} reachable")
    ok &= bad == 0

    worst = stuck = 0
    for closed in range(1 << len(GW.GATES)):
        if not GW.legal_close(closed):
            continue
        for s in GW.STARTS:
            for i in GW.INTENTS:
                pos, t = s, 0
                while t < GW.STEP_CAP:
                    m = GW.policy(i, pos, closed, 1)
                    if m == "stop":
                        break
                    pos = GW.step(pos, m)
                    t += 1
                worst = max(worst, t)
                stuck += pos != GW.DESTS[i[0]]
    print(f"  termination    longest episode {worst} steps (cap {GW.STEP_CAP}); "
          f"failed to arrive: {stuck}")
    ok &= stuck == 0 and worst < GW.STEP_CAP

    sig = {}
    for i in GW.INTENTS:
        tr = []
        for closed in range(1 << len(GW.GATES)):
            if not GW.legal_close(closed):
                continue
            for v in range(len(GW.VANTAGES)):
                pos, t = s, 0
                for s2 in GW.STARTS:
                    pos, t = s2, 0
                    while t < GW.STEP_CAP:
                        m = GW.policy(i, pos, closed, v)
                        tr.append(m)
                        if m == "stop":
                            break
                        pos = GW.step(pos, m)
                        t += 1
        sig.setdefault(tuple(tr), []).append(GW.name(i))
    dupes = [v for v in sig.values() if len(v) > 1]
    print(f"  distinctness   {len(sig)}/{len(GW.INTENTS)} intents behave distinctly"
          + ("" if not dupes else f"  DUPLICATES: {dupes}"))
    ok &= not dupes

    # --- the separation bound, machine-checked and pinned ---------------
    bs, bp, pl, n, nc = GW.static_bound()
    online = sum(1 for r in GW.all_episodes("adaptive", "online", random.Random(1))
                 if r["exact"])
    print(f"  static bound   {nc} fixed configurations x {n} episodes, brute-forced:")
    print(f"                   best single config   {bs}/{n} = {bs / n:.1%}")
    print(f"                   best per-start oracle {bp}/{n} = {bp / n:.1%}")
    print(f"                   pooled-trace bound    {pl}/{n} = {pl / n:.1%}")
    print(f"                   online adaptive       {online}/{n} = {online / n:.1%}")
    # Pinned so the separation cannot silently drift if the map is edited.
    EXPECT = (42, 46, 48, 48)      # new map, 2026-09-14; see RESULTS.md
    got = (bs, bp, pl, online)
    if got != EXPECT:
        print(f"  BOUND DRIFT: expected {EXPECT}, got {got} -- the separation claim "
              f"in RESULTS.md no longer matches the code")
    ok &= got == EXPECT
    ok &= bs <= bp <= pl          # the three bounds must stay nested
    # NOTE: no strict inequality is asserted any more. On the rebuilt map
    # online == pooled bound == 48/48, i.e. there is NO formal separation.
    # Asserting one would pin a claim the data does not support.
    ok &= online <= pl            # online can never EXCEED the pooled bound

    grd = GW.grd_legal_designs()
    print(f"  GRD constraint cost-preserving designs: {len(grd)} of "
          f"{1 << len(GW.GATES)} -> {grd}")
    wcds = sorted({GW.wcd(c) for c in grd})
    print(f"  wcd            over every legal design: {wcds}  "
          f"(gate choice does not move GRD's own metric on this map)")
    ok &= len(grd) > 1            # a real GRD instance has legal moves

    print("\n  " + ("PASS" if ok else "FAIL -- gridworld results are not trustworthy"))
    return ok


# ------------------------------------------------- guard the document itself
#
# The gate checks numbers, not prose -- which is how a splice edit silently
# deleted 347 lines of RESULTS.md (the whole gridworld and soft-elimination
# sections) without anything failing. This check is cheap and would have caught
# it immediately.

REQUIRED_SECTIONS = [
    "## Goal recognition and intent recognition are different problems",
    "## Confident misattribution",
    "## Kuhn (", "## Leduc (", "## Task 2: cost-aware misfit generator",
    "## Task 3: cost-aware observer", "## Hardening the misattribution result",
    "## Task 4c: a deception-aware observer", "## The consistent misattributor",
    "## Gridworld", "## Soft elimination", "## LLM as proposer",
    "## Vocabulary proposal", "## Gridworld human pilot",
    "## The specificity hypothesis", "## Descriptions vs behaviour", "## Commands",
]
RESULTS_LINE_FLOOR = 1100


def document_check(path="RESULTS.md"):
    print("\n### Document structure\n")
    try:
        text = open(path).read()
    except OSError as e:
        print(f"  FAIL cannot read {path}: {e}")
        return False
    lines = text.count("\n") + 1
    missing = [h for h in REQUIRED_SECTIONS if h not in text]
    print(f"  {path}: {lines} lines, {len(REQUIRED_SECTIONS) - len(missing)}"
          f"/{len(REQUIRED_SECTIONS)} required sections present")
    if missing:
        import difflib
        present = [ln for ln in text.splitlines() if ln.startswith("## ")]
        print("  MISSING SECTIONS:")
        for h in missing:
            near = difflib.get_close_matches(h, present, n=1, cutoff=0.6)
            hint = f"   closest actual heading: {near[0]!r}" if near else \
                   "   no similar heading found -- the section really is gone"
            print(f"    wanted {h!r}")
            print(f"   {hint}")
    if lines < RESULTS_LINE_FLOOR:
        print(f"  LINE-COUNT FLOOR BREACHED: {lines} < {RESULTS_LINE_FLOOR}. If this "
              f"shrinkage is intended, lower RESULTS_LINE_FLOOR in sweep.py in the "
              f"same commit that removes the content.")
    ok = not missing and lines >= RESULTS_LINE_FLOOR
    print("  " + ("PASS" if ok else "FAIL"))
    return ok


def grid():
    uniq, tot = GW.static_ceiling()
    print("\n## Gridworld: online probing vs. environment design\n")
    print("Episodes are ENUMERATED, not sampled: every start x every intent = "
          f"{len(GW.STARTS) * len(GW.INTENTS)} per cell.")
    print("Only the `random` observer varies with the seed; passive and adaptive are")
    print(f"deterministic, so their [min, max] is degenerate by construction.\n")
    print(f"Upper bound on exact ID for ANY FIXED layout (pooling every legal layout")
    print(f"and vantage): {uniq}/{tot} = {uniq / tot:.1%}. Environment design cannot beat this.")
    print("Online probing is not bound by it -- it can reconfigure mid-episode.\n")

    for subject in ["faithful", "misattribute"]:
        for mode in ["online", "design"]:
            results = {}
            for cond in ["passive", "random", "adaptive"]:
                per = []
                seeds = SEEDS if cond == "random" else SEEDS[:1]
                fn = (None if subject == "faithful" else
                      (lambda st, d, c=cond, m=mode: GW.misattributing_intent(st, d, c, m)))
                for s in seeds:
                    rows = GW.all_episodes(cond, mode, random.Random(s), 0.0, fn)
                    per.append(core.summarise(rows, GW.KEYS))
                results[cond] = core.aggregate(per)
            label = ("online probing" if mode == "online" else "environment design (GRD)")
            print(f"\n### {label} / {subject} subject")
            core.print_table(results, GW.KEYS, SEEDS, tot, "observer",
                             total_n=tot, notes=[])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("verify", "all"):
        if not verify() and what == "all":
            sys.exit("gate failed; not running the sweeps")
    if what in ("misfit", "adversary", "all"):    # "adversary": deprecated alias
        sweep_misfit()
    if what in ("observer", "all"):
        sweep_observer()
    if what in ("harden", "all"):
        harden()
    if what in ("deception", "all"):
        deception()
    if what in ("misattribute", "all"):
        misattribute()
    if what in ("grid", "all"):
        grid()
