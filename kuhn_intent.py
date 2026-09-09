#!/usr/bin/env python3
"""
Adaptive Counterfactual Inquiry for Intent Attribution -- MVP

Kuhn poker. One subject (Player 1) declares an intent BEFORE acting; the
declaration is hidden from the observer (Player 2). The observer maintains an
explicit hypothesis set over (intent, card) pairs, eliminates hypotheses that
are inconsistent with what it sees, and -- in the ADAPTIVE condition -- chooses
its own actions to minimise the expected size of the surviving intent set.

No LLM. No learned distance metric. Exact set elimination over a finite space.

Usage:
    python kuhn_intent.py --hands 200 --condition all
    python kuhn_intent.py --hands 200 --condition all --log runs.json
    python kuhn_intent.py --hands 2000 --condition all --subject both
    python kuhn_intent.py --human --hands 10 --subject misfit
"""

import argparse
import itertools
import json
import random
from functools import lru_cache
import sys
from collections import Counter

# ---------------------------------------------------------------- game basics

CARDS = ["J", "Q", "K"]
RANK = {"J": 1, "Q": 2, "K": 3}

# ------------------------------------------------------------- intent space H
#
# An intent is a POLICY: a total mapping from (decision node, card) -> action.
# This is what makes an intent a counterfactual object rather than a label --
# it says what the subject WOULD do in situations that did not occur.

INTENTS = ["value_bet", "bluff", "probe", "give_up", "trap"]

INTENT_GLOSS = {
    "value_bet": "bet a strong hand for value; call a raise with Q or better",
    "bluff":     "bet a weak hand to represent strength; fold if raised",
    "probe":     "bet regardless of holding, to observe the response",
    "give_up":   "check and fold; conceding the pot",
    "trap":      "check a strong hand to induce a bet, then call",
}


def opening_action(intent, card):
    """What P1 does at the opening node under this intent, holding this card."""
    if intent == "value_bet":
        return "bet" if card == "K" else "check"
    if intent == "bluff":
        return "bet" if card == "J" else "check"
    if intent == "probe":
        return "bet"
    if intent == "give_up":
        return "check"
    if intent == "trap":
        return "check"
    raise ValueError(intent)


def response_action(intent, card):
    """What P1 does facing a bet after having checked."""
    if intent == "value_bet":
        return "call" if RANK[card] >= 2 else "fold"
    if intent == "bluff":
        return "fold"
    if intent == "probe":
        return "call"          # unreachable: probe always bets
    if intent == "give_up":
        return "fold"
    if intent == "trap":
        return "call"
    raise ValueError(intent)


# ------------------------------------------------------- the hypothesis space
#
# The observer does NOT see the subject's card. So a hypothesis is a joint
# (intent, card) pair. Its own card removes one holding from consideration.

def initial_hypotheses(observer_card):
    return [
        (i, c)
        for i in INTENTS
        for c in CARDS
        if c != observer_card
    ]


def filter_opening(H, observed):
    return [(i, c) for (i, c) in H if opening_action(i, c) == observed]


def filter_response(H, observed):
    return [(i, c) for (i, c) in H if response_action(i, c) == observed]


def filter_card(H, revealed):
    return [(i, c) for (i, c) in H if c == revealed]


def intent_set(H):
    return sorted({i for (i, _) in H})


# ------------------------------------------------- observer action selection
#
# ADAPTIVE rule: for each legal action, partition the current hypothesis set by
# the outcome the action would induce, and score the action by the EXPECTED
# size of the resulting intent set (uniform prior over surviving hypotheses).
# Pick the minimiser. This is greedy expected-posterior-size reduction --
# the simplest defensible instance of the selection rule.

def _outcomes_if_observer_bets(H):
    """P1 has checked; we bet. P1 folds or calls; a call reaches showdown."""
    buckets = {}
    for (i, c) in H:
        r = response_action(i, c)
        # A call goes to showdown, which additionally reveals the card.
        key = (r, c) if r == "call" else (r, None)
        buckets.setdefault(key, []).append((i, c))
    return buckets


def _outcomes_if_observer_checks(H):
    """P1 has checked; we check. Showdown reveals the card."""
    buckets = {}
    for (i, c) in H:
        buckets.setdefault(("showdown", c), []).append((i, c))
    return buckets


def _outcomes_if_observer_calls(H):
    """P1 has bet; we call. Showdown reveals the card."""
    buckets = {}
    for (i, c) in H:
        buckets.setdefault(("showdown", c), []).append((i, c))
    return buckets


def _outcomes_if_observer_folds(H):
    """P1 has bet; we fold. Nothing further is revealed."""
    return {("folded", None): list(H)}


OUTCOME_FN = {
    "bet": _outcomes_if_observer_bets,
    "check": _outcomes_if_observer_checks,
    "call": _outcomes_if_observer_calls,
    "fold": _outcomes_if_observer_folds,
}


def expected_posterior_size(H, action):
    """Expected |intent set| after taking `action`, under a uniform prior."""
    if not H:
        return 0.0
    buckets = OUTCOME_FN[action](H)
    total = len(H)
    return sum(
        (len(bucket) / total) * len(intent_set(bucket))
        for bucket in buckets.values()
    )


def choose_observer_action(H, legal, condition, rng, hist=None, observer_card=None,
                           observer="greedy", mu=0.0):
    if condition == "passive":
        # A fixed, non-probing strategy: never raise, always see it through.
        return "check" if "check" in legal else "call"
    if condition == "random":
        return rng.choice(legal)
    if condition == "adaptive" and observer == "lookahead":
        return choose_lookahead_action(H, hist, observer_card, mu)
    if condition == "adaptive":
        assert mu == 0.0, "chip-aware observer requires --observer lookahead"
        scored = [(expected_posterior_size(H, a), a) for a in legal]
        best = min(s for s, _ in scored)
        # Deterministic tie-break: prefer the cheaper action.
        cost = {"check": 0, "fold": 0, "call": 1, "bet": 1}
        return min((a for s, a in scored if s == best), key=lambda a: cost[a])
    raise ValueError(condition)




# ------------------------------------------------ history-based tree (Kuhn)
#
# A second, independent formulation of the same game as explicit histories,
# used by the LOOKAHEAD observer and by the adversary. Kept separate from the
# greedy code above so the two can be checked against each other.

def to_act(hist):
    if hist == ():
        return "subject"
    if hist in (("check",), ("bet",)):
        return "observer"
    if hist == ("check", "bet"):
        return "subject"
    return "terminal"


def legal(hist):
    return ["check", "bet"] if hist in ((), ("check",)) else ["fold", "call"]


def subject_policy(intent, card, hist):
    return opening_action(intent, card) if hist == () else response_action(intent, card)


def is_showdown(hist):
    return hist[-1] != "fold"


def observer_payoff(hist, subject_card, observer_card):
    if hist == ("bet", "fold"):
        return -1
    if hist == ("check", "bet", "fold"):
        return 1
    pot = 2 if "call" in hist else 1
    return pot if RANK[observer_card] > RANK[subject_card] else -pot


def observer_H(observer_card, hist):
    H = initial_hypotheses(observer_card)
    prefix = ()
    for t in hist:
        if to_act(prefix) == "subject":
            H = [(i, c) for (i, c) in H if subject_policy(i, c, prefix) == t]
        prefix += (t,)
    return H


def final_H(observer_card, hist, subject_card):
    H = observer_H(observer_card, hist)
    return filter_card(H, subject_card) if is_showdown(hist) else H


ACTION_COST = {"check": 0, "fold": 0, "call": 1, "bet": 1}


def V(H, hist, observer_card, mu=0.0):
    """Exact lookahead: expected (final |intent set| - mu * observer chips)
    if the observer plays optimally from here and the subject is faithful
    to some hypothesis in H, uniform prior. mu = 0 is pure information."""
    who = to_act(hist)
    n = len(H)
    if who == "terminal":
        buckets = {}
        for (i, c) in H:
            buckets.setdefault(c if is_showdown(hist) else None, []).append((i, c))
        size = sum(len(b) / n * len(intent_set(b)) for b in buckets.values())
        chips = sum(observer_payoff(hist, c, observer_card) for (_, c) in H) / n
        return size - mu * chips
    if who == "subject":
        buckets = {}
        for (i, c) in H:
            buckets.setdefault(subject_policy(i, c, hist), []).append((i, c))
        return sum(len(b) / n * V(b, hist + (a,), observer_card, mu)
                   for a, b in buckets.items())
    return min(V(H, hist + (a,), observer_card, mu) for a in legal(hist))


def choose_lookahead_action(H, hist, observer_card, mu):
    acts = legal(hist)
    if not H:
        return "check" if "check" in acts else "call"
    scored = [(V(H, hist + (a,), observer_card, mu), a) for a in acts]
    best = min(s for s, _ in scored)
    return min((a for s, a in scored if s == best), key=lambda a: ACTION_COST[a])

# ---------------------------------------------------------- MODEL MISFIT
#
# AN INSTRUMENT, NOT A SUBJECT MODEL. It generates behaviour the intent model
# cannot explain, at controlled rates and with worst-case coverage. See the
# longer note in leduc_intent.py. "Adversarial" was the wrong name: it implied
# someone who declares an intent and then plays against it, which is incoherent
# as a model of a person. What this produces is model misfit -- an incomplete
# vocabulary, a nearest-option answer to a menu that did not fit, a change of
# mind, a misread label, bad play. They are indistinguishable to the observer.
#
# Mechanically: declares honestly, then picks each action to maximise the
# observer's expected final intent-set size, knowing the observer's rule and
# condition but not its card. Ties break toward the declared policy.

# Kuhn has no in_model / out_of_model split: every (card, action-path) pair is
# consistent with at least one intent, so the set can never empty and
# out_of_model is unreachable. `sweep.py verify` proves this by enumeration.
MISFIT_MODES = ["in_model", "misattribute"]
_SUBJECT_ALIASES = {"adversarial": "misfit"}
_MODE_ALIASES = {"impersonate": "in_model", "refute": "out_of_model"}


def normalise_subject(subject):
    """'adversarial' -> 'misfit'. Idempotent."""
    head, _, mode = subject.partition(":")
    head = _SUBJECT_ALIASES.get(head, head)
    mode = _MODE_ALIASES.get(mode, mode)
    return head + ":" + mode if mode else head


def concealment(H):
    """Misfit objective. Kuhn never reaches an empty set, so the
    in_model / out_of_model distinction of Leduc does not arise here."""
    return len(intent_set(H)) if H else len(INTENTS)


def observer_action_dist(H, hist, observer_card, condition, observer, mu):
    """The observer's action distribution as the subject can compute it."""
    acts = legal(hist)
    if condition == "random":
        return {a: 1.0 / len(acts) for a in acts}
    return {choose_observer_action(H, acts, condition, None, hist, observer_card,
                                   observer, mu): 1.0}


# Cost: a WEIGHTED objective, concealment - lam * E[chips lost] (see the
# matching comment in leduc_intent.py for why not a hard budget).
def A(hist, belief, subject_card, condition, observer, mu, lam):
    """Adversary's expected (concealment - lam * chips lost) from `hist` on.
    `belief` = unnormalised weights over the observer's card, updated by the
    observer's own actions. Every max compares branches under one scaling."""
    who = to_act(hist)
    if who == "terminal":
        return sum(w * (concealment(final_H(o, hist, subject_card))
                        - lam * observer_payoff(hist, subject_card, o))
                   for o, w in belief)
    if who == "subject":
        return max(A(hist + (a,), belief, subject_card, condition, observer, mu, lam)
                   for a in legal(hist))
    branches = {}
    for o, w in belief:
        dist = observer_action_dist(observer_H(o, hist), hist, o, condition, observer, mu)
        for a, p in dist.items():
            if p > 0:
                branches.setdefault(a, []).append((o, w * p))
    return sum(A(hist + (a,), bl, subject_card, condition, observer, mu, lam)
               for a, bl in branches.items())


def misfit_belief(hist, subject_card, condition, observer, mu):
    belief = [(o, 1.0) for o in CARDS if o != subject_card]
    prefix = ()
    for t in hist:
        if to_act(prefix) == "observer":
            new = []
            for o, w in belief:
                p = observer_action_dist(observer_H(o, prefix), prefix, o,
                                         condition, observer, mu).get(t, 0.0)
                if w * p > 0:
                    new.append((o, w * p))
            belief = new
        prefix += (t,)
    return belief


def misfit_action(hist, subject_card, declared, condition, observer, mu, lam):
    belief = misfit_belief(hist, subject_card, condition, observer, mu)
    scores = {a: A(hist + (a,), belief, subject_card, condition, observer, mu, lam)
              for a in legal(hist)}
    preferred = subject_policy(declared, subject_card, hist)
    best = max(scores.values())
    return preferred if scores[preferred] == best else max(scores, key=scores.get)


# ------------------------------------------- THE CONSISTENT MISATTRIBUTOR
#
# Same construction as Leduc's (see the long note there). Kuhn is the case where
# EVERY behaviour is in-model -- `sweep.py verify` proves all 15 (card, history)
# pairs are explained by some intent -- so "stay inside the model" costs the
# subject nothing at all here. Whether that translates into misattribution is a
# separate question about how pinnable Kuhn's intents are, and is measured, not
# assumed.

@lru_cache(maxsize=None)
def pin_probability(i_star, subject_card, condition, observer="lookahead", mu=0.0):
    """P(the observer's final intent set is exactly {i_star}) when the subject
    plays i_star's policy with subject_card. Exact enumeration over the
    observer's card and, for the random condition, its own randomisation."""
    def rec(hist, observer_card):
        who = to_act(hist)
        if who == "terminal":
            return 1.0 if intent_set(final_H(observer_card, hist, subject_card)) == [i_star] else 0.0
        if who == "subject":
            return rec(hist + (subject_policy(i_star, subject_card, hist),), observer_card)
        dist = observer_action_dist(observer_H(observer_card, hist), hist,
                                    observer_card, condition, observer, mu)
        return sum(p * rec(hist + (a,), observer_card) for a, p in dist.items() if p)

    obs = [o for o in CARDS if o != subject_card]
    return sum(rec((), o) for o in obs) / len(obs)


@lru_cache(maxsize=None)
def misattributing_intent(subject_card, declared, condition, observer="lookahead", mu=0.0):
    cands = [i for i in INTENTS if i != declared]
    return max(cands, key=lambda i: (pin_probability(i, subject_card, condition, observer, mu),
                                     -INTENTS.index(i)))


# --------------------------------------------------------------- one hand

def play_hand(condition, rng, human=False, hand_no=0, subject="faithful",
              observer="greedy", mu=0.0, lam=0.0, orng=None):
    subject = normalise_subject(subject)
    misfit = subject.startswith("misfit")
    objective = subject.split(":")[1] if ":" in subject else None
    # `orng` is the observer's OWN random stream, separate from `rng`, which
    # deals the cards and picks the declaration. Without the split, the random
    # condition drew its own actions from the deal stream and thereby dealt
    # ITSELF different cards than passive/adaptive saw -- so the conditions
    # were not paired. Defaults to rng only for direct callers that predate it.
    if orng is None:
        orng = rng
    deck = CARDS[:]
    rng.shuffle(deck)
    subject_card, observer_card = deck[0], deck[1]

    # --- the declaration. Made BEFORE acting. Never shown to the observer.
    if human:
        declared = prompt_declaration(subject_card, observer_card, hand_no, misfit)
    else:
        declared = rng.choice(INTENTS)

    play_as = (misattributing_intent(subject_card, declared, condition, observer, mu)
               if objective == "misattribute" else None)

    H = initial_hypotheses(observer_card)
    trace = []
    observer_chips = 0
    subject_actions = 0
    deviations = 0
    card_revealed = False

    # --- opening node -------------------------------------------------------
    faithful_open = opening_action(declared, subject_card)
    if human:
        opening = prompt_action("Your move", ["check", "bet"], suggested=faithful_open)
    elif objective == "misattribute":
        opening = subject_policy(play_as, subject_card, ())
    elif misfit:
        opening = misfit_action((), subject_card, declared, condition, observer, mu, lam)
    else:
        opening = faithful_open
    subject_actions += 1
    deviations += opening != faithful_open
    H = filter_opening(H, opening)
    trace.append({"who": "subject", "action": opening, "surviving": intent_set(H)})

    if opening == "bet":
        obs_act = choose_observer_action(H, ["fold", "call"], condition, orng,
                                         ("bet",), observer_card, observer, mu)
        trace.append({"who": "observer", "action": obs_act, "surviving": intent_set(H)})
        if obs_act == "fold":
            observer_chips = -1
        else:
            card_revealed = True
            H = filter_card(H, subject_card)
            observer_chips = 2 if RANK[observer_card] > RANK[subject_card] else -2
    else:
        obs_act = choose_observer_action(H, ["check", "bet"], condition, orng,
                                         ("check",), observer_card, observer, mu)
        trace.append({"who": "observer", "action": obs_act, "surviving": intent_set(H)})
        if obs_act == "check":
            card_revealed = True
            H = filter_card(H, subject_card)
            observer_chips = 1 if RANK[observer_card] > RANK[subject_card] else -1
        else:
            faithful_resp = response_action(declared, subject_card)
            if human:
                resp = prompt_action("Observer bets. Your move", ["fold", "call"],
                                     suggested=faithful_resp)
            elif objective == "misattribute":
                resp = subject_policy(play_as, subject_card, ("check", "bet"))
            elif misfit:
                resp = misfit_action(("check", "bet"), subject_card, declared,
                                     condition, observer, mu, lam)
            else:
                resp = faithful_resp
            subject_actions += 1
            deviations += resp != faithful_resp
            H = filter_response(H, resp)
            trace.append({"who": "subject", "action": resp, "surviving": intent_set(H)})
            if resp == "fold":
                observer_chips = 1
            else:
                card_revealed = True
                H = filter_card(H, subject_card)
                observer_chips = 2 if RANK[observer_card] > RANK[subject_card] else -2

    final = intent_set(H)
    return {
        "hand": hand_no,
        "condition": condition,
        "subject": subject,                  # log tag: faithful | misfit
        "played_as": play_as,                # misattribute mode: intent actually played
        "observer": observer, "mu": mu, "lam": lam,
        "subject_card": subject_card,
        "observer_card": observer_card,
        "declared": declared,
        "final_set": final,
        "final_size": len(final),
        "sound": declared in final,          # was the truth ever eliminated?
        "exact": final == [declared],        # pinned down to one intent
        "subject_actions": subject_actions,
        "deviations": deviations,            # subject moves off the declared policy
        "misattributed": len(final) == 1 and final != [declared],
        "card_revealed": card_revealed,
        "observer_chips": observer_chips,
        "trace": trace,
    }


# ------------------------------------------------------------- human prompts

MISFIT_BRIEF = """
MISFIT CONDITION. Declare your intent honestly -- that is the ground truth and
it is never shown to the observer. Then play however you like, including in
ways the declared intent does not describe. The point is to produce behaviour
the intent vocabulary cannot account for.
"""


def prompt_declaration(card, observer_card, hand_no, misfit=False):
    if misfit and hand_no == 1:
        print(MISFIT_BRIEF)
    print("\n" + "=" * 62)
    print(f"HAND {hand_no}   your card: {card}   (observer holds {observer_card}"
          f" -- shown to you only for calibration)")
    print("=" * 62)
    print("Declare your intent BEFORE you act. The observer never sees this.\n")
    for n, i in enumerate(INTENTS, 1):
        print(f"  {n}. {i:<10} -- {INTENT_GLOSS[i]}")
    while True:
        try:
            raw = input("\nIntent [1-5]: ").strip()
        except EOFError:
            sys.exit("\ninput ended.")
        if raw.isdigit() and 1 <= int(raw) <= len(INTENTS):
            return INTENTS[int(raw) - 1]
        print("  enter a number 1-5")


def prompt_action(label, options, suggested=None):
    hint = f"  (playing your declared intent faithfully = {suggested})" if suggested else ""
    print(f"\n{label}:{hint}")
    for n, o in enumerate(options, 1):
        print(f"  {n}. {o}")
    while True:
        try:
            raw = input(f"Action [1-{len(options)}]: ").strip()
        except EOFError:
            sys.exit("\ninput ended.")
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  enter a number 1-{len(options)}")


# ----------------------------------------------------------------- reporting

def summarise(rows):
    n = len(rows)
    return {
        "hands": n,
        "exact_id_rate": sum(r["exact"] for r in rows) / n,
        "soundness": sum(r["sound"] for r in rows) / n,
        "mean_final_set": sum(r["final_size"] for r in rows) / n,
        "mean_chips": sum(r["observer_chips"] for r in rows) / n,
        "reveal_rate": sum(r["card_revealed"] for r in rows) / n,
        "misattribution": sum(r["misattributed"] for r in rows) / n,
        "deviation_rate": sum(r["deviations"] for r in rows)
                          / max(1, sum(r["subject_actions"] for r in rows)),
    }


def aggregate(per_seed):
    """mean, min, max of each metric across seeds."""
    return {k: (sum(p[k] for p in per_seed) / len(per_seed),
                min(p[k] for p in per_seed), max(p[k] for p in per_seed))
            for k in per_seed[0]}


def _cell(v, kind, multi):
    m, lo, hi = v
    f = (lambda x: f"{x:.1%}") if kind == "pct" else \
        (lambda x: f"{x:.2f}") if kind == "num" else (lambda x: f"{x:+.3f}")
    return f"{f(m)} [{f(lo)},{f(hi)}]" if multi else f(m)


def print_table(results, seeds, hands):
    multi = len(seeds) > 1
    w = 24 if multi else 11
    hdr = (f"{'condition':<22} {'n':>7} {'exact ID':>{w}} {'|H| final':>{w}} {'sound':>{w}} "
           f"{'misID':>{w}} {'reveal':>{w}} {'deviate':>{w}} {'chips/hand':>{w}}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for cond, s in results.items():
        print(f"{cond:<22} {hands * len(seeds):>7} "
              f"{_cell(s['exact_id_rate'], 'pct', multi):>{w}} "
              f"{_cell(s['mean_final_set'], 'num', multi):>{w}} "
              f"{_cell(s['soundness'], 'pct', multi):>{w}} "
              f"{_cell(s['misattribution'], 'pct', multi):>{w}} "
              f"{_cell(s['reveal_rate'], 'pct', multi):>{w}} "
              f"{_cell(s['deviation_rate'], 'pct', multi):>{w}} "
              f"{_cell(s['mean_chips'], 'chips', multi):>{w}}")
    print()
    print(f"n          = hands per cell ({hands} x {len(seeds)} seed(s): {', '.join(map(str, seeds))})"
          + ("; cells show mean [min, max] across seeds" if multi else ""))
    print("Conditions are compared on IDENTICAL deals: every condition replays the same")
    print("shuffles and declarations from a re-seeded stream, and the observer draws its")
    print("own randomness from a separate one. Comparisons are PAIRED, not independent.")
    print("exact ID   = hypothesis set collapsed to exactly the declared intent")
    print("|H| final  = mean number of intents still standing at hand's end (lower is better)")
    print("sound      = declared intent never wrongly eliminated (100% by construction for a faithful subject)")
    print("misID      = set collapsed to exactly one intent, and it is the wrong one")
    print("deviate    = fraction of subject decisions that departed from the declared policy")
    print("reveal     = fraction of hands reaching showdown (card revealed)")
    print("chips/hand = observer's mean profit; the cost side of the information trade")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", type=int, default=200)
    ap.add_argument("--condition", default="all",
                    choices=["adaptive", "passive", "random", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", metavar="S",
                    help="run several seeds and report mean [min, max] per metric "
                         "(overrides --seed)")
    ap.add_argument("--subject", default="faithful",
                    choices=["faithful", "misfit", "misfit:misattribute", "both", "adversarial"],
                    help="faithful: plays the declared policy; misfit: declares honestly, "
                         "then produces behaviour the intent model cannot explain "
                         "('adversarial' is a deprecated alias for misfit)")
    ap.add_argument("--observer", default="greedy", choices=["greedy", "lookahead"],
                    help="adaptive rule: one-step greedy (original) or exact lookahead "
                         "(same code shape as Leduc); must agree in Kuhn")
    ap.add_argument("--mu", type=float, default=0.0,
                    help="observer chip weight: minimise E[|H|] - mu * E[chips]")
    ap.add_argument("--lam", type=float, default=0.0,
                    help="adversary chip weight: maximise E[|H|] - lam * E[chips lost]")
    ap.add_argument("--human", action="store_true", help="play as the subject")
    ap.add_argument("--log", metavar="PATH", help="write per-hand JSON records")
    args = ap.parse_args()

    conditions = ["passive", "random", "adaptive"] if args.condition == "all" \
        else [args.condition]

    want = _SUBJECT_ALIASES.get(args.subject, args.subject)
    subjects = ["faithful", "misfit"] if want == "both" else [want]

    seeds = args.seeds if args.seeds else [args.seed]
    if args.human and len(seeds) > 1:
        sys.exit("--seeds is for batch runs; use --seed with --human")

    all_rows, results = [], {}
    for subj in subjects:
        for cond in conditions:
            per_seed = []
            for s in seeds:
                # Deals and declarations come from `rng`; the observer's own
                # randomness from `orng`. Re-seeded per condition, so every
                # condition sees exactly the same deals -- paired comparison.
                rng = random.Random(s)
                orng = random.Random(s + 1_000_000)
                rows = [play_hand(cond, rng, human=args.human, hand_no=k + 1, subject=subj,
                                  observer=args.observer, mu=args.mu, lam=args.lam, orng=orng)
                        for k in range(args.hands)]
                if args.human:
                    for r in rows:
                        print(f"\n  observer's surviving intents: {', '.join(r['final_set'])}")
                        print(f"  you declared: {r['declared']}   ->  "
                              f"{'PINNED' if r['exact'] else ('in set' if r['sound'] else 'ELIMINATED')}")
                all_rows += rows
                per_seed.append(summarise(rows))
            results[f"{cond}/{subj}"] = aggregate(per_seed)
    print_table(results, seeds, args.hands)

    if args.log:
        with open(args.log, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"wrote {len(all_rows)} records to {args.log}")


if __name__ == "__main__":
    main()
