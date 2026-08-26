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
    python kuhn_intent.py --human --hands 10 --subject adversarial
"""

import argparse
import itertools
import json
import random
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


def choose_observer_action(H, legal, condition, rng):
    if condition == "passive":
        # A fixed, non-probing strategy: never raise, always see it through.
        return "check" if "check" in legal else "call"
    if condition == "random":
        return rng.choice(legal)
    if condition == "adaptive":
        scored = [(expected_posterior_size(H, a), a) for a in legal]
        best = min(s for s, _ in scored)
        # Deterministic tie-break: prefer the cheaper action.
        cost = {"check": 0, "fold": 0, "call": 1, "bet": 1}
        return min((a for s, a in scored if s == best), key=lambda a: cost[a])
    raise ValueError(condition)



# ------------------------------------------------------ adversarial subject
#
# The subject still DECLARES honestly (ground truth is unchanged) but then
# plays to defeat the observer: at each decision it picks the action that
# maximises the observer's expected final intent-set size. Worst case, as in
# the GRD literature: the subject knows the observer's selection rule and
# condition, but not the observer's card (uniform over the other two).
# Ties break toward the declared policy (minimal deviation).

def concealment(H):
    """Adversary's objective. An empty set means the observer's model was
    refuted outright, so it learned nothing: score it as the full space."""
    return len(intent_set(H)) if H else len(INTENTS)


def observer_action_dist(H, legal, condition):
    """The observer's action distribution as the subject can compute it."""
    if condition == "random":
        return {a: 1.0 / len(legal) for a in legal}
    return {choose_observer_action(H, legal, condition, None): 1.0}


def _adv_response_scores(subject_card, condition):
    """Score each response (fold/call) after check -> observer bets.
    Weighted by P(observer card) * P(observer bets | that card)."""
    scores = {"fold": 0.0, "call": 0.0}
    for o in CARDS:
        if o == subject_card:
            continue
        H = filter_opening(initial_hypotheses(o), "check")
        p_bet = 0.5 * observer_action_dist(H, ["check", "bet"], condition).get("bet", 0.0)
        if p_bet == 0.0:
            continue
        scores["fold"] += p_bet * concealment(filter_response(H, "fold"))
        scores["call"] += p_bet * concealment(filter_card(filter_response(H, "call"), subject_card))
    return scores


def _adv_opening_value(action, subject_card, condition):
    v = 0.0
    for o in CARDS:
        if o == subject_card:
            continue
        H = filter_opening(initial_hypotheses(o), action)
        if action == "bet":
            for a, p in observer_action_dist(H, ["fold", "call"], condition).items():
                v += 0.5 * p * concealment(H if a == "fold" else filter_card(H, subject_card))
        else:
            p_check = observer_action_dist(H, ["check", "bet"], condition).get("check", 0.0)
            v += 0.5 * p_check * concealment(filter_card(H, subject_card))
    if action == "check":
        v += max(_adv_response_scores(subject_card, condition).values())
    return v


def _argmax_prefer(scores, preferred):
    best = max(scores.values())
    return preferred if scores[preferred] == best else \
        max(scores, key=lambda a: scores[a])


def adversarial_opening(subject_card, declared, condition):
    scores = {a: _adv_opening_value(a, subject_card, condition) for a in ["check", "bet"]}
    return _argmax_prefer(scores, opening_action(declared, subject_card))


def adversarial_response(subject_card, declared, condition):
    scores = _adv_response_scores(subject_card, condition)
    return _argmax_prefer(scores, response_action(declared, subject_card))

# --------------------------------------------------------------- one hand

def play_hand(condition, rng, human=False, hand_no=0, subject="faithful"):
    adversarial = subject == "adversarial"
    deck = CARDS[:]
    rng.shuffle(deck)
    subject_card, observer_card = deck[0], deck[1]

    # --- the declaration. Made BEFORE acting. Never shown to the observer.
    if human:
        declared = prompt_declaration(subject_card, observer_card, hand_no, adversarial)
    else:
        declared = rng.choice(INTENTS)

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
    elif adversarial:
        opening = adversarial_opening(subject_card, declared, condition)
    else:
        opening = faithful_open
    subject_actions += 1
    deviations += opening != faithful_open
    H = filter_opening(H, opening)
    trace.append({"who": "subject", "action": opening, "surviving": intent_set(H)})

    if opening == "bet":
        legal = ["fold", "call"]
        obs_act = choose_observer_action(H, legal, condition, rng)
        trace.append({"who": "observer", "action": obs_act, "surviving": intent_set(H)})
        if obs_act == "fold":
            observer_chips = -1
        else:
            card_revealed = True
            H = filter_card(H, subject_card)
            observer_chips = 2 if RANK[observer_card] > RANK[subject_card] else -2
    else:
        legal = ["check", "bet"]
        obs_act = choose_observer_action(H, legal, condition, rng)
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
            elif adversarial:
                resp = adversarial_response(subject_card, declared, condition)
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
        "subject": subject,                  # log tag: faithful | adversarial
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

ADVERSARIAL_BRIEF = """
ADVERSARIAL CONDITION. Declare your intent honestly -- that is the ground
truth and it is never shown to the observer. Then play to DEFEAT the
observer: make it as hard as possible for it to work out what you declared.
You may depart from the suggested action whenever you like.
"""


def prompt_declaration(card, observer_card, hand_no, adversarial=False):
    if adversarial and hand_no == 1:
        print(ADVERSARIAL_BRIEF)
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


def print_table(results):
    hdr = (f"{'condition':<22} {'exact ID':>9} {'|H| final':>10} {'sound':>7} {'misID':>6} "
           f"{'reveal':>7} {'deviate':>8} {'chips/hand':>11}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for cond, s in results.items():
        print(f"{cond:<22} {s['exact_id_rate']:>8.1%} {s['mean_final_set']:>10.2f} "
              f"{s['soundness']:>6.0%} {s['misattribution']:>6.1%} {s['reveal_rate']:>7.0%} "
              f"{s['deviation_rate']:>8.1%} {s['mean_chips']:>+11.3f}")
    print()
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
    ap.add_argument("--subject", default="faithful",
                    choices=["faithful", "adversarial", "both"],
                    help="faithful: plays the declared policy; adversarial: declares "
                         "honestly, then plays to defeat the observer")
    ap.add_argument("--human", action="store_true", help="play as the subject")
    ap.add_argument("--log", metavar="PATH", help="write per-hand JSON records")
    args = ap.parse_args()

    conditions = ["passive", "random", "adaptive"] if args.condition == "all" \
        else [args.condition]

    subjects = ["faithful", "adversarial"] if args.subject == "both" else [args.subject]

    all_rows, results = [], {}
    for subj in subjects:
        for cond in conditions:
            rng = random.Random(args.seed)      # same deals across conditions
            rows = [play_hand(cond, rng, human=args.human, hand_no=k + 1, subject=subj)
                    for k in range(args.hands)]
            if args.human:
                for r in rows:
                    print(f"\n  observer's surviving intents: {', '.join(r['final_set'])}")
                    print(f"  you declared: {r['declared']}   ->  "
                          f"{'PINNED' if r['exact'] else ('in set' if r['sound'] else 'ELIMINATED')}")
            all_rows += rows
            results[f"{cond}/{subj}"] = summarise(rows)
    print_table(results)

    if args.log:
        with open(args.log, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"wrote {len(all_rows)} records to {args.log}")


if __name__ == "__main__":
    main()
