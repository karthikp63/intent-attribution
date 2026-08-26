#!/usr/bin/env python3
"""
Adaptive Counterfactual Inquiry for Intent Attribution -- Leduc hold'em

Same architecture as kuhn_intent.py, on a bigger tree:

  * 6-card deck (J, Q, K, two of each), one private card each, one public
    card after the first betting round, two betting rounds (bet 2 then 4),
    at most one bet and one raise per round. Pair beats high card.
  * An intent is a POLICY: a total mapping from (round, node, card, board)
    to an action. Hypotheses are (intent, card instance) pairs.
  * The observer eliminates hypotheses inconsistent with what it sees and,
    in the ADAPTIVE condition, chooses its action by EXACT LOOKAHEAD over the
    rest of the hand: minimise the expected size of the final intent set,
    uniform prior over surviving hypotheses. In Kuhn this reduces to the
    one-step greedy rule; here one step would stop at the round boundary.
  * The ADVERSARIAL subject declares honestly, then plays to maximise the
    observer's expected final intent-set size (it knows the observer's rule
    and condition, not its card). Ties break toward the declared policy.
    Two variants: "impersonate" must stay consistent with some intent;
    "refute" may play in a way that matches no intent at all.

Still no LLM, no learned metric: exact set elimination over 7 x 5 = 35
hypotheses, exhaustive tree walk for both players' lookahead.

Usage:
    python leduc_intent.py --hands 2000 --condition all --subject both
    python leduc_intent.py --human --hands 10
"""

import argparse
import json
import random
import sys
from functools import lru_cache

# ---------------------------------------------------------------- game basics

RANKS = ["J", "Q", "K"]
RANK = {"J": 1, "Q": 2, "K": 3}
DECK = ["J0", "J1", "Q0", "Q1", "K0", "K1"]     # card instances
ANTE = 1
BET = {1: 2, 2: 4}


def rank(card):
    return card[0]


def board_of(hist):
    for t in hist:
        if t.startswith("board:"):
            return t[6:]
    return None


def round_seq(hist):
    """(round number, actions so far in the current round)."""
    seq, rnd = [], 1
    for t in hist:
        if t.startswith("board:"):
            seq, rnd = [], 2
        else:
            seq.append(t)
    return rnd, tuple(seq)


def _round_over(seq):
    return bool(seq) and (seq[-1] in ("fold", "call") or seq == ("check", "check"))


def to_act(hist):
    """'subject' | 'observer' | 'chance' | 'terminal'. Subject acts first."""
    rnd, seq = round_seq(hist)
    if seq and seq[-1] == "fold":
        return "terminal"
    if _round_over(seq):
        return "chance" if rnd == 1 else "terminal"
    return "subject" if len(seq) % 2 == 0 else "observer"


def legal(hist):
    _, seq = round_seq(hist)
    if not seq or seq[-1] == "check":
        return ["check", "bet"]
    if seq[-1] == "bet":
        return ["fold", "call", "raise"]
    return ["fold", "call"]                      # facing a raise


def situation(hist):
    _, seq = round_seq(hist)
    if not seq or seq[-1] == "check":
        return "open"
    return "facing_bet" if seq[-1] == "bet" else "facing_raise"


def is_showdown(hist):
    return to_act(hist) == "terminal" and hist[-1] != "fold"


def contributions(hist):
    """Chips put in by (subject, observer) along this history."""
    put = [ANTE, ANTE]
    rnd, seq, who = 1, 0, 0
    for t in hist:
        if t.startswith("board:"):
            rnd, seq = 2, 0
            continue
        who = seq % 2
        if t == "bet":
            put[who] += BET[rnd]
        elif t == "call":
            put[who] = put[1 - who]
        elif t == "raise":
            put[who] = put[1 - who] + BET[rnd]
        seq += 1
    return put


def strength(card, board):
    """Hand strength for showdown: pairs beat high cards."""
    return (10 if rank(card) == rank(board) else 0) + RANK[rank(card)]


def observer_payoff(hist, subject_card, observer_card):
    put = contributions(hist)
    if hist[-1] == "fold":
        _, seq = round_seq(hist)
        folder = (len(seq) - 1) % 2            # 0 = subject folded
        return put[0] if folder == 0 else -put[1]
    s, o = strength(subject_card, board_of(hist)), strength(observer_card, board_of(hist))
    return put[0] if o > s else (-put[1] if s > o else 0)


# ------------------------------------------------------------- intent space H
#
# Each intent is a total policy over (round, situation, card, board).
# Round-1 nodes have no board. "pair" means own card matches the board.

INTENTS = ["value_bet", "bluff", "probe", "give_up", "trap", "represent", "pot_control"]

INTENT_GLOSS = {
    "value_bet":   "bet/raise strong hands (K, pairs) for value; call with medium",
    "bluff":       "bet J to represent strength; play strong hands passively; fold if raised",
    "probe":       "bet the first round with anything to see the response, then slow down",
    "give_up":     "check and fold; conceding the pot",
    "trap":        "check strong hands to induce a bet, then raise",
    "represent":   "check round 1, then bet a Q/K board regardless of holding",
    "pot_control": "bet K early, then check round 2 and call one bet with K or a pair",
}


def policy(intent, card, hist):
    r = rank(card)
    board = board_of(hist)
    rnd = 1 if board is None else 2
    sit = situation(hist)
    pair = board is not None and rank(board) == r
    strong = pair or r == "K"                    # round-2 notion of strong

    if intent == "value_bet":
        if rnd == 1:
            if sit == "open":         return "bet" if r == "K" else "check"
            if sit == "facing_bet":   return {"K": "raise", "Q": "call", "J": "fold"}[r]
            return "call" if r == "K" else "fold"
        if sit == "open":             return "bet" if strong else "check"
        if sit == "facing_bet":       return "raise" if pair else ("call" if r == "K" else "fold")
        return "call" if strong else "fold"

    if intent == "bluff":
        if rnd == 1:
            if sit == "open":         return "bet" if r == "J" else "check"
            if sit == "facing_bet":   return "fold" if r == "J" else "call"
            return "call" if r == "K" else "fold"
        if sit == "open":             return "bet" if r == "J" else "check"
        if sit == "facing_bet":       return "call" if strong else "fold"
        return "call" if pair else "fold"

    if intent == "probe":
        if rnd == 1:
            if sit == "open":         return "bet"
            if sit == "facing_bet":   return "call"          # unreachable
            return "call" if r == "K" else "fold"
        if sit == "open":             return "check"
        if sit == "facing_bet":       return "call" if strong else "fold"
        return "call" if pair else "fold"

    if intent == "give_up":
        return "check" if sit == "open" else "fold"

    if intent == "trap":
        if rnd == 1:
            if sit == "open":         return "check"
            if sit == "facing_bet":   return {"K": "raise", "Q": "call", "J": "fold"}[r]
            return "call" if r == "K" else "fold"
        if sit == "open":             return "check"
        if sit == "facing_bet":       return "raise" if strong else ("call" if r == "Q" else "fold")
        return "call" if strong else "fold"

    if intent == "represent":
        if rnd == 1:
            if sit == "open":         return "check"
            if sit == "facing_bet":   return "fold" if r == "J" else "call"
            return "call" if r == "K" else "fold"
        if sit == "open":             return "bet" if RANK[rank(board)] >= 2 else "check"
        if sit == "facing_bet":       return "call" if strong else "fold"
        return "call" if pair else "fold"

    if intent == "pot_control":
        if rnd == 1:
            if sit == "open":         return "bet" if r == "K" else "check"
            if sit == "facing_bet":   return "fold" if r == "J" else "call"
            return "call" if r == "K" else "fold"
        if sit == "open":             return "check"
        if sit == "facing_bet":       return "call" if strong else "fold"
        return "call" if pair else "fold"

    raise ValueError(intent)


# ------------------------------------------------------- the hypothesis space

def initial_hypotheses(observer_card):
    return tuple((i, c) for i in INTENTS for c in DECK if c != observer_card)


def intent_set(H):
    return sorted({i for (i, _) in H})


def observer_H(observer_card, hist):
    """The observer's hypothesis set after seeing `hist` (no showdown filter)."""
    H = initial_hypotheses(observer_card)
    prefix = ()
    for t in hist:
        if t.startswith("board:"):
            b = t[6:]
            H = tuple((i, c) for (i, c) in H if c != b)
        elif to_act(prefix) == "subject":
            H = tuple((i, c) for (i, c) in H if policy(i, c, prefix) == t)
        prefix = prefix + (t,)
    return H


def final_H(observer_card, hist, subject_card):
    H = observer_H(observer_card, hist)
    if is_showdown(hist):
        H = tuple((i, c) for (i, c) in H if c == subject_card)
    return H


# ------------------------------------------------- observer action selection
#
# Exact lookahead: V(H, hist) = expected final |intent set| if the observer
# plays optimally (for information) from here and the subject is faithful
# to some hypothesis in H. Chance nodes weight each board by the number of
# hypotheses compatible with it. Same uniform-prior expectation as Kuhn.

ACTION_COST = {"check": 0, "fold": 0, "call": 1, "bet": 2, "raise": 2}


@lru_cache(maxsize=None)
def V(H, hist, observer_card):
    who = to_act(hist)
    n = len(H)
    if who == "terminal":
        if not is_showdown(hist):
            return float(len(intent_set(H)))
        buckets = {}
        for (i, c) in H:
            buckets.setdefault(c, []).append((i, c))
        return sum(len(b) / n * len(intent_set(b)) for b in buckets.values())
    if who == "chance":
        v = 0.0
        for b in DECK:
            if b == observer_card:
                continue
            Hb = tuple((i, c) for (i, c) in H if c != b)
            if Hb:
                v += len(Hb) / (4 * n) * V(Hb, hist + ("board:" + b,), observer_card)
        return v
    if who == "subject":
        buckets = {}
        for (i, c) in H:
            buckets.setdefault(policy(i, c, hist), []).append((i, c))
        return sum(len(b) / n * V(tuple(b), hist + (a,), observer_card)
                   for a, b in buckets.items())
    return min(V(H, hist + (a,), observer_card) for a in legal(hist))


def choose_observer_action(H, hist, observer_card, condition, rng):
    acts = legal(hist)
    if condition == "passive":
        return "check" if "check" in acts else "call"
    if condition == "random":
        return rng.choice(acts)
    if condition == "adaptive":
        if not H:                                # model refuted; nothing to learn
            return "check" if "check" in acts else "call"
        scored = [(V(H, hist + (a,), observer_card), a) for a in acts]
        best = min(s for s, _ in scored)
        return min((a for s, a in scored if s == best), key=lambda a: ACTION_COST[a])
    raise ValueError(condition)


def observer_action_dist(H, hist, observer_card, condition):
    if condition == "random":
        acts = legal(hist)
        return {a: 1.0 / len(acts) for a in acts}
    return {choose_observer_action(H, hist, observer_card, condition, None): 1.0}


# ------------------------------------------------------ adversarial subject
#
# Belief over the observer's card instance, updated by the observer's own
# actions (which are a deterministic function of its card under adaptive /
# passive). Unnormalised weights are fine: every max compares branches
# under the same scaling.

# Two adversaries, differing only in how they score an EMPTY final set:
#   impersonate: 0   -- must still look like *some* intent (deceiver hides
#                       among the others; never refutes the model)
#   refute:      |INTENTS| -- breaking the observer's model counts as full
#                       concealment (the observer learned nothing)
ADVERSARY_OBJECTIVES = ["impersonate", "refute"]


def concealment(H, objective):
    if H:
        return len(intent_set(H))
    return 0 if objective == "impersonate" else len(INTENTS)


@lru_cache(maxsize=None)
def A(hist, belief, subject_card, condition, objective):
    """Adversary's expected concealment from `hist` onward."""
    who = to_act(hist)
    if who == "terminal":
        return sum(w * concealment(final_H(o, hist, subject_card), objective)
                   for o, w in belief)
    if who == "subject":
        return max(A(hist + (a,), belief, subject_card, condition, objective)
                   for a in legal(hist))
    if who == "observer":
        branches = {}
        for o, w in belief:
            dist = observer_action_dist(observer_H(o, hist), hist, o, condition)
            for a, p in dist.items():
                if p > 0:
                    branches.setdefault(a, []).append((o, w * p))
        return sum(A(hist + (a,), tuple(bl), subject_card, condition, objective)
                   for a, bl in branches.items())
    branches = {}                                # chance: board
    for o, w in belief:
        for b in DECK:
            if b not in (o, subject_card):
                branches.setdefault(b, []).append((o, w / 4))
    return sum(A(hist + ("board:" + b,), tuple(bl), subject_card, condition, objective)
               for b, bl in branches.items())


def adversary_belief(hist, subject_card, condition):
    """Weights over observer cards consistent with the observer's play so far."""
    belief = tuple((o, 1.0) for o in DECK if o != subject_card)
    prefix = ()
    for t in hist:
        if to_act(prefix) == "observer":
            new = []
            for o, w in belief:
                p = observer_action_dist(observer_H(o, prefix), prefix, o, condition).get(t, 0.0)
                if w * p > 0:
                    new.append((o, w * p))
            belief = tuple(new)
        elif t.startswith("board:"):
            belief = tuple((o, w / 4) for o, w in belief if o != t[6:])
        prefix = prefix + (t,)
    return belief


def adversarial_action(hist, subject_card, declared, condition, objective):
    belief = adversary_belief(hist, subject_card, condition)
    scores = {a: A(hist + (a,), belief, subject_card, condition, objective)
              for a in legal(hist)}
    preferred = policy(declared, subject_card, hist)
    best = max(scores.values())
    return preferred if scores[preferred] == best else max(scores, key=scores.get)


# --------------------------------------------------------------- one hand

def play_hand(condition, rng, human=False, hand_no=0, subject="faithful"):
    """subject: faithful | adversarial:impersonate | adversarial:refute"""
    adversarial = subject.startswith("adversarial")
    objective = subject.split(":")[1] if adversarial else None
    deck = DECK[:]
    rng.shuffle(deck)
    subject_card, observer_card, board = deck[0], deck[1], deck[2]

    if human:
        declared = prompt_declaration(subject_card, hand_no, adversarial)
    else:
        declared = rng.choice(INTENTS)

    hist = ()
    H = initial_hypotheses(observer_card)
    trace = []
    subject_actions = deviations = 0

    while True:
        who = to_act(hist)
        if who == "terminal":
            break
        if who == "chance":
            hist = hist + ("board:" + board,)
            H = tuple((i, c) for (i, c) in H if c != board)
            trace.append({"who": "board", "action": board, "surviving": intent_set(H)})
            continue
        if who == "subject":
            faithful = policy(declared, subject_card, hist)
            if human:
                a = prompt_action(f"Round {round_seq(hist)[0]}, your move", legal(hist), faithful)
            elif adversarial:
                a = adversarial_action(hist, subject_card, declared, condition, objective)
            else:
                a = faithful
            subject_actions += 1
            deviations += a != faithful
            H = tuple((i, c) for (i, c) in H if policy(i, c, hist) == a)
        else:
            a = choose_observer_action(H, hist, observer_card, condition, rng)
            if human:
                print(f"  observer: {a}")
        hist = hist + (a,)
        trace.append({"who": who, "action": a, "surviving": intent_set(H)})

    if is_showdown(hist):
        H = tuple((i, c) for (i, c) in H if c == subject_card)
    final = intent_set(H)
    return {
        "hand": hand_no,
        "condition": condition,
        "subject": subject,
        "subject_card": subject_card,
        "observer_card": observer_card,
        "board": board_of(hist),
        "declared": declared,
        "history": list(hist),
        "final_set": final,
        "final_size": len(final),
        "sound": declared in final,
        "exact": final == [declared],
        "misattributed": len(final) == 1 and final != [declared],
        "contradiction": len(final) == 0,
        "subject_actions": subject_actions,
        "deviations": deviations,
        "card_revealed": is_showdown(hist),
        "observer_chips": observer_payoff(hist, subject_card, observer_card),
        "trace": trace,
    }


# ------------------------------------------------------------- human prompts

ADVERSARIAL_BRIEF = """
ADVERSARIAL CONDITION. Declare your intent honestly -- that is the ground
truth and it is never shown to the observer. Then play to DEFEAT the
observer: make it as hard as possible for it to work out what you declared.
You may depart from the suggested action whenever you like.
"""


def prompt_declaration(card, hand_no, adversarial=False):
    if adversarial and hand_no == 1:
        print(ADVERSARIAL_BRIEF)
    print("\n" + "=" * 62)
    print(f"HAND {hand_no}   your card: {rank(card)}")
    print("=" * 62)
    print("Declare your intent BEFORE you act. The observer never sees this.\n")
    for n, i in enumerate(INTENTS, 1):
        print(f"  {n}. {i:<12} -- {INTENT_GLOSS[i]}")
    while True:
        try:
            raw = input(f"\nIntent [1-{len(INTENTS)}]: ").strip()
        except EOFError:
            sys.exit("\ninput ended.")
        if raw.isdigit() and 1 <= int(raw) <= len(INTENTS):
            return INTENTS[int(raw) - 1]
        print(f"  enter a number 1-{len(INTENTS)}")


def prompt_action(label, options, suggested=None):
    hint = f"  (faithful to your declared intent = {suggested})" if suggested else ""
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
        "misattribution": sum(r["misattributed"] for r in rows) / n,
        "contradiction": sum(r["contradiction"] for r in rows) / n,
        "reveal_rate": sum(r["card_revealed"] for r in rows) / n,
        "deviation_rate": sum(r["deviations"] for r in rows)
                          / max(1, sum(r["subject_actions"] for r in rows)),
        "mean_chips": sum(r["observer_chips"] for r in rows) / n,
    }


def print_table(results):
    hdr = (f"{'condition':<30} {'exact ID':>9} {'|H| final':>10} {'sound':>7} {'misID':>6} "
           f"{'contra':>7} {'reveal':>7} {'deviate':>8} {'chips/hand':>11}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for cond, s in results.items():
        print(f"{cond:<30} {s['exact_id_rate']:>8.1%} {s['mean_final_set']:>10.2f} "
              f"{s['soundness']:>6.0%} {s['misattribution']:>6.1%} {s['contradiction']:>7.1%} "
              f"{s['reveal_rate']:>7.0%} {s['deviation_rate']:>8.1%} {s['mean_chips']:>+11.3f}")
    print()
    print(f"intents: {len(INTENTS)}; hypotheses: {len(INTENTS)} x 5 card instances = {5 * len(INTENTS)}")
    print("exact ID   = hypothesis set collapsed to exactly the declared intent")
    print("|H| final  = mean number of intents still standing at hand's end (lower is better)")
    print("sound      = declared intent never wrongly eliminated (100% by construction for a faithful subject)")
    print("misID      = set collapsed to exactly one intent, and it is the wrong one")
    print("contra     = every hypothesis eliminated: the subject's play matched no intent at all")
    print("reveal     = fraction of hands reaching showdown (card revealed)")
    print("deviate    = fraction of subject decisions that departed from the declared policy")
    print("chips/hand = observer's mean profit; the cost side of the information trade")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", type=int, default=200)
    ap.add_argument("--condition", default="all",
                    choices=["adaptive", "passive", "random", "all"])
    ap.add_argument("--subject", default="faithful",
                    choices=["faithful", "adversarial", "both"],
                    help="faithful: plays the declared policy; adversarial: declares "
                         "honestly, then plays to defeat the observer")
    ap.add_argument("--adversary", default="both",
                    choices=ADVERSARY_OBJECTIVES + ["both"],
                    help="impersonate: must still look like some intent; "
                         "refute: may break the observer's model (empty set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--human", action="store_true", help="play as the subject")
    ap.add_argument("--log", metavar="PATH", help="write per-hand JSON records")
    args = ap.parse_args()

    conditions = ["passive", "random", "adaptive"] if args.condition == "all" \
        else [args.condition]
    advs = ADVERSARY_OBJECTIVES if args.adversary == "both" else [args.adversary]
    subjects = []
    if args.subject in ("faithful", "both"):
        subjects.append("faithful")
    if args.subject in ("adversarial", "both"):
        subjects += ["adversarial:" + o for o in advs]

    all_rows, results = [], {}
    for subj in subjects:
        for cond in conditions:
            rng = random.Random(args.seed)
            rows = [play_hand(cond, rng, human=args.human, hand_no=k + 1, subject=subj)
                    for k in range(args.hands)]
            if args.human:
                for r in rows:
                    print(f"\n  observer's surviving intents: {', '.join(r['final_set']) or '(none)'}")
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
