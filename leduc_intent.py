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
    Two misfit modes: "in_model" stays consistent with some intent;
    "out_of_model" may play in a way that matches no intent at all.

Still no LLM, no learned metric: exact set elimination over 7 x 5 = 35
hypotheses, exhaustive tree walk for both players' lookahead.

Usage:
    python leduc_intent.py --hands 2000 --condition all --subject both
    python leduc_intent.py --human --hands 10
    python leduc_intent.py --hands 2000 --subject misfit --deception-aware 1
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
def V(H, hist, observer_card, mu=0.0):
    """Expected (final |intent set| - mu * observer chips) under optimal
    observer play from here, faithful subject, uniform prior over H.
    mu = 0 is the pure information rule; mu > 0 buys chips with ambiguity."""
    who = to_act(hist)
    n = len(H)
    if who == "terminal":
        buckets = {}
        for (i, c) in H:
            buckets.setdefault(c if is_showdown(hist) else None, []).append((i, c))
        size = sum(len(b) / n * len(intent_set(b)) for b in buckets.values())
        chips = sum(observer_payoff(hist, c, observer_card) for (_, c) in H) / n
        return size - mu * chips
    if who == "chance":
        v = 0.0
        for b in DECK:
            if b == observer_card:
                continue
            Hb = tuple((i, c) for (i, c) in H if c != b)
            if Hb:
                v += len(Hb) / (4 * n) * V(Hb, hist + ("board:" + b,), observer_card, mu)
        return v
    if who == "subject":
        buckets = {}
        for (i, c) in H:
            buckets.setdefault(policy(i, c, hist), []).append((i, c))
        return sum(len(b) / n * V(tuple(b), hist + (a,), observer_card, mu)
                   for a, b in buckets.items())
    return min(V(H, hist + (a,), observer_card, mu) for a in legal(hist))


def chip_value_no_model(hist, observer_card, cards):
    """Observer's expected chips from `hist` on once its intent model is
    REFUTED: uniform over the subject's remaining card instances, and a
    subject whose future actions are uniform over its legal moves -- because
    an empty hypothesis set means the observer has no model of this subject
    left to plan against. Exact enumeration, no sampling."""
    who = to_act(hist)
    n = len(cards)
    if who == "terminal":
        return sum(observer_payoff(hist, c, observer_card) for c in cards) / n
    if who == "chance":
        v = 0.0
        for b in DECK:
            if b == observer_card:
                continue
            rem = [c for c in cards if c != b]
            if rem:
                v += len(rem) / (4 * n) * chip_value_no_model(
                    hist + ("board:" + b,), observer_card, rem)
        return v
    if who == "subject":
        acts = legal(hist)
        return sum(chip_value_no_model(hist + (a,), observer_card, cards)
                   for a in acts) / len(acts)
    return max(chip_value_no_model(hist + (a,), observer_card, cards)
               for a in legal(hist))


def choose_observer_action(H, hist, observer_card, condition, rng, mu=0.0,
                           refuted_chip_play=False):
    acts = legal(hist)
    if condition == "passive":
        return "check" if "check" in acts else "call"
    if condition == "random":
        return rng.choice(acts)
    if condition == "adaptive":
        if not H:                                # model refuted; nothing to learn
            # Deception-aware (task 4c, mechanism 1). The first version of
            # this folded -- "nothing left to learn, so stop paying". That was
            # WRONG and the measurement caught it: refuting the INTENT model
            # says nothing about CARD equity, and the observer still holds the
            # stronger hand in 45% of these spots, so folding forfeited pots it
            # would have won (-0.61 -> -6.75 chips/hand). What it should do is
            # play for chips with no model at all.
            if refuted_chip_play:
                board = board_of(hist)
                cards = [c for c in DECK if c != observer_card and c != board]
                return max(legal(hist),
                           key=lambda a: chip_value_no_model(hist + (a,), observer_card, cards))
            return "check" if "check" in acts else "call"
        scored = [(V(H, hist + (a,), observer_card, mu), a) for a in acts]
        best = min(s for s, _ in scored)
        return min((a for s, a in scored if s == best), key=lambda a: ACTION_COST[a])
    raise ValueError(condition)


def observer_action_dist(H, hist, observer_card, condition, mu=0.0):
    if condition == "random":
        acts = legal(hist)
        return {a: 1.0 / len(acts) for a in acts}
    return {choose_observer_action(H, hist, observer_card, condition, None, mu): 1.0}


# ---------------------------------------------------------- MODEL MISFIT
#
# THIS IS AN INSTRUMENT, NOT A SUBJECT MODEL. It generates behaviour the intent
# model cannot explain, at controlled rates and with worst-case coverage. It is
# not a claim that anyone plays this way.
#
# "Adversarial" was the wrong name for it. It implied someone who declares an
# intent and then plays against it -- incoherent as a model of a person; nobody
# lies to a dropdown and then acts against their own answer. What the condition
# actually produces is MODEL MISFIT: behaviour the intent vocabulary does not
# cover. That happens constantly with real subjects for ordinary reasons -- the
# vocabulary is incomplete, they picked the nearest option from a menu that did
# not fit, they changed their mind mid-hand, they misread a label, they played
# badly. All of those look identical to the observer: same signal, same
# contradiction, same failure mode. The generator maximises misfit because that
# is how you get an upper bound on it, not because subjects optimise.
#
# Mechanically: DECLARES honestly (ground truth unchanged), then picks each
# action to maximise the observer's expected final intent-set size. Worst case,
# as in GRD: it knows the observer's selection rule and condition but not its
# card. Ties break toward the declared policy, so it departs only when that
# widens the gap.
#
# Belief over the observer's card instance, updated by the observer's own
# actions (which are a deterministic function of its card under adaptive /
# passive). Unnormalised weights are fine: every max compares branches
# under the same scaling.

# Two misfit modes, differing only in how they score an EMPTY final set:
#   in_model:     0  -- the behaviour still matches *some* intent, so the
#                       vocabulary covers it; misfit is only in WHICH intent.
#   out_of_model: |INTENTS| -- the behaviour matches no intent at all, so the
#                       vocabulary does not cover it. The observer's model is
#                       refuted rather than merely confused.
#
# A third mode, `misattribute`, is built differently -- see THE CONSISTENT
# MISATTRIBUTOR below. It does not score concealment at all; it maximises the
# chance the observer lands confidently on the WRONG single intent.
MISFIT_MODES = ["in_model", "out_of_model", "misattribute"]

# Back-compatible names. The condition was called "adversarial" and its modes
# "impersonate"/"refute" until the reframing; old commands and old logged
# records still work.
_SUBJECT_ALIASES = {"adversarial": "misfit"}
_MODE_ALIASES = {"impersonate": "in_model", "refute": "out_of_model"}


def normalise_subject(subject):
    """'adversarial:impersonate' -> 'misfit:in_model'. Idempotent."""
    head, _, mode = subject.partition(":")
    head = _SUBJECT_ALIASES.get(head, head)
    mode = _MODE_ALIASES.get(mode, mode)
    return head + ":" + mode if mode else head


ADVERSARY_OBJECTIVES = MISFIT_MODES          # deprecated alias, kept for callers


def concealment(H, objective):
    if H:
        return len(intent_set(H))
    return 0 if objective == "in_model" else len(INTENTS)


# Cost: a WEIGHTED objective, concealment - lam * E[chips lost], rather than
# a hard budget. The recursion is already an expectation-max over the tree, so
# a weight folds into the terminal value and every node stays a plain max; a
# hard constraint on expected loss would need a Lagrangian (i.e. this lam,
# found by search) or a constrained search over mixed strategies. lam = 0 is
# the pure concealer; lam -> inf is a pure chip maximiser.

@lru_cache(maxsize=None)
def A(hist, belief, subject_card, condition, objective, mu=0.0, lam=0.0):
    """Adversary's expected (concealment - lam * chips lost) from `hist` on."""
    who = to_act(hist)
    if who == "terminal":
        return sum(w * (concealment(final_H(o, hist, subject_card), objective)
                        - lam * observer_payoff(hist, subject_card, o))
                   for o, w in belief)
    if who == "subject":
        return max(A(hist + (a,), belief, subject_card, condition, objective, mu, lam)
                   for a in legal(hist))
    if who == "observer":
        branches = {}
        for o, w in belief:
            dist = observer_action_dist(observer_H(o, hist), hist, o, condition, mu)
            for a, p in dist.items():
                if p > 0:
                    branches.setdefault(a, []).append((o, w * p))
        return sum(A(hist + (a,), tuple(bl), subject_card, condition, objective, mu, lam)
                   for a, bl in branches.items())
    branches = {}                                # chance: board
    for o, w in belief:
        for b in DECK:
            if b not in (o, subject_card):
                branches.setdefault(b, []).append((o, w / 4))
    return sum(A(hist + ("board:" + b,), tuple(bl), subject_card, condition, objective, mu, lam)
               for b, bl in branches.items())


def misfit_belief(hist, subject_card, condition, mu=0.0):
    """Weights over observer cards consistent with the observer's play so far."""
    belief = tuple((o, 1.0) for o in DECK if o != subject_card)
    prefix = ()
    for t in hist:
        if to_act(prefix) == "observer":
            new = []
            for o, w in belief:
                p = observer_action_dist(observer_H(o, prefix), prefix, o,
                                         condition, mu).get(t, 0.0)
                if w * p > 0:
                    new.append((o, w * p))
            belief = tuple(new)
        elif t.startswith("board:"):
            belief = tuple((o, w / 4) for o, w in belief if o != t[6:])
        prefix = prefix + (t,)
    return belief


def misfit_action(hist, subject_card, declared, condition, objective, mu=0.0, lam=0.0):
    belief = misfit_belief(hist, subject_card, condition, mu)
    scores = {a: A(hist + (a,), belief, subject_card, condition, objective, mu, lam)
              for a in legal(hist)}
    preferred = policy(declared, subject_card, hist)
    best = max(scores.values())
    return preferred if scores[preferred] == best else max(scores, key=scores.get)


# ------------------------------------------- THE CONSISTENT MISATTRIBUTOR
#
# The open limitation of the deception-aware observer (task 4c, M2) was that it
# abstains on CONTRADICTION, so it only defends against behaviour that leaves
# the model. A subject that stays strictly inside the model while still steering
# the observer to a confident WRONG single intent would defeat it entirely.
#
# The hard constraint is: never empty the hypothesis set. That constraint has an
# exact characterisation, and it is much stronger than it looks:
#
#   At showdown the observer filters its set to the subject's revealed card, so
#   the set survives only if SOME intent generates the subject's entire action
#   path with that card. The set of never-contradicting behaviours is therefore
#   exactly { play intent i's policy : i in INTENTS }. There is nothing else in
#   it -- a subject cannot mix two intents across a hand unless a third intent
#   happens to generate the mixture.
#
# Two consequences fall straight out:
#
#   1. If the subject plays i*'s policy, i* can never be eliminated: it
#      generated every action and it matches the revealed card. So i* is in the
#      final set always, and any singleton final set is necessarily exactly
#      {i*}. (Enumerated: 840/840 combos, 0 eliminations, 0 singletons != i*.)
#   2. Misattribution therefore means the observer pins i* and i* is not what
#      the subject declared. The whole attack is: declare d, then play some
#      OTHER intent faithfully, chosen to be as pinnable as possible.
#
# That is not an exotic strategy. It is precisely the "picked the nearest option
# from a menu that did not fit" case from the misfit reframing: the person's
# label and the policy they actually execute come apart. The observer is not
# being deceived -- it is being told the truth by someone whose vocabulary does
# not match their behaviour, and it answers with confident, wrong certainty.


@lru_cache(maxsize=None)
def pin_probability(i_star, subject_card, condition, mu=0.0):
    """P(the observer's final intent set is exactly {i_star}) when the subject
    plays i_star's policy with subject_card.

    Exact enumeration over the observer's card (uniform over the 5 remaining
    instances), the board (uniform over the 4 then remaining) and -- for the
    random condition -- the observer's own randomisation. No sampling.
    """
    def rec(hist, observer_card, board):
        who = to_act(hist)
        if who == "terminal":
            return 1.0 if intent_set(final_H(observer_card, hist, subject_card)) == [i_star] else 0.0
        if who == "chance":
            return rec(hist + ("board:" + board,), observer_card, board)
        if who == "subject":
            return rec(hist + (policy(i_star, subject_card, hist),), observer_card, board)
        dist = observer_action_dist(observer_H(observer_card, hist), hist,
                                    observer_card, condition, mu)
        return sum(p * rec(hist + (a,), observer_card, board) for a, p in dist.items() if p)

    obs = [o for o in DECK if o != subject_card]
    total = 0.0
    for o in obs:
        boards = [b for b in DECK if b not in (subject_card, o)]
        for b in boards:
            total += rec((), o, b) / (len(obs) * len(boards))
    return total


@lru_cache(maxsize=None)
def misattributing_intent(subject_card, declared, condition, mu=0.0):
    """The intent to actually play: the most pinnable one that is NOT declared.
    Ties break by INTENTS order, purely for determinism."""
    cands = [i for i in INTENTS if i != declared]
    return max(cands, key=lambda i: (pin_probability(i, subject_card, condition, mu),
                                     -INTENTS.index(i)))


# --------------------------------------------------------------- one hand

def play_hand(condition, rng, human=False, hand_no=0, subject="faithful", mu=0.0, lam=0.0,
              deception_aware=0, state=None, orng=None):
    """subject: faithful | misfit:in_model | misfit:out_of_model
    (the old names adversarial / :impersonate / :refute are accepted too)

    deception_aware (task 4c): 0 = off, and every field below is exactly what
    the plain observer produces. k > 0 turns on the deception-aware observer:

      M1  when the hypothesis set empties mid-hand, play for chips with no
          model rather than drifting on passively.
      M2  once this subject has refuted the model in k PRIOR hands, stop
          issuing single-intent conclusions and report "contradicted" instead.

    M2 is a deductive trigger, not a statistical one. An empty set is proof
    that the subject is not faithful to any intent -- and "the subject is
    faithful to some intent" is the premise every elimination conclusion rests
    on. After a refutation, a confident claim rests on a disproved premise.
    The flag uses ONLY what the observer can see (its own set going empty);
    it never touches the declaration.

    `state` carries the refutation count across hands within one subject.
    """
    # `orng` is the observer's OWN random stream, separate from `rng`, which
    # deals the cards and picks the declaration. Without the split, the random
    # condition drew its own actions from the deal stream and thereby dealt
    # ITSELF different cards than passive/adaptive saw -- so the conditions
    # were not paired. Defaults to rng only for direct callers that predate it.
    if orng is None:
        orng = rng
    subject = normalise_subject(subject)
    misfit = subject.startswith("misfit")
    objective = subject.split(":")[1] if misfit else None
    deck = DECK[:]
    rng.shuffle(deck)
    subject_card, observer_card, board = deck[0], deck[1], deck[2]

    if human:
        declared = prompt_declaration(subject_card, hand_no, misfit)
    else:
        declared = rng.choice(INTENTS)

    # The consistent misattributor commits to one intent for the whole hand;
    # that is the entire space of never-contradicting behaviour (see above).
    play_as = (misattributing_intent(subject_card, declared, condition, mu)
               if objective == "misattribute" else None)

    hist = ()
    H = initial_hypotheses(observer_card)
    # Flag reflects PRIOR hands only; this hand's refutation is counted after.
    refuted_before = state.get("refutations", 0) if state is not None else 0
    flagged = deception_aware > 0 and refuted_before >= deception_aware
    trace = []
    subject_actions = deviations = 0
    contradiction_at = None          # index in history at which H became empty
    observer_moves_after = 0         # observer decisions taken after that point

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
            elif objective == "misattribute":
                a = policy(play_as, subject_card, hist)
            elif misfit:
                a = misfit_action(hist, subject_card, declared, condition, objective, mu, lam)
            else:
                a = faithful
            subject_actions += 1
            deviations += a != faithful
            H = tuple((i, c) for (i, c) in H if policy(i, c, hist) == a)
            if not H and contradiction_at is None:
                contradiction_at = len(hist)
        else:
            a = choose_observer_action(H, hist, observer_card, condition, orng, mu,
                                       refuted_chip_play=deception_aware > 0)
            observer_moves_after += contradiction_at is not None
            if human:
                print(f"  observer: {a}")
        hist = hist + (a,)
        trace.append({"who": who, "action": a, "surviving": intent_set(H)})

    if is_showdown(hist):
        H = tuple((i, c) for (i, c) in H if c == subject_card)
        if not H and contradiction_at is None:
            contradiction_at = "showdown"    # play fit some intent, the revealed card fit none
    final = intent_set(H)

    # What the observer actually REPORTS, as distinct from the surviving set.
    # With deception_aware = 0 these mirror the set-level fields exactly, so
    # every existing number is untouched.
    if len(final) == 0:
        reported = "contradicted"        # this hand refuted the model outright
    elif flagged and len(final) == 1:
        reported = "contradicted"        # premise already disproved: no confident claim
    elif len(final) == 1:
        reported = final[0]
    else:
        reported = None                  # ambiguous set; not a confident claim either way

    if state is not None and len(final) == 0:
        state["refutations"] = refuted_before + 1

    return {
        "hand": hand_no,
        "condition": condition,
        "subject": subject, "mu": mu, "lam": lam,
        "subject_card": subject_card,
        "observer_card": observer_card,
        "board": board_of(hist),
        "board_card": board,          # the card that WAS dealt, even if never revealed
        "declared": declared,
        "history": list(hist),
        "final_set": final,
        "final_size": len(final),
        "sound": declared in final,
        "exact": final == [declared],
        "misattributed": len(final) == 1 and final != [declared],
        "contradiction": len(final) == 0,
        "contradiction_at": contradiction_at,   # history index, "showdown", or None
        "observer_moves_after_contradiction": observer_moves_after,
        # --- task 4c: report level (== set level when deception_aware = 0)
        "played_as": play_as,                     # misattribute mode: intent actually played
        "deception_aware": deception_aware,
        "flagged": flagged,                       # subject had already refuted the model
        "reported": reported,                     # intent | "contradicted" | None
        "report_exact": reported == declared,
        "report_misID": reported is not None and reported != "contradicted"
                        and reported != declared,
        "abstained": flagged and len(final) == 1,  # confident claim withheld
        "subject_actions": subject_actions,
        "deviations": deviations,
        "card_revealed": is_showdown(hist),
        "observer_chips": observer_payoff(hist, subject_card, observer_card),
        "trace": trace,
    }


# ------------------------------------------------------------- human prompts

MISFIT_BRIEF = """
MISFIT CONDITION. Declare your intent honestly -- that is the ground truth and
it is never shown to the observer. Then play however you like, including in
ways the declared intent does not describe. The point is to produce behaviour
the intent vocabulary cannot account for.
"""


def prompt_declaration(card, hand_no, misfit=False):
    if misfit and hand_no == 1:
        print(MISFIT_BRIEF)
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
        "exact_id_rate": sum(r["report_exact"] for r in rows) / n,
        "soundness": sum(r["sound"] for r in rows) / n,
        "mean_final_set": sum(r["final_size"] for r in rows) / n,
        # Report level: what the observer actually CLAIMED. Identical to the
        # set-level fields whenever --deception-aware is 0.
        "misattribution": sum(r["report_misID"] for r in rows) / n,
        "abstain": sum(r["reported"] == "contradicted" for r in rows) / n,
        "contradiction": sum(r["contradiction"] for r in rows) / n,
        "reveal_rate": sum(r["card_revealed"] for r in rows) / n,
        "deviation_rate": sum(r["deviations"] for r in rows)
                          / max(1, sum(r["subject_actions"] for r in rows)),
        "mean_chips": sum(r["observer_chips"] for r in rows) / n,
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


COLS = [("exact ID", "exact_id_rate", "pct"), ("|H| final", "mean_final_set", "num"),
        ("sound", "soundness", "pct"), ("misID", "misattribution", "pct"),
        ("contra", "contradiction", "pct"), ("abstain", "abstain", "pct"),
        ("reveal", "reveal_rate", "pct"), ("deviate", "deviation_rate", "pct"),
        ("chips/hand", "mean_chips", "chips")]


def print_table(results, seeds, hands):
    multi = len(seeds) > 1
    w = 24 if multi else 11
    hdr = f"{'condition':<30} {'n':>7} " + " ".join(f"{t:>{w}}" for t, _, _ in COLS)
    print("\n" + hdr)
    print("-" * len(hdr))
    for cond, s in results.items():
        print(f"{cond:<30} {hands * len(seeds):>7} " +
              " ".join(f"{_cell(s[k], kind, multi):>{w}}" for _, k, kind in COLS))
    print()
    print(f"n          = hands per cell ({hands} x {len(seeds)} seed(s): {', '.join(map(str, seeds))})"
          + ("; cells show mean [min, max] across seeds" if multi else ""))
    print("Conditions are compared on IDENTICAL deals: every condition replays the same")
    print("shuffles and declarations from a re-seeded stream, and the observer draws its")
    print("own randomness from a separate one. Comparisons are PAIRED, not independent.")
    print(f"intents: {len(INTENTS)}; hypotheses: {len(INTENTS)} x 5 card instances = {5 * len(INTENTS)}")
    print("exact ID   = observer named exactly the declared intent")
    print("|H| final  = mean number of intents still standing at hand's end (lower is better)")
    print("sound      = declared intent never wrongly eliminated (100% by construction for a faithful subject)")
    print("misID      = observer named exactly one intent, and it is the wrong one")
    print("contra     = every hypothesis eliminated: the subject's play matched no intent at all")
    print("abstain    = observer reported 'contradicted' rather than naming a single intent")
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
                    choices=["faithful", "misfit", "both", "adversarial"],
                    help="faithful: plays the declared policy; misfit: declares honestly, "
                         "then produces behaviour the intent model cannot explain "
                         "('adversarial' is a deprecated alias for misfit)")
    ap.add_argument("--misfit-mode", "--adversary", default="both", dest="misfit_mode",
                    choices=MISFIT_MODES + ["both", "impersonate", "refute"],
                    help="in_model: behaviour still matches SOME intent, so the vocabulary "
                         "covers it; out_of_model: behaviour matches no intent at all "
                         "(impersonate/refute are deprecated aliases)")
    ap.add_argument("--mu", type=float, default=0.0,
                    help="observer chip weight: minimise E[|H|] - mu * E[chips]")
    ap.add_argument("--lam", type=float, default=0.0,
                    help="adversary chip weight: maximise E[|H|] - lam * E[chips lost]")
    ap.add_argument("--deception-aware", type=int, default=0, metavar="K",
                    dest="deception_aware",
                    help="deception-aware observer: after K prior hands in which the "
                         "subject refuted the model, report 'contradicted' rather than a "
                         "single intent; and play for chips once refuted mid-hand. "
                         "0 = off (default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", metavar="S",
                    help="run several seeds and report mean [min, max] per metric "
                         "(overrides --seed)")
    ap.add_argument("--human", action="store_true", help="play as the subject")
    ap.add_argument("--log", metavar="PATH", help="write per-hand JSON records")
    args = ap.parse_args()

    conditions = ["passive", "random", "adaptive"] if args.condition == "all" \
        else [args.condition]
    mode = _MODE_ALIASES.get(args.misfit_mode, args.misfit_mode)
    modes = MISFIT_MODES if mode == "both" else [mode]
    want = _SUBJECT_ALIASES.get(args.subject, args.subject)
    subjects = []
    if want in ("faithful", "both"):
        subjects.append("faithful")
    if want in ("misfit", "both"):
        subjects += ["misfit:" + m for m in modes]

    seeds = args.seeds if args.seeds else [args.seed]
    if args.human and len(seeds) > 1:
        sys.exit("--seeds is for batch runs; use --seed with --human")

    all_rows, results = [], {}
    for subj in subjects:
        for cond in conditions:
            per_seed = []
            for s in seeds:
                # Deals and declarations from `rng`; the observer's own
                # randomness from `orng`. Re-seeded per condition, so every
                # condition sees exactly the same deals -- paired comparison.
                rng = random.Random(s)
                orng = random.Random(s + 1_000_000)
                state = {"refutations": 0}      # carried across this subject's hands
                rows = [play_hand(cond, rng, human=args.human, hand_no=k + 1, subject=subj,
                                  mu=args.mu, lam=args.lam,
                                  deception_aware=args.deception_aware, state=state, orng=orng)
                        for k in range(args.hands)]
                if args.human:
                    for r in rows:
                        print(f"\n  observer's surviving intents: {', '.join(r['final_set']) or '(none)'}")
                        if r["reported"] == "contradicted":
                            print("  observer reports: CONTRADICTED -- no intent explains this"
                                  + ("  (subject already refuted the model)" if r["abstained"] else ""))
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
