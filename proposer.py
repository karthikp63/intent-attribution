#!/usr/bin/env python3
"""
LLM as PROPOSER -- first pass.

The architecture commitment: the LLM PROPOSES, never DECIDES, and is never in
the trust path. It does not eliminate hypotheses, rank them, or select probes.
Those stay symbolic. Every proposal is either validated by the symbolic layer
or rejected, and THE RESULT IS THE REJECTION RATE.

Pipeline:
    1. serialise    observed history -> plain language
    2. propose      LLM emits candidate intents
    3. compile      proposal -> the internal policy representation (a total
                    mapping over every decision cell)
    4. validate     total? legal actions? consistent with what was observed?
    5. classify     ill_formed | inconsistent | duplicate | novel_valid

Isolated from kuhn_intent / leduc_intent's experiment path on purpose: those
run with zero dependencies and `sweep.py verify` never imports this file. The
API backend uses urllib from the stdlib, so even this file has no third-party
imports.

    python3 proposer.py --backend fixture                 # replay recorded proposals
    python3 proposer.py --backend api --model claude-opus-5   # needs ANTHROPIC_API_KEY
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter

import leduc_intent as L

# --------------------------------------------------------------- the schema
#
# A policy is a TOTAL mapping over every decision cell the subject can face.
# Round 1 has no board; round 2 has one, and whether it pairs the hand is
# derived from the two ranks rather than stated separately.

SITUATIONS = ["open", "facing_bet", "facing_raise"]
LEGAL = {"open": ["check", "bet"],
         "facing_bet": ["fold", "call", "raise"],
         "facing_raise": ["fold", "call"]}
RANKS = ["J", "Q", "K"]


def cells():
    """Every (round, situation, card rank, board rank) the subject can face.
    9 round-1 cells + 27 round-2 cells = 36."""
    out = []
    for sit in SITUATIONS:
        for r in RANKS:
            out.append((1, sit, r, None))
    for sit in SITUATIONS:
        for r in RANKS:
            for b in RANKS:
                out.append((2, sit, r, b))
    return out


CELLS = cells()


def _hist_for(rnd, sit, board_rank):
    """A concrete history that puts the subject at this decision cell."""
    h = ()
    if rnd == 2:
        h = ("check", "check", "board:" + board_rank + "1")
    if sit == "facing_bet":
        h = h + ("bet",)
    elif sit == "facing_raise":
        h = h + ("bet", "raise")
    return h


def table_of_intent(intent):
    """Compile one of the built-in intents into the same table representation,
    so proposals and incumbents are compared as like for like."""
    t = {}
    for (rnd, sit, r, b) in CELLS:
        t[(rnd, sit, r, b)] = L.policy(intent, r + "0", _hist_for(rnd, sit, b))
    return t


BUILT_IN = {i: table_of_intent(i) for i in L.INTENTS}


# ------------------------------------------------------------- 4. validation

def check_total(table):
    """Every cell present, every action legal there. Returns an error or None."""
    missing = [c for c in CELLS if c not in table]
    if missing:
        return f"not total: {len(missing)}/{len(CELLS)} cells undetermined " \
               f"(e.g. {missing[0]})"
    for (rnd, sit, r, b), a in table.items():
        if a not in LEGAL[sit]:
            return f"illegal action {a!r} in situation {sit!r} (legal: {LEGAL[sit]})"
    return None


def consistent_with(table, observations):
    """Could this policy have produced what was observed?

    An observation is (history_prefix, action_taken, possible_cards). The policy
    is consistent if SOME card the observer has not ruled out generates every
    observed action. Same test the symbolic layer already applies to the
    built-in intents -- the proposer gets no special treatment.
    """
    for cards in [observations["cards"]]:
        for c in cards:
            if all(act_of(table, c, hist) == a for hist, a in observations["moves"]):
                return True
    return False


def act_of(table, card, hist):
    board = L.board_of(hist)
    rnd = 1 if board is None else 2
    return table.get((rnd, L.situation(hist), L.rank(card),
                      L.rank(board) if board else None))


def semantic_duplicate(table):
    """Identical decisions in all 36 cells as an existing intent."""
    for name, t in BUILT_IN.items():
        if all(table[c] == t[c] for c in CELLS):
            return name
    return None


def classify(table, err, observations):
    if err:
        return "ill_formed", err
    err = check_total(table)
    if err:
        return "ill_formed", err
    dup = semantic_duplicate(table)
    if dup:
        return "duplicate", f"identical to built-in intent {dup!r}"
    if observations and not consistent_with(table, observations):
        return "inconsistent", "well-formed, but no card makes it generate the observed actions"
    return "novel_valid", ""


# ------------------------------------------------- 3a. constrained compiler
#
# The constrained strategy asks for the table directly, so "compiling" is
# reading it and checking the keys are the ones we asked for. Reliable, and
# close to a lookup -- the model is barely reasoning.

def compile_constrained(obj):
    if not isinstance(obj, dict):
        return {}, f"expected a JSON object, got {type(obj).__name__}"
    table, bad = {}, []
    for k, v in obj.items():
        m = re.fullmatch(r"\s*(1|2)\s*[|/,]\s*(\w+)\s*[|/,]\s*([JQK])\s*(?:[|/,]\s*([JQK-]))?\s*", k)
        if not m:
            bad.append(k)
            continue
        rnd, sit, r, b = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        b = None if (b in (None, "-")) else b
        if sit not in SITUATIONS or (rnd == 1) != (b is None):
            bad.append(k)
            continue
        table[(rnd, sit, r, b)] = str(v).strip().lower()
    if bad:
        return table, f"{len(bad)} unparseable key(s), e.g. {bad[0]!r}"
    return table, None


# -------------------------------------------------- 3b. free-form compiler
#
# The interesting one. The model describes an intent in prose; a DETERMINISTIC
# parser turns that prose into a table. The compiler is symbolic on purpose: a
# second LLM pass would put the model back in the trust path, which is exactly
# what the architecture forbids. So the compile-failure rate measures how well
# free-form description maps onto the formal representation -- which is the
# number we actually want.

_ACTIONS = ["check", "bet", "fold", "call", "raise"]
_RANKWORD = {"jack": "J", "queen": "Q", "king": "K", "j": "J", "q": "Q", "k": "K"}


def _parse_clause(clause):
    """One rule -> (predicate, action), or None if nothing usable is in it."""
    c = clause.lower().strip()
    act = next((a for a in _ACTIONS if re.search(r"\b" + a + r"(s|ing)?\b", c)), None)
    if not act:
        return None

    rnd = None
    if re.search(r"\b(round\s*1|first round|pre-?board|preflop)\b", c):
        rnd = 1
    elif re.search(r"\b(round\s*2|second round|post-?board|after the board|on the board)\b", c):
        rnd = 2

    if re.search(r"\b(facing|against|if|when)\b[^.]*\braise[sd]?\b", c) and act != "raise":
        sit = "facing_raise"
    elif re.search(r"\b(facing|against|if|when)\b[^.]*\bbet(s|ting)?\b", c) and act != "bet":
        sit = "facing_bet"
    elif re.search(r"\b(open|opening|first to act|checked to|lead)\b", c):
        sit = "open"
    else:
        sit = None

    ranks = set()
    for w, r in _RANKWORD.items():
        if re.search(r"\b" + w + r"\b", c):
            ranks.add(r)
    if re.search(r"\bpair(ed|s)?\b", c):
        ranks |= {"__pair__"}
    if re.search(r"\b(strong|premium|nuts|value)\b", c) and not ranks:
        ranks |= {"K", "__pair__"}
    if re.search(r"\b(weak|bad|air|nothing)\b", c) and not ranks:
        ranks |= {"J"}
    if re.search(r"\b(any|anything|always|regardless|all)\b", c):
        ranks |= set(RANKS) | {"__pair__"}

    return {"rnd": rnd, "sit": sit, "ranks": ranks, "act": act,
            "default": bool(re.search(r"\b(otherwise|else|any other|in all other)\b", c))}


def compile_freeform(text):
    """Prose -> table. Later clauses win; a clause flagged 'otherwise' fills only
    what is still undetermined."""
    clauses = [c for c in re.split(r"[.;\n]|,\s*(?=and\b|but\b|otherwise\b)", text) if c.strip()]
    rules = [r for r in (_parse_clause(c) for c in clauses) if r]
    if not rules:
        return {}, "no rule of the form <condition> -> <action> could be extracted"

    table, defaults = {}, []
    for r in rules:
        rnds = [r["rnd"]] if r["rnd"] else [1, 2]
        sits = [r["sit"]] if r["sit"] else SITUATIONS
        target = defaults if r["default"] else None
        for (rnd, sit, rank_, b) in CELLS:
            if rnd not in rnds or sit not in sits:
                continue
            if r["act"] not in LEGAL[sit]:
                continue
            hit = (rank_ in r["ranks"]) or ("__pair__" in r["ranks"] and b == rank_)
            if not r["ranks"] or hit:
                if target is None:
                    table[(rnd, sit, rank_, b)] = r["act"]
                else:
                    target.append(((rnd, sit, rank_, b), r["act"]))
    for cell, a in defaults:
        table.setdefault(cell, a)
    return table, None


# ------------------------------------------------------------- 1. serialise

def serialise(hist, revealed_card=None):
    """Observed history -> plain language, as the LLM sees it.

    Who acted matters: only the SUBJECT's moves constrain the intent, and
    labelling the observer's moves as the subject's would ask the model to
    explain actions the subject never took.
    """
    out, prefix, rnd = [], (), 1
    for t in hist:
        if t.startswith("board:"):
            rnd = 2
            out.append(f"The board card is revealed: {L.rank(t[6:])}.")
        else:
            who = "subject" if L.to_act(prefix) == "subject" else "opponent"
            out.append(f"Round {rnd}: the {who} {t}s.")
        prefix += (t,)
    if revealed_card:
        out.append(f"At showdown the subject's card is revealed: {L.rank(revealed_card)}.")
    return " ".join(out) if out else "No actions observed yet."


# ------------------------------------------------------------- 2. the prompts

VOCAB = "\n".join(f"  - {i}: {g}" for i, g in L.INTENT_GLOSS.items())

GAME = f"""\
Leduc hold'em. Deck: two each of J, Q, K. Each player is dealt one card, there
is a betting round, one community (board) card is dealt, then a second betting
round. Round-1 bets are 2 chips, round-2 bets are 4. A pair (your card matching
the board) beats any unpaired hand; otherwise higher rank wins.

An INTENT is a complete policy: it says what the subject would do in EVERY
situation, including ones that did not occur. The situations are:
  round 1 or 2; open / facing_bet / facing_raise; card J, Q or K;
  and in round 2 the board is J, Q or K as well.
Legal actions: open -> check or bet; facing_bet -> fold, call or raise;
facing_raise -> fold or call.

Intents already in the vocabulary:
{VOCAB}
"""

CONSTRAINED_PROMPT = GAME + """
Propose ONE new intent that is not already in the vocabulary above.

Return ONLY a JSON object. Every key is "round|situation|card" for round 1 and
"round|situation|card|board" for round 2. Every one of the 36 cells must be
present. Values are the action. Example entries:
  "1|open|K": "bet",  "2|facing_bet|Q|J": "fold"
"""

FREEFORM_PROMPT = GAME + """
Propose ONE new intent that is not already in the vocabulary above.

Describe it in plain English: a short name, then the rules you would follow.
Do not use JSON or tables. Write it the way you would explain a plan to another
player.
"""

REPAIR_PROMPT = GAME + """
The subject's observed play is explained by NONE of the intents in the
vocabulary. Here is what was observed:

{observed}

The vocabulary is therefore incomplete. Propose ONE additional intent that
WOULD explain this behaviour, described in plain English as a complete policy.
"""


# ---------------------------------------------------------------- backends

def api_call(prompt, model, max_tokens=1500):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY is not set; use --backend fixture")
    body = json.dumps({"model": model, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["content"][0]["text"]


def load_fixture(path):
    with open(path) as f:
        return json.load(f)
