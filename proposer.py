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
_NUM = {"one": 1, "two": 2, "1": 1, "2": 2, "first": 1, "second": 2}

# Condition phrases, longest/most specific first. Matched text is REMOVED from
# the clause before actions are scanned, so "facing a bet I raise" does not read
# the condition's "bet" as the action taken.
_SIT_PATTERNS = [
    ("facing_raise", r"\b(?:facing|against|if|when|to)\s+(?:a\s+|any\s+|they\s+|he\s+|she\s+|someone\s+)?"
                     r"(?:re-?)?raises?(?:\s+me(?:\s+back)?)?\b"),
    ("facing_raise", r"\bif\s+(?:they|he|she|someone)\s+(?:comes?\s+back|re-?raises?)\b"),
    ("facing_bet",   r"\b(?:facing|against|if|when|to)\s+(?:a\s+|any\s+|they\s+|he\s+|she\s+|someone\s+)?"
                     r"bets?(?:\s+(?:at|into)\s+me)?\b"),
    ("facing_bet",   r"\b(?:they|he|she|someone|the\s+opponent)\s+bets?\b"),
    ("facing_bet",   r"\bbet\s+into\s+me\b"),
    ("open",         r"\b(?:first\s+to\s+act|when\s+i\s+am\s+first|i\s+open|opening|"
                     r"checked\s+to(?:\s+me)?|lead(?:ing)?\s+out)\b"),
]

_DEFAULT_RE = r"\b(otherwise|else|any\s+other|in\s+all\s+other|the\s+rest|everything\s+else|anything\s+else)\b"


def _ranks_in(text):
    rs = set()
    for w, r in _RANKWORD.items():
        if re.search(r"\b" + w + r"\b", text):
            rs.add(r)
    if re.search(r"\bpair(ed|s)?\b", text):
        rs.add("__pair__")
    if re.search(r"\b(strong|premium|nuts|value)\b", text) and not rs:
        rs |= {"K", "__pair__"}
    if re.search(r"\b(weak|bad|air|nothing)\b", text) and not rs:
        rs |= {"J"}
    if re.search(r"\b(any|anything|always|regardless|all|every|whatever|no matter)\b", text):
        rs |= set(RANKS) | {"__pair__"}
    return rs


def _parse_clause(clause):
    """One clause -> a LIST of rules. A clause can carry several actions
    ('raise with a pair and call otherwise'), and each gets its own rule."""
    c = " " + clause.lower().strip() + " "

    rnd = None
    m = re.search(r"\bround\s+(one|two|1|2)\b", c) or re.search(r"\b(first|second)\s+round\b", c)
    if m:
        rnd = _NUM[m.group(1)]
    elif re.search(r"\b(pre-?board|preflop|street\s+one)\b", c):
        rnd = 1
    elif re.search(r"\b(post-?board|after\s+the\s+board|on\s+the\s+board|second\s+barrel|street\s+two)\b", c):
        rnd = 2

    sit = None
    for name, pat in _SIT_PATTERNS:
        m = re.search(pat, c)
        if m:
            sit = name
            c = c[:m.start()] + " , " + c[m.end():]      # remove the condition text
            break

    # Scan the remainder for action verbs; each takes the words after it, up to
    # the next action verb, as its rank restriction.
    hits = []
    for m in re.finditer(r"\b(" + "|".join(_ACTIONS) + r")(?:s|es|ing|ed)?\b", c):
        hits.append((m.start(), m.end(), m.group(1)))
    if not hits:
        return []
    rules = []
    for k, (st, en, act) in enumerate(hits):
        scope = c[en: hits[k + 1][0] if k + 1 < len(hits) else len(c)]
        rules.append({"rnd": rnd, "sit": sit, "ranks": _ranks_in(scope), "act": act,
                      "default": bool(re.search(_DEFAULT_RE, scope))})
    # A clause-wide "otherwise" with no ranks on the last action makes it the default.
    return rules


def compile_freeform(text):
    """Prose -> table. Rules apply in order, later ones overwriting earlier;
    a rule flagged 'otherwise' fills only cells still undetermined at the end."""
    # Split sentences, and also split a sentence that carries a SECOND condition
    # ("... facing a bet I call, and facing a raise I fold"): one clause can only
    # hold one condition, so a second one has to start a new clause.
    clauses = [c for c in re.split(
        r"[.;\n]"
        r"|,\s+(?=and\s+(?:facing|against|if|when|in\s+round)|otherwise\b|but\b)"
        r"|,\s+(?=(?:and\s+)?i\s+\w+)",           # comma-joined independent clauses
        text, flags=re.I) if c.strip()]
    rules = [r for c in clauses for r in _parse_clause(c)]
    if not rules:
        return {}, "no rule of the form <condition> -> <action> could be extracted"

    table, defaults = {}, []
    for r in rules:
        rnds = [r["rnd"]] if r["rnd"] else [1, 2]
        sits = [r["sit"]] if r["sit"] else SITUATIONS
        for (rnd, sit, rank_, b) in CELLS:
            if rnd not in rnds or sit not in sits:
                continue
            if r["act"] not in LEGAL[sit]:
                continue
            hit = (rank_ in r["ranks"]) or ("__pair__" in r["ranks"] and b == rank_)
            if r["default"]:
                defaults.append(((rnd, sit, rank_, b), r["act"]))
            elif not r["ranks"] or hit:
                table[(rnd, sit, rank_, b)] = r["act"]
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


# ------------------------------------------------------------------- runner

def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    import math
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


CATEGORIES = ["ill_formed", "inconsistent", "duplicate", "novel_valid"]
results_gap = {}          # proposal name -> cells left undetermined


def observations_for(hist, card):
    """The subject's own moves in this history, plus the cards still possible.
    The card is revealed at showdown, so the consistency test is against it."""
    moves, prefix = [], ()
    for t in hist:
        if not t.startswith("board:") and L.to_act(prefix) == "subject":
            moves.append((prefix, t))
        prefix += (t,)
    return {"moves": moves, "cards": [card]}


def _undetermined(table):
    return len([c for c in CELLS if c not in table])


def report(title, results, note=""):
    n = len(results)
    print(f"\n### {title}   (n = {n})")
    if note:
        print(note)
    counts = Counter(cat for cat, _, _ in results)
    rejected = n - counts["novel_valid"]
    print(f"\n{'category':<16}{'count':>7}{'rate':>9}   {'95% Wilson CI':>18}")
    print("-" * 54)
    for cat in CATEGORIES:
        k = counts[cat]
        p, lo, hi = wilson(k, n)
        print(f"{cat:<16}{k:>7}{p:>9.1%}   [{lo:>6.1%}, {hi:>6.1%}]")
    p, lo, hi = wilson(rejected, n)
    print("-" * 54)
    print(f"{'REJECTED':<16}{rejected:>7}{p:>9.1%}   [{lo:>6.1%}, {hi:>6.1%}]")
    print("\n  per proposal:")
    for cat, name, why in results:
        print(f"    {name:<22} {cat:<14} {why[:70]}")
    # How incomplete, for the ones that failed to be total. "missing 1 of 36"
    # and "missing all 36" are different failures: the first is a policy with a
    # gap, the second is not a policy at all.
    gaps = sorted((g, nm) for cat, nm, g in
                  ((c, nm, results_gap.get(nm)) for c, nm, _ in results)
                  if results_gap.get(nm))
    if gaps:
        print("\n  cells undetermined (of 36), for proposals that were not total:")
        for g, nm in gaps:
            print(f"    {nm:<22} {g:>2}  {'|' * min(g, 36)}")
    return counts


def run(fixture, backend, model, n_api):
    print("=" * 74)
    print("LLM AS PROPOSER -- rejection rate against a symbolic validator")
    print("=" * 74)
    prov = fixture.get("provenance", {})
    if backend == "fixture":
        print(f"\nbackend: fixture   model: {prov.get('model', '?')}")
        print(f"provenance: {prov.get('how', '')}")
        print(f"CAVEAT: {prov.get('caveat', '')}")

    # ---------------------------------------------------------- constrained
    res = []
    results_gap.clear()
    for item in fixture["constrained"]:
        table, err = compile_constrained(item["table"])
        cat, why = classify(table, err, None)
        results_gap[item["name"]] = _undetermined(table)
        res.append((cat, item["name"], why))
    report("CONSTRAINED: exact schema, model fills a 36-cell table", res,
           "Reliable but close to a lookup -- the model is barely reasoning.")

    # ------------------------------------------------------------- free-form
    res = []
    results_gap.clear()
    for item in fixture["freeform"]:
        table, err = compile_freeform(item["text"])
        cat, why = classify(table, err, None)
        results_gap[item["name"]] = _undetermined(table)
        res.append((cat, item["name"], why))
    report("FREE-FORM: prose, compiled by a DETERMINISTIC parser", res,
           "The compiler is symbolic on purpose: a second LLM pass would put the\n"
           "model back in the trust path, which the architecture forbids. So the\n"
           "compile-failure rate measures how far free-form description is from the\n"
           "formal representation -- which is the number we actually want.")

    # ---------------------------------------------------------------- repair
    kept = [i for i in fixture["repair"] if not i.get("excluded")]
    dropped = [i for i in fixture["repair"] if i.get("excluded")]
    res = []
    results_gap.clear()
    for item in kept:
        hist, card = tuple(item["hist"]), item["card"]
        table, err = compile_freeform(item["text"])
        obs = observations_for(hist, card)
        cat, why = classify(table, err, obs)
        results_gap[item["name"]] = _undetermined(table)
        res.append((cat, item["name"], why))
    counts = report("REPAIR: propose a NEW intent to explain a contradiction", res,
                    "The case where proposal earns its place: the observed play is\n"
                    "explained by NO intent in the vocabulary, so the vocabulary is by\n"
                    "definition incomplete. 'novel_valid' here means the proposal both\n"
                    "compiles AND actually covers the behaviour that broke the model.")
    if dropped:
        print(f"\n  excluded ({len(dropped)}): " +
              "; ".join(f"{i['name']} -- {i['excluded']}" for i in dropped))
    if kept:
        print(f"\n  COVERAGE: {counts['novel_valid']}/{len(kept)} proposed intents actually "
              f"explain the behaviour that refuted the model.")

    print("\n" + "=" * 74)
    print("The LLM proposed. The symbolic layer decided. Nothing entered the")
    print("hypothesis set without compiling to a total policy and surviving the")
    print("same consistency test the built-in intents face.")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="fixture", choices=["fixture", "api"])
    ap.add_argument("--fixture", default="fixtures/proposals.json")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--n", type=int, default=8, help="proposals per strategy (api backend)")
    args = ap.parse_args()

    if args.backend == "api":
        fx = {"provenance": {"model": args.model, "how": "live API"},
              "constrained": [], "freeform": [], "repair": []}
        for i in range(args.n):
            raw = api_call(CONSTRAINED_PROMPT, args.model)
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                fx["constrained"].append({"name": f"c{i}", "table": json.loads(m.group(0)) if m else {}})
            except json.JSONDecodeError as e:
                fx["constrained"].append({"name": f"c{i}", "table": {"__unparseable__": str(e)}})
            fx["freeform"].append({"name": f"f{i}", "text": api_call(FREEFORM_PROMPT, args.model)})
        json.dump(fx, open("fixtures/api_run.json", "w"), indent=2)
        print("wrote fixtures/api_run.json")
        run(fx, args.backend, args.model, args.n)
    else:
        run(load_fixture(args.fixture), args.backend, args.model, args.n)


if __name__ == "__main__":
    main()
