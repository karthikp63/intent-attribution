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
import hashlib
import time
import urllib.request
from collections import Counter

import random

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


# ---------------------------------------------------------------- proposers
#
# One interface, several providers. The key comes from the environment only --
# never a flag, never a file in the repo. Every response is cached to disk by a
# hash of (provider, model, prompt), so a re-run costs nothing, a crash loses
# nothing, and the exact bytes a number was computed from stay on disk.

CACHE_DIR = "fixtures/cache"


def _cache_path(provider, model, prompt, nonce=None):
    """Cache key is (provider, model, prompt, nonce).

    The nonce is NOT part of the prompt -- it exists so that repeated samples of
    the SAME prompt get distinct cache entries. Without it every re-draw would
    hit the first cached response and the reported variance would be zero by
    construction, which would look like a very stable model."""
    h = hashlib.sha256(f"{provider}\x00{model}\x00{prompt}\x00{nonce}".encode()
                       ).hexdigest()[:24]
    return os.path.join(CACHE_DIR, f"{provider}-{h}.json")


class Proposer:
    """Base: caching, retry, and truncation detection are shared by every provider."""

    name = "base"
    env_key = None

    def __init__(self, model, max_tokens=4096, retries=4, use_cache=True):
        self.model, self.max_tokens = model, max_tokens
        self.retries, self.use_cache = retries, use_cache
        self.key = os.environ.get(self.env_key) if self.env_key else None
        if self.env_key and not self.key:
            raise SystemExit(f"{self.env_key} is not set; use --backend fixture")

    def _post(self, prompt):
        raise NotImplementedError

    def ask(self, prompt, nonce=None):
        """-> (text, truncated). Cached, retried, and never silently truncated."""
        path = _cache_path(self.name, self.model, prompt, nonce)
        if self.use_cache and os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            return d["text"], d["truncated"]
        last = None
        for attempt in range(self.retries):
            try:
                text, truncated = self._post(prompt)
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(path, "w") as f:
                    json.dump({"provider": self.name, "model": self.model,
                               "nonce": nonce, "prompt": prompt, "text": text,
                               "truncated": truncated}, f, indent=2)
                return text, truncated
            except Exception as e:                     # noqa: BLE001 - provider-agnostic
                last = e
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"{self.name} failed after {self.retries} attempts: {last}")


class Anthropic(Proposer):
    name, env_key = "anthropic", "ANTHROPIC_API_KEY"

    def _post(self, prompt):
        body = json.dumps({"model": self.model, "max_tokens": self.max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"content-type": "application/json", "x-api-key": self.key,
                     "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        return d["content"][0]["text"], d.get("stop_reason") == "max_tokens"


class Gemini(Proposer):
    name, env_key = "gemini", "GEMINI_API_KEY"

    def _post(self, prompt):
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.key}")
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                           "generationConfig": {"maxOutputTokens": self.max_tokens}}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        cand = d["candidates"][0]
        text = "".join(p.get("text", "") for p in cand["content"]["parts"])
        return text, cand.get("finishReason") == "MAX_TOKENS"


class Ollama(Proposer):
    """Local, so no key. Host from OLLAMA_HOST, default localhost."""
    name, env_key = "ollama", None

    def _post(self, prompt):
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"num_predict": self.max_tokens}}).encode()
        req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        return d["response"], d.get("done_reason") == "length"


PROVIDERS = {"anthropic": Anthropic, "gemini": Gemini, "ollama": Ollama}
DEFAULT_MODEL = {"anthropic": "claude-opus-5", "gemini": "gemini-3.1-pro-preview",
                 "ollama": "llama3.1"}


# ------------------------------------------- real contradiction histories
#
# The repair case needs observations the vocabulary genuinely cannot explain.
# These are drawn from a seeded out_of_model run rather than invented, so the
# prompt describes something that actually happened.

def contradiction_cases(n=6, seed=1, hands=600):
    rng, orng = random.Random(seed), random.Random(seed + 10 ** 6)
    state = {"refutations": 0}
    seen, out = set(), []
    for k in range(hands):
        r = L.play_hand("adaptive", rng, hand_no=k + 1, subject="misfit:out_of_model",
                        state=state, orng=orng)
        if not r["contradiction"]:
            continue
        key = (tuple(r["history"]), r["subject_card"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"hist": list(r["history"]), "card": r["subject_card"]})
        if len(out) >= n:
            break
    return out


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
        table, err = compile_constrained(item.get("table", {}))
        if item.get("truncated"):
            err = "response truncated by max_tokens -- not a model failure"
        cat, why = classify(table, err, None)
        results_gap[item["name"]] = _undetermined(table)
        res.append((cat, item["name"], why))
    report("CONSTRAINED: exact schema, model fills a 36-cell table", res,
           "Reliable but close to a lookup -- the model is barely reasoning.")

    # ------------------------------------------------------------- free-form
    res = []
    results_gap.clear()
    for item in fixture["freeform"]:
        table, err = compile_freeform(item.get("text", ""))
        if item.get("truncated"):
            err = "response truncated by max_tokens -- not a model failure"
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
        table, err = compile_freeform(item.get("text", ""))
        if item.get("truncated"):
            err = "response truncated by max_tokens -- not a model failure"
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


def live_run(provider, model, n, no_cache):
    """Generate proposals from a real model. All three strategies, including
    REPAIR -- which the previous version defined a prompt for and never called,
    so the only scientifically interesting case silently reported n = 0."""
    P = PROVIDERS[provider](model or DEFAULT_MODEL[provider], use_cache=not no_cache)
    fx = {"provenance": {"model": P.model, "provider": provider,
                         "how": "live API via proposer.py", "n_requested": n},
          "constrained": [], "freeform": [], "repair": []}
    truncated = 0

    for i in range(n):
        raw, tr = P.ask(CONSTRAINED_PROMPT)
        truncated += tr
        m = re.search(r"\{.*\}", raw, re.S)
        entry = {"name": f"c{i}", "truncated": tr}
        try:
            entry["table"] = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError as e:
            entry["table"] = {}
            entry["parse_error"] = str(e)
        fx["constrained"].append(entry)

        raw, tr = P.ask(FREEFORM_PROMPT)
        truncated += tr
        fx["freeform"].append({"name": f"f{i}", "text": raw, "truncated": tr})

    for j, case in enumerate(contradiction_cases(n=min(n, 6))):
        hist, card = tuple(case["hist"]), case["card"]
        prompt = REPAIR_PROMPT.format(observed=serialise(hist, card))
        raw, tr = P.ask(prompt)
        truncated += tr
        fx["repair"].append({"name": f"r{j}", "text": raw, "truncated": tr,
                             "hist": case["hist"], "card": card})

    fx["provenance"]["truncated_responses"] = truncated
    os.makedirs("fixtures", exist_ok=True)
    out = f"fixtures/{provider}_run.json"
    with open(out, "w") as f:
        json.dump(fx, f, indent=2)
    print(f"wrote {out}  ({truncated} truncated responses)")
    return fx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="fixture",
                    choices=["fixture", "live"],
                    help="fixture: replay a recorded run; live: call a model")
    ap.add_argument("--proposer", default="anthropic", choices=sorted(PROVIDERS),
                    help="provider for --backend live; key from the environment only")
    ap.add_argument("--fixture", default=None,
                    help="path to a recorded run to score")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=8, help="proposals per strategy")
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass the on-disk response cache")
    args = ap.parse_args()

    if args.backend == "live":
        fx = live_run(args.proposer, args.model, args.n, args.no_cache)
        run(fx, "live", fx["provenance"]["model"], args.n)
        return

    if not args.fixture:
        print(__doc__)
        print("\nNo recorded run supplied and no live run requested.\n")
        print("STATUS: pipeline built and validated against ground truth; the")
        print("proposer itself is NOT yet measured. All 7 built-in Leduc intents")
        print("compile to total, legal 36-cell tables:\n")
        for name_, t in BUILT_IN.items():
            err = check_total(t)
            print(f"  {name_:<14} {'OK' if err is None else err}")
        print(f"\n  decision cells per policy: {len(CELLS)}")
        print("\nTo measure a real rejection rate:")
        print("  export ANTHROPIC_API_KEY=...   # or GEMINI_API_KEY; ollama needs none")
        print("  python3 proposer.py --backend live --proposer anthropic --n 20")
        print("\nEarlier drafts reported 12.5%/60%/75% from an in-session fixture.")
        print("Those measured the author, not a model, and have been withdrawn.")
        return

    run(load_fixture(args.fixture), "fixture", args.model or "?", args.n)


if __name__ == "__main__":
    main()
