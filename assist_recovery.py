#!/usr/bin/env python3
"""Experimental, feedback-guided vocabulary proposer.

The proposer only sees the original 36 shown trajectories and counterexamples
from those same trajectories. The 36 held-back episodes are used *only* for
the final evaluation. This is a new method, not a replacement for the v1/v2
results in RESULTS.md. No rule or response is accepted without exact symbolic
agreement on the shown trajectories.

    python3 assist_recovery.py --selftest
    python3 assist_recovery.py openai MODEL --samples 5 --revisions 2
"""

import argparse
import json
import re

import gridworld as G
import proposer as P
import vocab as V


def parse_spec(answer):
    """Extract one JSON list from an answer; reject invalid DSL entries."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(answer):
        if ch != "[":
            continue
        try:
            raw, _ = decoder.raw_decode(answer[i:])
        except json.JSONDecodeError:
            continue
        return V.compile_spec(raw)
    return None, "no JSON list found"


def first_mismatch(spec, rule):
    """Find a training counterexample, never looking at HELD_STARTS."""
    for start, dest_key, closed, vantage in V.episodes(V.SHOWN_STARTS):
        dest = G.DESTS[dest_key]
        pos, last = start, None
        for _ in range(G.STEP_CAP):
            if pos == dest:
                break
            observed = G.policy((dest_key, rule), pos, closed, vantage)
            predicted = V.rule_move(spec, pos, closed, vantage, dest, last)
            if predicted != observed:
                d = G.dist_field(dest, closed)
                choices = [m for m, nxt in G.neighbours(pos, closed)
                           if d.get(nxt, 1 << 20) == d[pos] - 1]
                return (f"From start {start} toward {dest_key}{dest}, with closed-gate "
                        f"mask {closed} and observer at {G.VANTAGES[vantage]}, the "
                        f"subject reached {pos} after last move {last}. Shortest "
                        f"legal next moves were {choices}. It chose {observed}; your "
                        f"rule predicts {predicted}.")
            last, pos = observed, G.step(pos, observed)
    return None


def recover(ask, rule, revisions=2, nonce=0):
    """Return a validated training fit or abstain; never inspect held-back data."""
    known = [r for r in G.RULES if r != rule]
    prompt = V.build_prompt(rule, known, True, version=2) + """

An empty list [] is a valid answer: it means no preference beyond shortest
path and the fixed move-order tie-break. Consider it before adding criteria.
Your final answer must contain one JSON list, including [] if appropriate.
"""
    log = []
    for attempt in range(revisions + 1):
        answer, truncated = ask(prompt, nonce=("assisted-v1", nonce, attempt))
        spec, error = (None, "response truncated") if truncated else parse_spec(answer)
        if error:
            log.append({"attempt": attempt, "error": error})
            feedback = f"Your previous answer could not be compiled: {error}."
        else:
            mismatch = first_mismatch(spec, rule)
            if mismatch is None:
                log.append({"attempt": attempt, "spec": spec, "fits_shown": True})
                return spec, log
            log.append({"attempt": attempt, "spec": spec, "fits_shown": False})
            feedback = mismatch
        if attempt < revisions:
            prompt += ("\nYour last answer was " + json.dumps(answer) + "\n" + feedback
                       + "\nRevise the rule. Use only the shown trajectories and this "
                         "counterexample. Return a single JSON list; [] is allowed.\n")
    return None, log


def baseline_result(ask, rule, nonce):
    """The unmodified v2 constrained prompt, evaluated after its one answer."""
    known = [r for r in G.RULES if r != rule]
    answer, truncated = ask(V.build_prompt(rule, known, True, version=2),
                            nonce=(2, nonce))
    if truncated:
        return False
    # Keep the v2 runner's parsing, even where it differs from our new parser.
    match = re.search(r"\[.*\]", answer, re.S)
    if match is None:
        return False
    try:
        spec, error = V.compile_spec(json.loads(match.group(0)))
    except json.JSONDecodeError:
        return False
    return not error and V.classify(spec, None, rule, known)[2]


def selftest():
    # Wrong initial guesses followed by corrections; the second answer is
    # allowed to depend on feedback, but cannot access held-back examples.
    calls = []

    def direct_model(prompt, nonce):
        calls.append(prompt)
        return ('[{"criterion":"openness","direction":"max"}]' if len(calls) == 1
                else "[]"), False

    spec, log = recover(direct_model, "direct", revisions=1)
    assert spec == [] and len(log) == 2 and "your rule predicts" in calls[1]
    assert log[0]["fits_shown"] is False and log[1]["fits_shown"] is True

    def wrong_model(prompt, nonce):
        return '[{"criterion":"openness","direction":"max"}]', False

    spec, log = recover(wrong_model, "wall_hug", revisions=1)
    assert spec is None and len(log) == 2  # fail closed
    assert first_mismatch(V.BUILTIN_SPEC["wall_hug"], "wall_hug") is None
    assert baseline_result(lambda prompt, nonce: ("[]", False), "direct", 0)
    assert not baseline_result(wrong_model, "wall_hug", 0)
    print("assisted proposer self-check PASS (training repair and abstention)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("provider", nargs="?", choices=sorted(P.PROVIDERS))
    ap.add_argument("model", nargs="?")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--revisions", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.provider or not a.model or a.samples < 1 or a.revisions < 0:
        ap.error("supply provider, model, positive samples and nonnegative revisions")
    ask = P.PROVIDERS[a.provider](a.model).ask
    counts = {r: {"baseline": 0, "accepted": 0, "recovered": 0,
                   "assisted_calls": 0} for r in G.RULES}
    for rule in G.RULES:
        for sample in range(a.samples):
            counts[rule]["baseline"] += baseline_result(ask, rule, sample)
            spec, log = recover(ask, rule, a.revisions, sample)
            counts[rule]["assisted_calls"] += len(log)
            if spec is None:
                print(f"{rule} #{sample}: abstained after {len(log)} attempts")
                continue
            counts[rule]["accepted"] += 1
            known = [r for r in G.RULES if r != rule]
            category, note, success = V.classify(spec, None, rule, known)
            counts[rule]["recovered"] += success
            print(f"{rule} #{sample}: {category} after {len(log)} attempts; {note}")
    print("Original v2 (one call) versus assisted (one to "
          f"{a.revisions + 1} calls); held-back episodes used only after answers:")
    for rule, c in counts.items():
        print(f"  {rule}: original {c['baseline']}/{a.samples} recovered; "
              f"assisted {c['recovered']}/{a.samples} recovered, "
              f"{c['accepted']}/{a.samples} fit shown, "
              f"{c['assisted_calls']} assisted calls")
    n = a.samples * len(G.RULES)
    print(f"TOTAL: original {sum(c['baseline'] for c in counts.values())}/{n}; "
          f"assisted {sum(c['recovered'] for c in counts.values())}/{n} "
          f"using {sum(c['assisted_calls'] for c in counts.values())} calls "
          f"versus {n} original calls")


if __name__ == "__main__":
    main()
