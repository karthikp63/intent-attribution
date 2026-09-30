#!/usr/bin/env python3
"""Open rule proposals with explicit choices and simulated active probing.

No rule list, marker, or preferred location is shown to the model. It writes
up to two executable hypotheses using observable map values. An exact
controller picks an unseen TRAINING episode where those hypotheses disagree,
queries the simulated subject, and gives the model its actual branching
choices. The model can revise twice. A symbolic gate checks all 36 training
episodes before the 36 held-out episodes from different starting positions.

This is simulated active evidence collection, not a human experiment.
    python3 active_open_rule.py --selftest
    python3 active_open_rule.py openai gpt-5-mini --cases toward_2_0 toward_4_2 toward_7_4 direct
"""

import argparse
import collections
import json
import re

import gridworld as G
import latent_place_benchmark as B
import proposer as P
import recorded_replay as R
import vocab as V


def observations(episode, moves):
    start, goal, closed, vantage = episode
    pos, previous, dest = start, None, G.DESTS[goal]
    for action in moves:
        if action == "stop":
            break
        distances = G.dist_field(dest, closed)
        options = tuple((m, n) for m, n in G.neighbours(pos, closed)
                        if distances.get(n, 1 << 20) == distances[pos] - 1)
        if len(options) > 1:
            yield goal, closed, vantage, pos, previous, options, action
        pos, previous = G.step(pos, action), action


def format_observation(row):
    goal, closed, vantage, pos, previous, options, chosen = row
    opt = []
    for move, next_cell in options:
        vals = {name: V.CRITERIA[name](next_cell, closed, vantage,
                                       G.DESTS[goal], move, previous)
                for name in ("openness", "observer_dist", "momentum",
                             "goal_row_align", "goal_col_align")}
        opt.append(f"{move}->{next_cell} values={vals}")
    return (f"goal={goal}{G.DESTS[goal]} closed_gate_mask={closed} "
            f"observer={G.VANTAGES[vantage]} at={pos} "
            f"previous_move={previous or '-'} options=[{' ; '.join(opt)}] "
            f"subject_chose={chosen}")


def initial_rows(name):
    # Selection uses shown decisions only, exactly as in the earlier focused
    # prompt. No ground-truth coordinate is put in the prompt.
    return B.diagnostic_choices(name)[0][:8]


def build_prompt(name, revealed, last_feedback=None):
    examples = "\n".join(format_observation(r) for r in revealed)
    feedback = f"\nA fresh observed episode challenges your last answer: {last_feedback}\n" \
        if last_feedback else ""
    return f"""A subject moves on this 8x8 grid (# wall, G gate) to A, B, or C.
The subject ALWAYS follows a shortest route. Its unknown rule decides between
equally short next steps; remaining ties use fixed move order N,E,S,W. Observer
stays in the square stated in each episode. Coordinates are (row, col). A
closed gate mask of 4 means the third gate is closed; 0 means all are open.

{V.describe_grid()}

Observed choices (the subject's actual move is at the end of each row):
{examples}
{feedback}
Infer a rule from behavior. Propose up to TWO different explanations in
plain English; for each, also provide a numeric score for a possible NEXT
square, and whether the lowest or highest score is preferred. You are not
given a list of candidate rules or special locations. Your score may use
row, col, openness, observer_dist, momentum, goal_row_align, goal_col_align,
integer constants -8..8, +, -, abs(...), and parentheses. The values for
five features are in each observation; row and col are in each next square.
You may multiply a feature by an integer 0..8 to express relative priority.
Use the simplest explanations that account for the choices. If you cannot
propose a meaningful rule, return an empty list.

Return only JSON: {{"hypotheses":[{{"rule":"prefer open next squares",
"score":"openness","direction":"max"}}]}}
"""


def parse_hypotheses(raw):
    start = raw.find("{")
    if start < 0:
        return [], "missing JSON object"
    try:
        data, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return [], "invalid JSON"
    if not isinstance(data, dict) or set(data) != {"hypotheses"} or \
            not isinstance(data["hypotheses"], list):
        return [], "expected hypotheses array"
    if len(data["hypotheses"]) > 2:
        return [], "more than two hypotheses"
    proposals, errors = [], []
    for item in data["hypotheses"]:
        try:
            if not isinstance(item, dict) or not {"rule", "score", "direction"} <= set(item):
                raise ValueError("expected rule, score, direction")
            # Ignore explanatory fields: only the actual executable rule is
            # used. A generated 'next_square_score' is not ground truth.
            obj = {k: item[k] for k in ("rule", "score", "direction")}
            if isinstance(obj["score"], str):
                feature = r"(?:row|col|openness|observer_dist|momentum|goal_row_align|goal_col_align)"
                expression = obj["score"]
                expression = re.sub(r"\b([0-8])\s*\*\s*(" + feature + r")\b",
                                    lambda m: "(" + "+".join([m[2]] * int(m[1]))
                                    + ")" if m[1] != "0" else "0", expression)
                expression = re.sub(r"\b(" + feature + r")\s*\*\s*([0-8])\b",
                                    lambda m: "(" + "+".join([m[1]] * int(m[2]))
                                    + ")" if m[2] != "0" else "0", expression)
                obj["score"] = expression
            obj, kind, score = B.parse_answer(json.dumps(obj))
            if kind is not None:
                proposals.append((obj, kind, score))
        except ValueError as exc:
            errors.append(str(exc))
    return proposals, "; ".join(errors) if errors else None


def predicted_action(candidate, row):
    obj, kind, score = candidate
    d, closed, vantage, pos, prev, options, _ = row
    if kind == "known":
        return V.rule_move([(score, obj["direction"])], pos, closed,
                           vantage, G.DESTS[d], prev)
    values = [score(cell, closed, vantage, G.DESTS[d], m, prev)
              for m, cell in options]
    best = min(values) if obj["direction"] == "min" else max(values)
    return min((m for (m, _), val in zip(options, values) if val == best),
               key=G.MOVE_ORDER.index)


def choose_probe(name, candidates, used):
    """Select an unseen episode from candidate predictions ONLY.

    The subject's answers are accessed only after the episode is chosen.
    If every hypothesis predicts the same thing, query a novel episode to
    check them. Tie-breaking depends on episode metadata, never its true moves.
    """
    available = [(i, e) for i, e in enumerate(B.SHOWN) if i not in used]
    if not available:
        return None
    # Repeatedly moving only the observer, while preserving the same start,
    # goal and gates, often produces the same path. Prefer a distinct scenario.
    used_settings = {(B.SHOWN[i][0], B.SHOWN[i][1], B.SHOWN[i][2]) for i in used}
    diverse = [(i, e) for i, e in available if e[:3] not in used_settings]
    if diverse:
        available = diverse

    def score(episode):
        predicted = [B.candidate_traces(obj, kind, rule, [episode])[0]
                     for obj, kind, rule in candidates]
        counts = collections.Counter(predicted)
        return len(counts), -sum(n * n for n in counts.values())

    return max(available, key=lambda item: (score(item[1]), -item[0]))


def evaluate(ask, name, sample=0, revisions=2):
    revealed = list(initial_rows(name))
    used, record, last_feedback = set(), [], None
    target_shown = B.traces_for_case(name, B.SHOWN)
    for attempt in range(revisions + 1):
        nonce = ("active-open-v2", name, sample, attempt)
        raw, truncated = ask(build_prompt(name, revealed, last_feedback), nonce=nonce)
        candidates, error = ([], "truncated") if truncated else parse_hypotheses(raw)
        fits = [(obj, kind, score) for obj, kind, score in candidates
                if B.candidate_traces(obj, kind, score, B.SHOWN) == target_shown]
        record.append({"attempt": attempt, "valid": len(candidates),
                       "training_fits": len(fits), "error": error})
        if fits:
            selected = fits[0]
            obj, kind, score = selected
            held = B.candidate_traces(obj, kind, score, B.HELD) \
                == B.traces_for_case(name, B.HELD)
            return {"status": "accepted", "held": held,
                    "place_claim": B.anchor_claim(obj, kind),
                    "proposal": obj, "log": record}
        if attempt == revisions:
            break
        probe = choose_probe(name, candidates, used)
        if probe is None:
            break
        idx, episode = probe
        used.add(idx)
        rows = list(observations(episode, target_shown[idx]))
        revealed.extend(rows)
        disagreements = []
        for j, candidate in enumerate(candidates):
            mismatch = next((r for r in rows
                             if predicted_action(candidate, r) != r[-1]), None)
            if mismatch:
                disagreements.append(f"hypothesis {j + 1} predicted "
                                     f"{predicted_action(candidate, mismatch)} "
                                     f"at {mismatch[3]}, actual {mismatch[-1]}")
        last_feedback = (f"new start={episode[0]} goal={episode[1]} "
                         f"gates_closed={episode[2]} observer={G.VANTAGES[episode[3]]}. "
                         + ("; ".join(disagreements) if disagreements
                            else "Earlier hypotheses still need validation."))
        record[-1]["probe"] = {"start": episode[0], "goal": episode[1],
                                "closed": episode[2], "vantage": episode[3],
                                "branching_moves": len(rows),
                                "disagreed": len(disagreements)}
    return {"status": "abstained", "log": record}


def selftest():
    name = "toward_4_2"
    rows = initial_rows(name)
    assert rows and all(r[5] and r[6] in {m for m, _ in r[5]} for r in rows)
    assert "landmark" not in build_prompt(name, rows).lower()
    wrong = {"rule": "prefer goal column", "score": "goal_col_align",
             "direction": "max"}
    correct = {"rule": "prefer a fixed place", "score": "abs(row-4)+abs(col-2)",
               "direction": "min"}
    def fake(prompt, nonce):
        assert "landmark" not in prompt.lower()
        return json.dumps({"hypotheses": [wrong if nonce[-1] == 0 else correct]}), False
    result = evaluate(fake, name, revisions=1)
    assert result["held"] and result["place_claim"]
    assert result["log"][0]["probe"]["branching_moves"] > 0
    assert result["log"][0]["probe"]["disagreed"] > 0
    proposals, _ = parse_hypotheses(json.dumps({"hypotheses": [wrong]}))
    first = choose_probe(name, proposals, set())
    second = choose_probe(name, proposals, {first[0]})
    assert first[1][:3] != second[1][:3]
    fail = evaluate(lambda prompt, nonce: (json.dumps({"hypotheses": [wrong]}), False),
                    name, revisions=1)
    assert fail["status"] == "abstained"
    assert evaluate(lambda prompt, nonce: ('{"hypotheses":[]}', False),
                    name, revisions=0)["status"] == "abstained"
    control = {"rule": "prefer fewest free neighbors", "score": "openness",
               "direction": "min"}
    good = evaluate(lambda prompt, nonce: (json.dumps({"hypotheses": [control]}), False),
                    "wall_hug", revisions=0)
    assert good["held"] and not good["place_claim"]
    extra = dict(control, next_square_score=3)
    assert len(parse_hypotheses(json.dumps({"hypotheses": [extra]}))[0]) == 1
    weighted = {"rule": "keep heading", "score": "8*momentum+openness",
                "direction": "max"}
    assert len(parse_hypotheses(json.dumps({"hypotheses": [weighted]}))[0]) == 1
    assert not parse_hypotheses('{"hypotheses":[{"rule":"x",'
                                '"score":"__import__(\"os\")","direction":"max"}]}')[0]
    print("PASS: open proposals, simulated active episode, counterexample repair, "
          "no-place control, held-out separation, safe compiler")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("provider", nargs="?", choices=sorted(P.PROVIDERS))
    ap.add_argument("model", nargs="?")
    ap.add_argument("--cases", nargs="+", choices=B.CASES, default=list(B.CASES))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--revisions", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--from-cache", action="store_true",
                    help="replay the checked-in model responses without an API key")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.provider or not args.model or args.samples < 1 or args.revisions < 0:
        ap.error("provide provider, model, positive samples and nonnegative revisions")
    if args.from_cache:
        ask = R.ask(args.provider, args.model, args.max_tokens)
    else:
        ask = P.PROVIDERS[args.provider](args.model,
                                         max_tokens=args.max_tokens).ask
    for name in args.cases:
        for sample in range(args.samples):
            result = evaluate(ask, name, sample, args.revisions)
            print(f"{name} #{sample}: {result}", flush=True)


if __name__ == "__main__":
    main()
