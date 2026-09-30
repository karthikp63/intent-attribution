#!/usr/bin/env python3
"""Out-of-vocabulary grid routing rule: follow a movable landmark.

The symbolic baseline has the seven pre-existing vocab.py features (all
ordered rules of length <= 2). A proposal may compose primitive variables
into a new score; it is checked against ALL shown trajectories before being
tested on held-back starts AND held-back landmark locations.

    python3 novel_rule_benchmark.py --selftest
    python3 novel_rule_benchmark.py --show-prompt
    python3 novel_rule_benchmark.py openai gpt-5-mini --samples 3
"""

import argparse
import ast
import itertools
import json

import gridworld as G
import proposer as P
import recorded_replay as R
import vocab as V

# Cosmetic L marker does not obstruct the route or replace the destination.
# Landmark locations, as well as starts, are disjoint across splits.
SHOWN_MARKERS = ((2, 0), (7, 4))
HELD_MARKERS = ((2, 6), (5, 1))
VARS = ("row", "col", "landmark_row", "landmark_col")
TRUTH = "abs(row-landmark_row)+abs(col-landmark_col)"


def episodes(starts, markers):
    return [(G.STARTS[i], d, closed, v, marker)
            for i in starts for d in sorted(G.DESTS)
            for closed in V.LAYOUTS for v in V.VANTS for marker in markers]


SHOWN = episodes(V.SHOWN_STARTS, SHOWN_MARKERS)
HELD = episodes(V.HELD_STARTS, HELD_MARKERS)


def compile_score(expr):
    """Tiny bounded arithmetic language. No Python eval, indexing, or calls
    except abs(). Reject expressions outside the language before execution.
    """
    if not isinstance(expr, str) or len(expr) > 120:
        raise ValueError("score must be a short expression string")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid score syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 35:
        raise ValueError("score too complex")

    def check(node):
        if isinstance(node, ast.Expression):
            check(node.body)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            check(node.left)
            check(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            check(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "abs" and len(node.args) == 1 \
                and not node.keywords:
            check(node.args[0])
        elif isinstance(node, ast.Name) and node.id in VARS:
            pass
        elif isinstance(node, ast.Constant) and type(node.value) is int \
                and -8 <= node.value <= 8:
            pass
        else:
            raise ValueError("unsupported score expression")

    check(tree)

    def value(node, env):
        if isinstance(node, ast.Expression):
            return value(node.body, env)
        if isinstance(node, ast.BinOp):
            a, b = value(node.left, env), value(node.right, env)
            return a + b if isinstance(node.op, ast.Add) else a - b
        if isinstance(node, ast.UnaryOp):
            n = value(node.operand, env)
            return n if isinstance(node.op, ast.UAdd) else -n
        if isinstance(node, ast.Call):
            return abs(value(node.args[0], env))
        if isinstance(node, ast.Name):
            return env[node.id]
        return node.value

    return lambda next_cell, marker: value(tree, dict(zip(VARS, (*next_cell, *marker))))


def proposal(raw):
    """Extract one JSON object, then compile score to a total safe policy."""
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(obj, dict) or set(obj) != {"rule", "score", "direction"}:
        raise ValueError("expected rule, score, direction")
    if not isinstance(obj["rule"], str) or not obj["rule"].strip():
        raise ValueError("missing plain-English rule")
    if obj["direction"] not in ("min", "max"):
        raise ValueError("direction must be min or max")
    return obj, compile_score(obj["score"])


def move(score, direction, pos, dest, closed, marker):
    if pos == dest:
        return "stop"
    dist = G.dist_field(dest, closed)
    if pos not in dist:
        return "stop"
    options = [(m, n) for m, n in G.neighbours(pos, closed)
               if dist.get(n, 1 << 20) == dist[pos] - 1]
    if not options:
        return "stop"
    values = [score(n, marker) for _, n in options]
    best = min(values) if direction == "min" else max(values)
    return min((m for (m, _), v in zip(options, values) if v == best),
               key=G.MOVE_ORDER.index)


def trace(score, direction, episode):
    start, dest_key, closed, _, marker = episode
    dest, pos, out = G.DESTS[dest_key], start, []
    for _ in range(G.STEP_CAP):
        action = move(score, direction, pos, dest, closed, marker)
        out.append(action)
        if action == "stop":
            break
        pos = G.step(pos, action)
    return tuple(out)


def observed_decisions(episode):
    start, dest_key, closed, vantage, marker = episode
    pos, prev = start, None
    for action in trace(compile_score(TRUTH), "min", episode):
        if action == "stop":
            break
        dist = G.dist_field(G.DESTS[dest_key], closed)
        choices = [(m, n) for m, n in G.neighbours(pos, closed)
                   if dist.get(n, 1 << 20) == dist[pos] - 1]
        if len(choices) > 1:
            yield (dest_key, closed, vantage, marker, pos, prev, choices, action)
        prev, pos = action, G.step(pos, action)


def baseline_candidates():
    atoms = list(itertools.product(V.CRITERIA, V.DIRECTIONS))
    return ([[]] + [[a] for a in atoms]
            + [[a, b] for a in atoms for b in atoms if a != b])


def baseline_fit():
    """Known DSL cannot depend on marker; exhaustive for <=2 criteria."""
    truth = [trace(compile_score(TRUTH), "min", e) for e in SHOWN]
    return [spec for spec in baseline_candidates()
            if all(V.trajectory_spec(spec, *e[:4]) == target
                   for e, target in zip(SHOWN, truth))]


def marker_contradiction():
    """An observed identical state requires two different moves as L changes.

    This proves *any* rule using only the old landmark-blind features fails,
    even if a longer criterion list than the 197 enumerated baselines is used.
    """
    grouped = {}
    for episode in SHOWN:
        for d, closed, v, marker, pos, prev, _, action in observed_decisions(episode):
            grouped.setdefault((d, closed, v, pos, prev), set()).add((marker, action))
    return next(((state, choices) for state, choices in grouped.items()
                 if len({action for _, action in choices}) > 1), None)


def diagnostic_examples():
    """Pick witnessed disagreements using only shown episodes and the baseline.

    Include matched states where changing the landmark changes the move; then
    cover landmarks and destinations with representative branching choices.
    """
    records = list(itertools.chain.from_iterable(map(observed_decisions, SHOWN)))
    grouped = {}
    for r in records:
        grouped.setdefault((r[0], r[1], r[2], r[4], r[5]), {})[r[3]] = r
    selected, distinct_positions = [], set()
    for key in grouped:
        group = grouped[key]
        if (len(group) == 2 and len({r[-1] for r in group.values()}) > 1
                and key[3] not in distinct_positions):
            selected.extend(group[m] for m in SHOWN_MARKERS)
            distinct_positions.add(key[3])
    # Additional examples show different destinations and both gate settings.
    for d in sorted(G.DESTS):
        for closed in V.LAYOUTS:
            for marker in SHOWN_MARKERS:
                found = next((r for r in records if r[0] == d and r[1] == closed
                              and r[3] == marker and r not in selected
                              and r[4] not in distinct_positions), None)
                if found:
                    selected.append(found)
                    distinct_positions.add(found[4])
    return selected


def prompt():
    lines = []
    for d, closed, v, marker, pos, prev, choices, chosen in diagnostic_examples():
        options = " ".join(f"{m}->{cell}" for m, cell in choices)
        lines.append(f"goal={d} closed_gate_mask={closed} observer={G.VANTAGES[v]} "
                     f"landmark={marker} at={pos} previous={prev or '-'} "
                     f"options=[{options}] subject_chose={chosen}")
    return f"""A subject moves on this 8x8 grid to A, B, or C (# walls, G gates).
The subject always takes a shortest route; its unknown rule breaks ties
between equally short steps. A movable landmark L is placed at the coordinates
stated in each example. L is passable and is not the goal. Gate mask 4 means
the third gate in reading order is closed; mask 0 means all gates are open.
Fixed move order N,E,S,W breaks any remaining ties.

{V.describe_grid()}

Existing rule vocabulary uses only: {', '.join(V.CRITERIA)}.
It does not explain the subject's shown behavior. These are selected decisions
from training episodes; all training decisions will be checked before acceptance.
Markers and starting positions on the final test are different from these.

{chr(10).join(lines)}

Suggest ONE new rule in words, and a numeric score for each candidate NEXT
square. You may use variables row, col, landmark_row, landmark_col; integer
constants -8..8; +, -, parentheses, and abs(...). The score is evaluated for
each shortest next step. Set direction to min or max. You may combine the
variables into a new expression; existing feature names are not required.
Return a JSON object exactly like this schema (the example is format only):
{{"rule":"prefer squares farther down","score":"row","direction":"max"}}
"""


def evaluate(ask, sample):
    raw, truncated = ask(prompt(), nonce=("novel-landmark-v1", sample))
    if truncated:
        return {"status": "rejected", "reason": "truncated"}
    try:
        obj, score = proposal(raw)
    except ValueError as exc:
        return {"status": "rejected", "reason": str(exc)}
    truth = compile_score(TRUTH)
    if any(trace(score, obj["direction"], e) != trace(truth, "min", e)
           for e in SHOWN):
        return {"status": "rejected", "reason": "fails shown episodes",
                "proposal": obj}
    held = all(trace(score, obj["direction"], e) == trace(truth, "min", e)
               for e in HELD)
    return {"status": "accepted", "held": held, "proposal": obj}


def selftest():
    assert len(SHOWN) == len(HELD) == 72
    assert not baseline_fit(), "old vocabulary fits all shown episodes"
    assert marker_contradiction(), "no direct witness for vocabulary mismatch"
    assert all(not G.blocked(marker, c) for marker in SHOWN_MARKERS + HELD_MARKERS
               for c in V.LAYOUTS)
    assert any(r[-1] != V.rule_move([], r[4], r[1], r[2], G.DESTS[r[0]], r[5])
               for r in diagnostic_examples())
    good = json.dumps({"rule": "prefer squares nearer the landmark",
                       "score": TRUTH, "direction": "min"})
    assert evaluate(lambda p, nonce: (good, False), 0)["held"]
    assert evaluate(lambda p, nonce: (good.replace('"min"', '"max"'), False),
                    0)["status"] == "rejected"
    for bad in ('{"rule":"hack","score":"__import__(\'os\').system(\'id\')","direction":"min"}',
                '{"rule":"hack","score":"row**1000000","direction":"min"}',
                '{"rule":"hack","score":"row[0]","direction":"min"}'):
        assert evaluate(lambda p, nonce: (bad, False), 0)["status"] == "rejected"
    print(f"PASS: {len(SHOWN)} shown, {len(HELD)} held; old-DSL fit 0; "
          f"{len(diagnostic_examples())} diagnostic choices; safe compiler checks")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("provider", nargs="?", choices=sorted(P.PROVIDERS))
    ap.add_argument("model", nargs="?")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--from-cache", action="store_true",
                    help="replay the checked-in model responses without an API key")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.show_prompt:
        print(prompt())
        return
    if not args.provider or not args.model or args.samples < 1:
        ap.error("supply provider, model, and positive --samples")
    print(f"Symbolic only: {'fit' if baseline_fit() else 'abstain'} "
          f"on {len(SHOWN)} shown episodes", flush=True)
    ask = (R.ask(args.provider, args.model) if args.from_cache else
           P.PROVIDERS[args.provider](args.model).ask)
    outcomes = []
    for i in range(args.samples):
        result = evaluate(ask, i)
        outcomes.append(result)
        print(f"sample #{i}: {result}", flush=True)
    print(f"LLM accepted {sum(o['status'] == 'accepted' for o in outcomes)}/"
          f"{len(outcomes)}; predicts all {len(HELD)} unseen episodes "
          f"{sum(o.get('held', False) for o in outcomes)}/{len(outcomes)}")


if __name__ == "__main__":
    main()
