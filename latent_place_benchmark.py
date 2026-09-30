#!/usr/bin/env python3
"""Blinded fixed-place inference plus no-place controls.

No position is marked on the map, and no place is named in the LLM prompt.
Three policies secretly favor an unmarked coordinate; three controls do not.
The LLM proposes a score; the symbolic layer checks all observed episodes,
then scores a disjoint set of starts. Existing rules and exhaustive coordinate
search are measured separately. This remains a simulated benchmark.

    python3 latent_place_benchmark.py --selftest
    python3 latent_place_benchmark.py --show-prompt toward_2_0
    python3 latent_place_benchmark.py openai gpt-5-mini --samples 1
"""

import argparse
import ast
import json

import gridworld as G
import novel_rule_benchmark as N
import proposer as P
import recorded_replay as R
import vocab as V


# Fixed before any live call. None of these coordinates appear in prompts.
CASES = {
    "toward_2_0": ("place", (2, 0)),
    "toward_4_2": ("place", (4, 2)),
    "toward_7_4": ("place", (7, 4)),
    "direct": ("known", []),
    "wall_hug": ("known", [("openness", "min")]),
    "keep_heading": ("known", [("momentum", "max")]),
}
SHOWN = V.episodes(V.SHOWN_STARTS)
HELD = V.episodes(V.HELD_STARTS)


def anchor_score(point):
    return N.compile_score(f"abs(row-{point[0]})+abs(col-{point[1]})")


def traces_for_case(name, episodes):
    kind, arg = CASES[name]
    if kind == "place":
        score = anchor_score(arg)
        return tuple(N.trace(score, "min", (*episode, (0, 0)))
                     for episode in episodes)
    return tuple(V.trajectory_spec(arg, *episode) for episode in episodes)


def old_candidates():
    return N.baseline_candidates()


def old_fits(name):
    target = traces_for_case(name, SHOWN)
    return [spec for spec in old_candidates()
            if tuple(V.trajectory_spec(spec, *e) for e in SHOWN) == target]


def all_anchor_fits(name):
    target = traces_for_case(name, SHOWN)
    return [point for point in ((r, c) for r in range(G.R) for c in range(G.C))
            if tuple(N.trace(anchor_score(point), "min", (*e, (0, 0)))
                     for e in SHOWN) == target]


def branching_records(name):
    """Observed branching choices from shown episodes; test is never read."""
    for episode, moves in zip(SHOWN, traces_for_case(name, SHOWN)):
        start, dest_key, closed, vantage = episode
        pos, prev, dest = start, None, G.DESTS[dest_key]
        for action in moves:
            if action == "stop":
                break
            distances = G.dist_field(dest, closed)
            options = tuple((m, n) for m, n in G.neighbours(pos, closed)
                            if distances.get(n, 1 << 20) == distances[pos] - 1)
            if len(options) > 1:
                yield (dest_key, closed, vantage, pos, prev, options, action)
            pos, prev = G.step(pos, action), action


def diagnostic_choices(name):
    """Keep choices that eliminate most old or constant-location policies."""
    rows = list(branching_records(name))
    candidates = [("old", spec) for spec in old_candidates()] + [
        ("coordinate", (r, c)) for r in range(G.R) for c in range(G.C)]
    selected, used = [], set()

    def predicted(candidate, row):
        d, closed, vantage, pos, prev, _, _ = row
        kind, payload = candidate
        if kind == "old":
            return V.rule_move(payload, pos, closed, vantage, G.DESTS[d], prev)
        return N.move(anchor_score(payload), "min", pos, G.DESTS[d], closed,
                      (0, 0))

    while True:
        best, kept = None, candidates
        for i, row in enumerate(rows):
            if i in used:
                continue
            survivors = [c for c in candidates if predicted(c, row) == row[-1]]
            if len(survivors) < len(kept):
                best, kept = i, survivors
        if best is None:
            break
        selected.append(rows[best])
        used.add(best)
        candidates = kept
    # Give a few more contexts so the model can see any general pattern.
    for i, row in enumerate(rows):
        if len(selected) >= 12:
            break
        if i not in used and (row[3], row[0]) not in {(r[3], r[0]) for r in selected}:
            selected.append(row)
            used.add(i)
    return selected, candidates


def prompt(name, view="full"):
    target = traces_for_case(name, SHOWN)
    if view == "full":
        body = "\n".join(V.serialise_episode(*e, moves)
                         for e, moves in zip(SHOWN, target))
        intro = "Shown complete routes from two starting positions:"
    else:
        rows, _ = diagnostic_choices(name)
        body = "\n".join(
            f"goal={d} gates_closed={closed} observer={G.VANTAGES[v]} "
            f"at={pos} previous_move={prev or '-'} "
            + "options=[" + " ".join(f"{m}->{n}" for m, n in options)
            + f"] subject_chose={chosen}"
            for d, closed, v, pos, prev, options, chosen in rows)
        intro = ("Selected branching decisions from routes starting in two "
                 "positions. Every option is a shortest next step; each arrow "
                 "gives its destination square. Omitted decisions are checked "
                 "before acceptance:")
    return f"""A subject navigates this 8x8 grid to A, B, or C. # is a wall,
G is a gate, and N/S/E/W are moves. The subject ALWAYS takes a shortest route
to the destination; its unknown rule selects between equally short next steps.
When still tied, N,E,S,W is the fixed move order. Observer stays in the
specified square throughout each episode. Each coordinate is (row, col).

{V.describe_grid()}

{intro}
{body}

Propose ONE rule that predicts which shortest step the subject chooses, or
say you cannot tell. Describe it in plain English AND as a numeric score for
each possible NEXT square, choosing the smallest or largest score. You may
use variables row and col for that square, the names openness, observer_dist,
momentum, goal_row_align, goal_col_align, and integers -8..8 with +, -, abs(...),
and parentheses. 'openness' counts free adjacent cells; 'observer_dist' is
distance to the observer; 'momentum' is 1 if the direction is unchanged;
the goal alignment values increase as the next square's row or column gets
closer to the goal. You may compose row and col however you wish.
Return ONLY a JSON object with keys rule, score, direction, such as
{{"rule":"prefer more open squares","score":"openness","direction":"max"}}.
If the routes do not justify a rule, use score:null and direction:null.
"""


def parse_answer(raw):
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(obj, dict) or set(obj) != {"rule", "score", "direction"}:
        raise ValueError("expected rule, score, direction")
    score, direction = obj["score"], obj["direction"]
    if score is None and direction is None:
        return obj, None, None
    if not isinstance(obj["rule"], str) or not obj["rule"].strip():
        raise ValueError("missing rule description")
    if direction not in ("max", "min"):
        raise ValueError("bad direction")
    if isinstance(score, str) and score in V.CRITERIA:
        return obj, "known", score
    if not isinstance(score, str):
        raise ValueError("score must be a string or null")
    return obj, "compound", compile_context_score(score)


def compile_context_score(expr):
    """Safe arithmetic over coordinates and the old per-step features."""
    if len(expr) > 120:
        raise ValueError("score too long")
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
        elif isinstance(node, ast.Name) and node.id in {"row", "col", *V.CRITERIA}:
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

    def score(cell, closed, vantage, dest, action, previous):
        env = {"row": cell[0], "col": cell[1]}
        env.update({name: feature(cell, closed, vantage, dest, action, previous)
                    for name, feature in V.CRITERIA.items()})
        return value(tree, env)

    return score


def compound_trace(score, direction, episode):
    start, dest_key, closed, vantage = episode
    dest, pos, prev, out = G.DESTS[dest_key], start, None, []
    for _ in range(G.STEP_CAP):
        if pos == dest:
            out.append("stop")
            break
        distances = G.dist_field(dest, closed)
        options = [(m, n) for m, n in G.neighbours(pos, closed)
                   if distances.get(n, 1 << 20) == distances[pos] - 1]
        if not options:
            out.append("stop")
            break
        values = [score(n, closed, vantage, dest, m, prev) for m, n in options]
        best = min(values) if direction == "min" else max(values)
        move = min((m for (m, _), v in zip(options, values) if v == best),
                   key=G.MOVE_ORDER.index)
        out.append(move)
        pos, prev = G.step(pos, move), move
    return tuple(out)


def candidate_traces(obj, kind, compiled, episodes):
    if kind == "known":
        spec = [(compiled, obj["direction"])]
        return tuple(V.trajectory_spec(spec, *e) for e in episodes)
    return tuple(compound_trace(compiled, obj["direction"], e) for e in episodes)


def anchor_claim(obj, kind):
    """Conservative operational flag; the prose is retained for human review."""
    if kind != "compound":
        return False
    tree = ast.parse(obj["score"], mode="eval")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    uses_abs = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "abs" for n in ast.walk(tree))
    return uses_abs and {"row", "col"} <= names


def evaluate(ask, name, sample, view="full"):
    nonce = ("latent-place-v1", name, sample) if view == "full" else \
        ("latent-place-choices-v1", name, sample)
    raw, truncated = ask(prompt(name, view), nonce=nonce)
    if truncated:
        return {"status": "rejected", "reason": "truncated"}
    try:
        obj, kind, compiled = parse_answer(raw)
    except ValueError as exc:
        return {"status": "rejected", "reason": str(exc)}
    if kind is None:
        return {"status": "abstained", "place_case": CASES[name][0] == "place"}
    if candidate_traces(obj, kind, compiled, SHOWN) != traces_for_case(name, SHOWN):
        return {"status": "rejected", "reason": "fails shown episodes",
                "place_claim": anchor_claim(obj, kind), "proposal": obj}
    return {"status": "accepted", "held": candidate_traces(obj, kind, compiled, HELD)
            == traces_for_case(name, HELD), "place_claim": anchor_claim(obj, kind),
            "proposal": obj}


def selftest():
    assert len(SHOWN) == len(HELD) == 36
    for name, (kind, truth) in CASES.items():
        old, anchors = old_fits(name), all_anchor_fits(name)
        rows, survivors = diagnostic_choices(name)
        assert rows and len(rows) <= 12
        assert [payload for family, payload in survivors if family == "old"] == old
        assert [payload for family, payload in survivors if family == "coordinate"] == anchors
        assert "landmark" not in prompt(name, "choices").lower()
        if kind == "place":
            assert not old and anchors == [truth], (name, old, anchors)
            correct = json.dumps({"rule": "prefer proximity to an unmarked place",
                                  "score": f"abs(row-{truth[0]})+abs(col-{truth[1]})",
                                  "direction": "min"})
            result = evaluate(lambda p, nonce: (correct, False), name, 0)
            assert result["status"] == "accepted" and result["held"]
            assert result["place_claim"]
        else:
            assert old, name
            correct = json.dumps({"rule": "known rule", "score": truth[0][0]
                                  if truth else "row", "direction": truth[0][1]
                                  if truth else "min"})
            if truth:
                result = evaluate(lambda p, nonce: (correct, False), name, 0)
                assert result["status"] == "accepted" and result["held"]
            assert evaluate(lambda p, nonce: ('{"rule":"unsure","score":null,"direction":null}',
                                               False), name, 0)["status"] == "abstained"
    for bad in ('{"rule":"x","score":"row**100000","direction":"max"}',
                '{"rule":"x","score":"landmark_row","direction":"max"}'):
        assert evaluate(lambda p, nonce: (bad, False), "toward_2_0", 0)["status"] \
            == "rejected"
    combined = '{"rule":"row over col","score":"goal_row_align - goal_col_align","direction":"max"}'
    assert evaluate(lambda p, nonce: (combined, False), "toward_2_0", 0)["reason"] \
        == "fails shown episodes"
    print("PASS: 3 uniquely identifiable unmarked places, 3 no-place controls, "
          "36 shown / 36 held per case, expression checks")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("provider", nargs="?", choices=sorted(P.PROVIDERS))
    ap.add_argument("model", nargs="?")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show-prompt", choices=CASES)
    ap.add_argument("--view", choices=("full", "choices"), default="full")
    ap.add_argument("--from-cache", action="store_true",
                    help="replay the checked-in model responses without an API key")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.show_prompt:
        print(prompt(a.show_prompt, a.view))
        return
    if not a.provider or not a.model or a.samples < 1:
        ap.error("provide provider, model and positive --samples")
    if a.from_cache:
        ask = R.ask(a.provider, a.model)
    else:
        ask = P.PROVIDERS[a.provider](a.model).ask
    results = {}
    for name in CASES:
        print(f"{name}: old-DSL fits={len(old_fits(name))}, "
              f"64-place search fits={len(all_anchor_fits(name))}", flush=True)
        results[name] = []
        for sample in range(a.samples):
            result = evaluate(ask, name, sample, a.view)
            results[name].append(result)
            print(f"  sample #{sample}: {result}", flush=True)
    for group, names in (("place", list(CASES)[:3]), ("control", list(CASES)[3:])):
        outputs = [o for name in names for o in results[name]]
        correct = sum(bool(o.get("held") and o.get("place_claim")) for o in outputs) \
            if group == "place" else sum(not o.get("place_claim", False) for o in outputs)
        print(f"{group}: {correct}/{len(outputs)} "
              f"{'held-out place predictions' if group == 'place' else 'without place claims'}")


if __name__ == "__main__":
    main()
