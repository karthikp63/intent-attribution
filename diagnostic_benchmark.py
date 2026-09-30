#!/usr/bin/env python3
"""Fresh-rule comparison: full trajectories vs. explicit branching decisions.

These four rules were selected before calling a model. They are absent from
the original four-rule LLM benchmark. Both prompts use the SAME 36 shown
episodes; neither prompt contains any of the 36 held-back episodes. The new
prompt uses a minimal set of branching decisions chosen from the shown data
by exact candidate elimination, and lists feature values for each option.
No revisions or extra calls.

    python3 diagnostic_benchmark.py --selftest
    python3 diagnostic_benchmark.py openai gpt-5-mini --samples 2
"""

import argparse
import itertools

import assist_recovery as A
import gridworld as G
import proposer as P
import recorded_replay as R
import vocab as V


CASES = {
    "keep_heading": [("momentum", "max")],
    "change_heading": [("momentum", "min")],
    "approach_observer": [("observer_dist", "min")],
    "wall_then_straight": [("openness", "min"), ("momentum", "max")],
}
FEATURES = tuple(V.CRITERIA)


def shown_episodes():
    return V.episodes(V.SHOWN_STARTS)


def held_episodes():
    return V.episodes(V.HELD_STARTS)


def common_head():
    known = "\n".join(f"  - {r}: {G.RULE_GLOSS[r]}" for r in G.RULES)
    return f"""A subject navigates an 8x8 grid to A, B or C. # is a wall and G
is a gate; N/S/E/W are moves. The subject ALWAYS follows a SHORTEST route.
Their unknown rule only chooses between equally short next steps.
The observer stays in the stated square for each episode.

{V.describe_grid()}

Known rules that do not explain this subject's behavior:
{known}

"""


def full_prompt(spec):
    lines = [V.serialise_episode(*e, V.trajectory_spec(spec, *e))
             for e in shown_episodes()]
    return (common_head() + "The subject's complete shown trajectories:\n"
            + "\n".join(lines) + "\n\nPropose one NEW routing rule.\n"
            + V._V2_CONSTRAINED)


def observed_choices(spec):
    """Every branching state actually visited on SHOWN trajectories."""
    records = []
    for start, dest_key, closed, vantage in shown_episodes():
        dest = G.DESTS[dest_key]
        pos, last = start, None
        for _ in range(G.STEP_CAP):
            if pos == dest:
                break
            dist = G.dist_field(dest, closed)
            choices = [(m, nxt) for m, nxt in G.neighbours(pos, closed)
                       if dist.get(nxt, 1 << 20) == dist[pos] - 1]
            chosen = V.rule_move(spec, pos, closed, vantage, dest, last)
            if len(choices) > 1:
                records.append((dest_key, closed, vantage, pos, last,
                                tuple(choices), chosen))
            last, pos = chosen, G.step(pos, chosen)
    return records


def diagnostic_choices(spec):
    """Keep observed choices that rule out the most candidate DSL policies.

    Selection sees the subject's SHOWN decisions only. Candidate predictions
    are computed at the observed state (teacher forcing), not on held-back
    episodes. This makes the prompt concise without selecting on test data.
    """
    records = observed_choices(spec)
    atoms = list(itertools.product(FEATURES, V.DIRECTIONS))
    candidates = ([[]] + [[x] for x in atoms]
                  + [[x, y] for x in atoms for y in atoms if x != y])
    selected = []
    while True:
        best, survivors = None, candidates
        for record in records:
            if record in selected:
                continue
            dest_key, closed, vantage, pos, last, _, chosen = record
            kept = [candidate for candidate in candidates
                    if V.rule_move(candidate, pos, closed, vantage,
                                   G.DESTS[dest_key], last) == chosen]
            if len(kept) < len(survivors):
                best, survivors = record, kept
        if best is None:
            break
        selected.append(best)
        candidates = survivors
    return selected


def choice_lines(spec):
    """Render the selected branching observations with explicit feature data."""
    lines = []
    for dest_key, closed, vantage, pos, last, choices, chosen in diagnostic_choices(spec):
        options = []
        for m, nxt in choices:
            vals = ",".join(str(V.CRITERIA[f](nxt, closed, vantage,
                                                 G.DESTS[dest_key], m, last))
                            for f in FEATURES)
            options.append(f"{m}@{nxt}:[{vals}]")
        lines.append(f"{dest_key} closed={closed} observer={G.VANTAGES[vantage]} "
                     f"at={pos} after={last or '-'}: "
                     + " ".join(options) + f" => {chosen}")
    return lines


def choices_prompt(spec):
    header = ("Only DIAGNOSTIC CHOICES from the shown episodes are listed "
              "below. Each has at least two shortest next steps. An exact "
              "symbolic selection retained the choices that eliminate the "
              "most candidate rules; omitted choices are consistent with "
              "every rule still standing.\n"
              "Each option gives its next square and feature values in this "
              f"order: {', '.join(FEATURES)}. The arrow marks the actual move. "
              "'after' is the previous move.\n\n")
    return (common_head() + header + "\n".join(choice_lines(spec))
            + "\n\nPropose one NEW routing rule.\n"
            + V._V2_CONSTRAINED.replace("trajectories above", "choices above"))


def fits(spec, truth, episodes):
    return all(V.trajectory_spec(spec, *e) == V.trajectory_spec(truth, *e)
               for e in episodes)


def evaluate(ask, truth, kind, sample):
    prompt = full_prompt(truth) if kind == "full" else choices_prompt(truth)
    answer, truncated = ask(prompt, nonce=("diagnostic-v1", kind, sample))
    spec, error = (None, "truncated") if truncated else A.parse_spec(answer)
    if error:
        return {"spec": None, "shown": False, "held": False, "error": error}
    return {"spec": spec, "shown": fits(spec, truth, shown_episodes()),
            "held": fits(spec, truth, held_episodes()), "error": None}


def selftest():
    for name, truth in CASES.items():
        assert fits(truth, truth, shown_episodes() + held_episodes()), name
        assert not any(fits(truth, known, shown_episodes())
                       for known in V.BUILTIN_SPEC.values()), name
        # Candidate selection is built only from shown decisions; every
        # remaining short DSL candidate must match ALL shown trajectories.
        assert choice_lines(truth), name
        atoms = list(itertools.product(FEATURES, V.DIRECTIONS))
        candidates = ([[]] + [[x] for x in atoms]
                      + [[x, y] for x in atoms for y in atoms if x != y])
        selected = diagnostic_choices(truth)
        for candidate in candidates:
            if all(V.rule_move(candidate, pos, closed, vantage, G.DESTS[d], last)
                   == chosen for d, closed, vantage, pos, last, _, chosen
                   in selected):
                assert fits(candidate, truth, shown_episodes()), (name, candidate)
        for episode in shown_episodes():
            assert V.trajectory_spec(truth, *episode)[-1] == "stop", name
    # A wrong candidate must be rejected even if it sounds plausible.
    result = evaluate(lambda prompt, nonce: ('[]', False),
                      CASES["keep_heading"], "choices", 0)
    assert not result["shown"] and not result["held"]
    print("fresh-rule prompt and held-back evaluation checks PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("provider", nargs="?", choices=sorted(P.PROVIDERS))
    ap.add_argument("model", nargs="?")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--from-cache", action="store_true",
                    help="replay the checked-in model responses without an API key")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.provider or not args.model or args.samples < 1:
        ap.error("supply provider, model and a positive number of samples")
    ask = (R.ask(args.provider, args.model) if args.from_cache else
           P.PROVIDERS[args.provider](args.model).ask)
    totals = {kind: {"shown": 0, "held": 0, "n": 0} for kind in ("full", "choices")}
    for name, truth in CASES.items():
        for sample in range(args.samples):
            for kind in ("full", "choices"):
                result = evaluate(ask, truth, kind, sample)
                totals[kind]["n"] += 1
                totals[kind]["shown"] += result["shown"]
                totals[kind]["held"] += result["shown"] and result["held"]
                print(f"{name} #{sample} {kind}: {result}", flush=True)
    print("TOTAL: training fit and exact recovery on unseen episodes")
    for kind, t in totals.items():
        print(f"  {kind}: shown {t['shown']}/{t['n']}; "
              f"unseen {t['held']}/{t['n']}")


if __name__ == "__main__":
    main()
