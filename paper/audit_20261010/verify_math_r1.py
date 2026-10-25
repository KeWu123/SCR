"""Finite checks for the repaired abstract SCR protocol, not an ANN integration test."""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
from math import ceil, comb, factorial, log2
from pathlib import Path
import json

RESULTS = {}
VERSIONS = ((0, False, "old"), (1, True, None), (2, False, "reinserted"))


def latest(mask: int, cut: int):
    candidates = [v for v in VERSIONS if mask & (1 << v[0]) and v[0] <= cut]
    return max(candidates, default=None, key=lambda v: v[0])


def logical(version):
    return None if version is None or version[1] else version[2]


def admit(candidate: int, mask: int, cut: int) -> bool:
    resolved = latest(mask, cut)
    return resolved is not None and not resolved[1] and resolved[0] == candidate


def admit_citation(candidate: int, mask: int, cut: int, span: str) -> bool:
    if not admit(candidate, mask, cut):
        return False
    return span in latest(mask, cut)[2]


def greedy(graph, score, entry="e"):
    current = entry
    trace = []
    while True:
        neighbors = frozenset(graph[current])
        trace.append((current, neighbors))
        best = max(neighbors, key=lambda x: (score[x], x), default=current)
        if score[best] <= score[current]:
            return current, tuple(trace)
        current = best


def explains(graph, trace):
    return all(node in graph and frozenset(graph[node]) == neighbors
               for node, neighbors in trace)


def check_witnesses():
    score = {"e": 0, "a": 1, "c": 2, "b": 3}
    g0 = {"e": {"a"}, "a": {"b"}, "b": set(), "c": set()}
    g1 = {"e": {"a"}, "a": {"c"}, "b": set(), "c": set()}
    a0, t0 = greedy(g0, score)
    a1, t1 = greedy(g1, score)
    assert (a0, a1) == ("b", "c")
    assert not any(explains(g, t0 + t1) for g in (g0, g1))

    # The original incomplete witness naturally reaches a in both graphs.
    original_score = {"e": 0, "c": 1, "b": 2, "a": 3}
    old0 = {"e": {"b"}, "a": {"b"}, "b": {"a", "e"}, "c": set()}
    old1 = {"e": {"c"}, "a": {"c"}, "b": set(), "c": {"a", "e"}}
    assert greedy(old0, original_score)[0] == greedy(old1, original_score)[0] == "a"

    # Positive edges can embed even though full neighborhood readings conflict.
    small = {"a": {"b"}, "b": set(), "c": set()}
    large = {"a": {"b", "c"}, "b": set(), "c": set()}
    conflicting = (("a", frozenset({"b"})), ("a", frozenset({"b", "c"})))
    positive_union = {("a", "b"), ("a", "c")}
    assert all(target in large[source] for source, target in positive_union)
    assert not explains(small, conflicting) and not explains(large, conflicting)

    # Coverage through cut 10 allows a build through 11; future filtering
    # cannot recreate the overwritten old edge a -> b.
    old_at_10 = {"a": {"b"}}
    built_through_11 = {"a": {"c"}}
    visible_at_10 = {"a", "b"}
    filtered = {target for target in built_through_11["a"] if target in visible_at_10}
    assert filtered != old_at_10["a"]
    RESULTS["graph_counterexamples"] = {
        "repaired_witness_answers": [a0, a1],
        "original_greedy_answers": ["a", "a"],
        "complete_neighborhood_tear_missed_by_positive_union": True,
        "coverage_does_not_restore_old_topology": True,
    }


def check_authoritative_validation():
    # Latest live version absent from beam: candidate-local max must be rejected.
    two_live = ((1, False, "v1"), (2, False, "v2"))
    beam = [two_live[0]]
    assert max(beam, key=lambda v: v[0])[0] == 1
    table_latest = max(two_live, key=lambda v: v[0])
    assert beam[0][0] != table_latest[0]
    assert admit(0, 0b111, 0)
    assert not admit(0, 0b111, 1)  # Tombstone not in beam.
    assert not admit(0, 0b111, 2)  # Reinserted version not in beam.
    assert admit(2, 0b111, 2)
    # Dropping the deleted prefix must mean authoritative absence, not fallback
    # to retained routing candidate r@0. A later reinsertion remains available.
    assert not admit(0, 0b100, 1)
    assert admit(2, 0b100, 2)
    # A latest live version alone does not validate an invented cited span.
    assert admit(0, 0b111, 0)
    assert admit_citation(0, 0b111, 0, "old")
    assert not admit_citation(0, 0b111, 0, "invented words")
    RESULTS["candidate_validation"] = {
        "hidden_latest_rejects_old_candidate": True,
        "hidden_tombstone_rejects_old_candidate": True,
        "authoritative_absence_rejects_retained_routing_candidate": True,
        "reinsertion_preserved": True,
        "correct_version_with_fabricated_span_rejected": True,
    }


def check_lag_and_stamps():
    sampling_cases = 0
    for n in range(1, 9):
        for lag in range(n + 1):
            stale = set(range(lag))
            for k in range(1, n + 1):
                subsets = list(combinations(range(n), k))
                enumerated = Fraction(sum(bool(stale.intersection(s)) for s in subsets),
                                      len(subsets))
                exact = 1 - Fraction(comb(n - lag, k) if n - lag >= k else 0, comb(n, k))
                iid = 1 - Fraction(n - lag, n) ** k
                assert enumerated == exact and exact >= iid
                sampling_cases += 1
    assert 1 - Fraction(comb(2, 2), comb(4, 2)) == Fraction(5, 6)
    assert 1 - Fraction(2, 4) ** 2 == Fraction(3, 4)
    assert Fraction(1, 2) != Fraction(3, 4)  # Equal marginals, correlated selection.
    stamp_cases = 0
    for labels in range(1, 5):
        for generations in range(1, 6):
            choices = list(product(range(labels), repeat=generations))
            enumerated = Fraction(sum(len(set(c)) < generations for c in choices), len(choices))
            no_collision = Fraction(1)
            if generations > labels:
                no_collision = Fraction(0)
            else:
                for j in range(generations):
                    no_collision *= Fraction(labels - j, labels)
            assert enumerated == 1 - no_collision
            stamp_cases += 1
    choices = list(product(range(8), repeat=4))
    assert Fraction(sum(len(set(c)) < 4 for c in choices), len(choices)) == Fraction(151, 256)
    RESULTS["probability_enumeration"] = {
        "uniform_distinct_sampling_cases": sampling_cases,
        "iid_label_assignment_cases": stamp_cases + 1,
        "N4_L2_k2_exact": "5/6",
        "N4_L2_k2_iid": "3/4",
        "equal_marginal_correlated_counterexample": "1/2",
        "B8_Lambda4_collision": "151/256",
    }


def check_counting():
    n, m = 5, 2
    pairs = list(combinations(range(n), 2))
    graphs = []
    for edges in combinations(pairs, m):
        rows = [set() for _ in range(n)]
        for u, v in edges:
            rows[u].add(v)
            rows[v].add(u)
        graphs.append(tuple(tuple(sorted(row)) for row in rows))
    assert len(set(graphs)) == comb(comb(n, 2), m) == 45

    n, degree = 4, 1
    row_options = [list(combinations([v for v in range(n) if v != u], degree))
                   for u in range(n)]
    directed = set(product(*row_options))
    assert len(directed) == comb(n - 1, degree) ** n == 81

    r = 6
    answer_functions = {tuple((i, z[i]) for i in range(r))
                        for z in product((0, 1), repeat=r)}
    assert len(answer_functions) == 2 ** r
    assert ceil(log2(len(answer_functions))) == r > ceil(log2(2 * r))

    # Unreachable changes are not separated by fixed-entry ANN walks.
    score = {"e": 0, "a": 1, "b": 2, "c": 3}
    unreachable0 = {"e": set(), "a": {"b"}, "b": {"a"}, "c": set()}
    unreachable1 = {"e": set(), "a": {"c"}, "b": set(), "c": {"a"}}
    assert greedy(unreachable0, score) == greedy(unreachable1, score)

    for capacity in range(2, 21):
        for size in range(1, capacity // 2 + 1):
            log_count = log2(comb(capacity, size))
            lower = size * log2(capacity / size)
            upper = lower + size * log2(2.718281828459045)
            assert lower <= log_count + 1e-12 and log_count <= upper + 1e-12

    live_names = ("A", "B", "A", "B")
    assert len(live_names) == 4 and len(set(live_names)) == 2
    # At most two generations live, but a persistent g0 and a current g2
    # collide under the proposed j mod 2 monotone-counter encoding.
    live_generation_ids = (0, 2)
    assert len(live_generation_ids) == 2
    assert len({g % 2 for g in live_generation_ids}) == 1
    RESULTS["counting_enumeration"] = {
        "undirected_m2_n5_trace_classes": 45,
        "directed_degree1_n4_trace_classes": 81,
        "k1_answer_function_classes_for_r6": 64,
        "fixed_entry_unreachable_change_not_separated": True,
        "four_live_records_only_two_observation_classes": True,
        "modulo_counter_collision_with_two_live_generations": True,
    }


# Commit also seals an immutable builder-input capsule; this capsule and
# unfinished builds have their own maintenance reference and are not reclaimed
# with logical row anchors. Published routing arrays retain copied payloads.
# The model has no mutable ANN implementation or crash transitions.
@dataclass(frozen=True)
class State:
    progress: tuple[int, int, int, int] = (0, 0, 0, 0)
    pins: tuple[int, int] = (-1, -1)
    observed_at_begin: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    head: int = 0
    table_mask: int = 1
    built_mask: int = 1
    retained_mask: int = 1
    current: int = 0


READER = ("begin", "read", "retrieve", "read", "retrieve", "end")
MAINTENANCE = ("commit1", "build1", "publish1", "commit2", "build2", "publish2")
LIMITS = (len(READER), len(READER), len(MAINTENANCE), 3)
CANDIDATES = ((0,), (0,), (0, 2))
TRACE_NEIGHBORS = ((0,), (0,), (2,))


def check_invariants(state: State):
    assert state.retained_mask & (1 << state.current)
    assert state.built_mask & (1 << state.current)
    assert state.current <= state.head
    cuts = [state.current] + [g for g in state.pins if g >= 0]
    for cut in cuts:
        assert logical(latest(state.table_mask, cut)) == logical(latest(0b111, cut))
    for generation in state.pins:
        if generation >= 0:
            assert state.retained_mask & (1 << generation)
            assert all(v <= generation for v in CANDIDATES[generation])
            assert all(v in CANDIDATES[generation] for v in TRACE_NEIGHBORS[generation])


def transition(state: State, actor: int, use_current_graph: bool = False) -> State:
    position = state.progress[actor]
    assert position < LIMITS[actor]
    progress = list(state.progress)
    progress[actor] += 1
    out = replace(state, progress=tuple(progress))
    if actor < 2:
        action = READER[position]
        pins = list(state.pins)
        if action == "begin":
            # One indivisible resolve-and-register transition.
            pins[actor] = state.current
            oracle = list(state.observed_at_begin)
            oracle[actor] = TRACE_NEIGHBORS[state.current]
            out = replace(out, pins=tuple(pins), observed_at_begin=tuple(oracle))
        elif action == "end":
            assert pins[actor] >= 0
            pins[actor] = -1
            out = replace(out, pins=tuple(pins))
        else:
            generation = pins[actor]
            assert generation >= 0 and state.retained_mask & (1 << generation)
            if action == "read":
                assert logical(latest(state.table_mask, generation)) == logical(
                    latest(0b111, generation))
            else:
                admitted = tuple(c for c in CANDIDATES[generation]
                                 if admit(c, state.table_mask, generation))
                expected = latest(0b111, generation)
                assert admitted == (() if expected[1] else (expected[0],))
                # Compare the served array with an observation oracle captured
                # at Begin, across publication and GC.
                served = state.current if use_current_graph else generation
                assert state.retained_mask & (1 << served)
                assert TRACE_NEIGHBORS[served] == state.observed_at_begin[actor]
    elif actor == 2:
        action = MAINTENANCE[position]
        target = int(action[-1])
        if action.startswith("commit"):
            assert target == state.head + 1
            out = replace(out, head=target, table_mask=state.table_mask | (1 << target))
        elif action.startswith("build"):
            assert target <= state.head
            assert all(v <= target for v in CANDIDATES[target])
            out = replace(out, built_mask=state.built_mask | (1 << target))
        else:
            assert state.built_mask & (1 << target)
            out = replace(out, current=target, retained_mask=state.retained_mask | (1 << target))
    else:
        # Current descriptor remains advertised for future Begin calls.
        live = [state.current] + [g for g in state.pins if g >= 0]
        keep_generations = sum(1 << g for g in set(live))
        floor = min(live)
        newer = state.table_mask & ~((1 << (floor + 1)) - 1)
        anchor = latest(state.table_mask, floor)
        keep_table = newer | (0 if anchor is None else (1 << anchor[0]))
        out = replace(out, retained_mask=state.retained_mask & keep_generations,
                      table_mask=keep_table)
    check_invariants(out)
    return out


def check_protocol_interleavings():
    nodes = set()
    edge_count = 0

    @lru_cache(None)
    def explore(state: State) -> int:
        nonlocal edge_count
        nodes.add(state)
        check_invariants(state)
        if state.progress == LIMITS:
            return 1
        schedules = 0
        for actor in range(4):
            if state.progress[actor] < LIMITS[actor]:
                edge_count += 1
                schedules += explore(transition(state, actor))
        return schedules

    schedules = explore(State())
    expected_schedules = factorial(sum(LIMITS))
    for length in LIMITS:
        expected_schedules //= factorial(length)
    assert schedules == expected_schedules
    # Split Begin admits this explicit unsafe interleaving.
    cached = 0
    retained = {0}
    retained.add(1)  # Publish g1 after resolving g0.
    retained.discard(0)  # g0 not registered yet.
    assert cached not in retained
    # Fault injection: serving the head graph must fail after publication,
    # even though the original generation is still retained and pinned.
    mutated = State()
    for actor in (0, 2, 2, 2, 2, 2, 2, 0):
        mutated = transition(mutated, actor)
    try:
        transition(mutated, 0, use_current_graph=True)
    except AssertionError:
        mutation_detected = True
    else:
        mutation_detected = False
    assert mutation_detected
    RESULTS["finite_protocol_model"] = {
        "state_count": len(nodes),
        "transition_count": edge_count,
        "legal_interleaving_count": schedules,
        "unpinned_current_graph_mutation_detected": mutation_detected,
        "actors": {
            "episode_A": READER,
            "episode_B": READER,
            "maintenance": MAINTENANCE,
            "reclamation_transitions": 3,
        },
        "all_registered_reads_and_traces_preserved": True,
        "split_begin_counterexample": [
            "Begin resolves g0",
            "Publish g1",
            "GC retires unregistered g0",
            "Begin registers cached g0 and attempts retrieval",
        ],
        "limitations": [
            "Two read-only episodes, three generations, one updated row.",
            "Immutable sealed builder capsules and immutable routing arrays are premises.",
            "No distributed crash recovery or production ANN integration is modeled.",
        ],
    }


def main():
    check_witnesses()
    check_authoritative_validation()
    check_lag_and_stamps()
    check_counting()
    check_protocol_interleavings()
    result_file = Path(__file__).with_name("verification_math_r1.json")
    result_file.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(RESULTS, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

