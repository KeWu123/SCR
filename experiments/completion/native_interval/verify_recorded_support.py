"""Check retained support/answer records and source identities without native DBs."""
from pathlib import Path
import hashlib
import json
from _evidence import load_record, ROOT

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(record, entries):
    for field, path in entries:
        assert record[field] == sha(path), f'Measurement source mismatch: {field}'


def visible(events, cut):
    state = {}
    for event in events:
        if event['epoch'] <= cut:
            previous = state.get(event['row'])
            if previous is None or previous['epoch'] < event['epoch']:
                state[event['row']] = event
    return {row: event for row, event in state.items() if not event['deleted']}


def answers(rows, state):
    assert len(rows) <= 10 and len({row for row, version, payload in rows}) == len(rows)
    for row, version, payload in rows:
        event = state[row]
        assert (version, payload) == (event['version'], event['payload'])


def main():
    pointer = load_record(HERE / 'latest.json')
    assert sha(pointer['results']) == pointer['sha256']
    result = load_record(pointer['results'])
    source_hashes(result, [
        ('protocol_sha256', HERE / 'protocol.md'),
        ('driver_sha256', HERE / 'run_native_intervals.py'),
        ('extension_sha256', HERE / 'interval_broker.py'),
        ('base_broker_sha256', HERE.parent.parent / 'live_broker.py'),
        ('support_compiler_sha256', HERE.parent / 'support_spectrum.py'),
    ])
    counts = {'disconnected_cases': 0, 'continuing_cases': 0, 'answer_records': 0,
              'scaling_cells': 0, 'input_hashes': 0}
    for cell in result['disconnected']:
        events = load_record(cell['lineage_file'])
        assert len(events) == 3200
        population = {}
        for cut in [0, 1, 4, 7, 8]:
            population.update({(row, e['version']): e['vector_hex']
                               for row, e in visible(events, cut).items()})
        support = [cut for cut in range(9) if all(
            population.get((row, e['version'])) == e['vector_hex']
            for row, e in visible(events, cut).items())]
        assert support == [0, 1, 4, 7, 8]
        for arm in cell['arms'].values():
            assert arm['support'] == [[0, 1], [4, 4], [7, 8]]
            assert [test['cut'] for test in arm['tests'] if test['accepted']] == support
            assert [test['cut'] for test in arm['tests'] if not test['accepted']] == [2, 3, 5, 6]
            for test in arm['tests']:
                if test['accepted']:
                    for record in test['queries']:
                        answers(record['answers'], visible(events, record['cut']))
                        counts['answer_records'] += 1
            for field, cut in [('inflight_answer', 0), ('late_answer', 7)]:
                answers(arm[field], visible(events, cut))
                counts['answer_records'] += 1
            assert [row['floor'] for row in arm['lifecycle']] == [0, 0, 4, 7, 7]
            assert [row['state']['history_rows'] for row in arm['lifecycle']] == [3200, 3200, 2600, 2000, 2000]
        counts['disconnected_cases'] += 1
    for cell in result['continuing']:
        events = load_record(cell['lineage_file'])
        assert cell['cycles'] == 8
        for arm in cell['arms'].values():
            records = arm['old_queries'] + arm['new_queries'] + [
                publication['newcut_query'] for publication in arm['publications']
            ] + [arm['final_query']]
            for record in records:
                answers(record['answers'], visible(events, record['cut']))
                counts['answer_records'] += 1
            assert len(arm['publications']) == 8
        counts['continuing_cases'] += 1
    pointer = load_record(HERE / 'spectrum_latest.json')
    assert sha(pointer['results']) == pointer['sha256']
    raw = load_record(pointer['results'])
    source_hashes(raw, [
        ('driver_sha256', HERE / 'spectrum_scaling.py'),
        ('broker_sha256', HERE / 'interval_broker.py'),
        ('compiler_sha256', HERE.parent / 'support_spectrum.py'),
        ('protocol_sha256', HERE / 'spectrum_protocol.md'),
    ])
    assert {(cell['commits'], cell['seed']) for cell in raw['cells']} == {
        (commits, seed) for commits in [32, 128, 512] for seed in [11, 29, 47]
    }
    for cell in raw['cells']:
        assert sha(cell['input_file']) == cell['input_sha256']
        counts['input_hashes'] += 1
        support = list(range(0, cell['commits'] + 1, 8))
        for arm in cell['arms'].values():
            assert arm['initial_support'] == [[cut, cut] for cut in support]
            assert [test['cut'] for test in arm['tests'] if test['accepted']] == support
            assert len(arm['tests']) == cell['commits'] + 1
            for test in arm['tests']:
                if test['accepted']:
                    assert test['query_id'] == (test['cut'] // 8) % 16
                    assert all(payload == f'{row}:{version}' for row, version, payload in test['answers'])
        counts['scaling_cells'] += 1
    assert counts['disconnected_cases'] == counts['continuing_cases'] == 3
    print(json.dumps({'status': 'passed', 'counts': counts,
        'scope': 'Retained support and answer records, fixed input hashes, and unchanged measurement-source identities. No physical native replay or fresh performance measurement.'}, indent=2))


if __name__ == '__main__':
    main()
