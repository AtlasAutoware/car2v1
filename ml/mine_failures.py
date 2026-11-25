#!/usr/bin/env python3
"""mine_failures: tasks that start just before the places where a policy failed (run D, 10/15).

Reads an eval's episodes.jsonl (sim_rollout eval on a task set this script re-derives with the same
--maps/--n/--seed), takes every episode that did not succeed, and for each of a few distances before
the failure point (the furthest point along the route the car reached) writes a task that starts
there and runs to the same goal, with a fresh instruction for the shorter route. The list is
written to --out, sized to exactly --count tasks (sampled with a seed if there are more, repeated
if fewer), for `sim_rollout.py collect --tasks-file`.
A start point is kept only if the clean expert (car_train conditions, no noise) succeeds from it:
many failure points are goals in spots so tight that nobody can stop there, or starts with the car
body already touching a wall, and demos there teach nothing.

    python3 ml/mine_failures.py --eval data/_pool/dartbc_s0 --maps levine,... --n 375 --seed 20000 \
        --count 600 --out data/_tasks/mined.json
"""
import argparse, json, os, random, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_rollout as SR                                     # noqa: E402
import car_model as CM                                       # noqa: E402
from goal_core import describe_route                          # noqa: E402


def _expert_ok(args):
    """Is the segment recoverable: the clean expert succeeds from this start under the collection
    conditions, and the expert with the collection's steering noise in at least 2 of 3 tries (a start
    the noisy expert crashes from most of the time would only yield crash demos)."""
    task, seed, dart = args
    clean = CM.resolve('car_train', ['dart=0']); clean['preset'] = 'car_train'
    if not SR.rollout_car(task, None, clean, beta=1.0, record=False, seed=seed)[0]['success']: return False
    noisy = CM.resolve('car_train', [f'dart={dart}']); noisy['preset'] = 'car_train'
    ok = sum(SR.rollout_car(task, None, noisy, beta=1.0, record=False, seed=seed + 1 + k)[0]['success'] for k in range(3))
    return ok >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True, help='eval dir with episodes.jsonl')
    ap.add_argument('--maps', required=True); ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True, help='task seed of that eval')
    ap.add_argument('--before', default='2,3.5,5', help='metres before the failure point to start at')
    ap.add_argument('--count', type=int, default=600); ap.add_argument('--out', required=True)
    ap.add_argument('--min-left', type=float, default=1.5, help='shortest route (m) worth a task')
    ap.add_argument('--workers', type=int, default=6, help='processes for the expert check')
    ap.add_argument('--no-check', action='store_true', help='skip the expert check')
    ap.add_argument('--dart', type=float, default=0.12, help='steering noise the demos will be collected with (for the check)')
    ap.add_argument('--max-progress', type=float, default=0.98,
                    help='mine only failures before this share of the route. A crash after reaching the goal '
                         'happens while the bridge is already commanding a stop (inside goal_tol), which no '
                         'demonstration changes: the expert approaches tight goals at full speed too')
    a = ap.parse_args()
    tasks = {t['task_id']: t for t in SR.get_tasks(a.maps.split(','), a.n, a.seed)}
    eps = [json.loads(l) for l in open(os.path.join(a.eval, 'episodes.jsonl'))]
    fails = [r for r in eps if not r['success'] and r['progress'] < a.max_progress]
    prng = random.Random(a.seed); out = []; seen = set()
    for r in fails:
        t = tasks.get(r['task_id'])
        if t is None: continue
        P = np.asarray(t['path']); cum = np.concatenate([[0.0], np.hypot(*np.diff(P, axis=0).T).cumsum()])
        s_fail = float(r['progress']) * float(cum[-1])
        for d in (float(x) for x in a.before.split(',')):
            s0 = max(0.0, s_fail - d); i0 = int(np.searchsorted(cum, s0))
            i0 = min(i0, len(P) - 4)
            if i0 < 0 or (t['task_id'], i0) in seen or cum[-1] - cum[i0] < a.min_left: continue
            seen.add((t['task_id'], i0)); path = [list(map(float, p)) for p in P[i0:]]
            out.append({'map': t['map'], 'start': path[0], 'goal': path[-1], 'path': path,
                        'instruction': prng.choice(SR.PARA)(describe_route([tuple(p) for p in path])),
                        'task_id': f"{t['map']}:mined:{len(out)}", 'from_task': t['task_id'],
                        'failed_at_m': round(s_fail, 2), 'start_before_m': d,
                        'why': 'collided' if r['collided'] else 'timeout'})
    after_goal = sum(1 for r in eps if not r['success'] and r['progress'] >= a.max_progress)
    print(f'{len(eps)} episodes, {len(fails)} failures before {a.max_progress:.0%} of the route '
          f'({sum(r["collided"] for r in fails)} crashes; {after_goal} more failed after reaching the goal, not mined), '
          f'{len(out)} start points', flush=True)
    if out and not a.no_check:
        with ProcessPoolExecutor(a.workers, initializer=SR._exit_with_parent) as ex:
            ok = list(ex.map(_expert_ok, [(t, a.seed * 7919 + 10 * i, a.dart) for i, t in enumerate(out)], chunksize=2))
        kept = [t for t, k in zip(out, ok) if k]
        print(f'the expert (clean, and with noise {a.dart} in 2 of 3 tries) succeeds from {len(kept)} of {len(out)} '
              f'start points; the rest are dropped', flush=True)
        out = kept
    if not out: sys.exit('no failures to mine')
    if len(out) > a.count: out = prng.sample(out, a.count)
    while len(out) < a.count: out = out + out[:a.count - len(out)]
    for i, t in enumerate(out): t['task_id'] = f"{t['map']}:mined:{i}"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    SR._write_lines(a.out, [json.dumps(out)])
    by_map = {}
    for t in out: by_map[t['map']] = by_map.get(t['map'], 0) + 1
    print(f'{a.out}: {len(out)} tasks {by_map}', flush=True)


if __name__ == '__main__':
    main()
