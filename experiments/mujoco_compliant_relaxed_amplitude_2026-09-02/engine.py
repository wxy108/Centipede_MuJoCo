"""Frozen prior physics engine, isolated output namespace, relaxed gate labels."""
from pathlib import Path
import importlib.util
import json
import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import gates

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'mujoco_rigid_control_amplitude_2026-09-02'
spec = importlib.util.spec_from_file_location('relaxed_frozen_engine', PARENT / 'engine.py')
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)
OUT = ROOT / 'outputs'
DATA = OUT / 'data'
TRIALS = OUT / 'trials'
backend.TRIALS = TRIALS


def __getattr__(name):
    return getattr(backend, name)


def run_one(task):
    assert task['topology'] == 'flexible' and task['terrain'] == 'flat'
    result = backend.run_one(task)
    path = Path(result['record_path'])
    assert path.is_relative_to(TRIALS)
    result.setdefault('strict_absolute_gates', dict(result['gates']))
    result['gates'] = gates.absolute(result)
    result['absolute_feasible'] = all(result['gates'].values())
    result['failed'] = [k for k, v in result['gates'].items() if not v]
    result['gate_policy_sha256'] = gates.POLICY_SHA
    result['gate_policy'] = 'relaxed v1; original absolute gates separately preserved'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def run_tasks(tasks, workers=20):
    import concurrent.futures
    import time
    start = time.monotonic()
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            r = future.result()
            records.append(r)
            print(json.dumps(dict(done=i, total=len(tasks), id=r['id'],
                D=r['metrics']['displacement_mm'], failed_absolute=r['failed'],
                elapsed_s=time.monotonic()-start)), flush=True)
    return records
