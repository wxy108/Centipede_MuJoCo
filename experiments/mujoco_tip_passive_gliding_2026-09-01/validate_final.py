"""Final candidate validation suite; exact fixed controller and amplitudes."""
from __future__ import annotations
import concurrent.futures
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('final_fx', ROOT / 'run_tip_flexion_search.py')
fx = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fx
spec.loader.exec_module(fx)
SELECTED = 'localz_k2em05_d3p2_r38'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default=SELECTED)
    args = parser.parse_args()
    candidate = next(c for c in fx.stage_a_bank()+fx.stage_b_bank()+fx.stage_c_bank() if c['name'] == args.candidate)
    tasks = [(candidate, wave, terrain, duration, dt, trace)
             for terrain, duration, dt, trace in (
                 ('flat', 12., 1., True),
                 ('flat', 6., .5, False),
                 ('rough', 6., 1., True))
             for wave in fx.WAVES]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fx.run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(dict(terrain=result['terrain'],wave=result['case']['wave'],
                duration=result['duration'],dt=result['dt'],D=result['metrics']['displacement_mm'],
                failed=[key for key,value in result['gates'].items() if not value])),flush=True)
    payload = dict(candidate=candidate,results=results,controller_and_amplitudes_frozen=True,
                   default_model_replaced=False,scope='Centipede Model only')
    (fx.DATA / f"final_validation_{candidate['name']}.json").write_text(json.dumps(payload,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
