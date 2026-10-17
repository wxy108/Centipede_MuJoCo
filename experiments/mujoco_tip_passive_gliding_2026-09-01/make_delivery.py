"""Render an explicitly named candidate; never promote or overwrite a baseline."""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def module(filename,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',required=True)
    args=p.parse_args()
    render=module('render_flexion_demo.py','delivery_render')
    fx=render.fx
    candidate=next(c for c in fx.stage_a_bank()+fx.stage_b_bank()+fx.stage_c_bank() if c['name']==args.candidate)
    out=fx.OUT/'delivery'/args.candidate
    out.mkdir(parents=True,exist_ok=True)
    paths={}
    for terrain,duration in (('flat',12.),('rough',6.)):
        records=[]
        for wave in fx.WAVES:
            tag=f"{args.candidate}_w{wave:g}_t{duration:g}_dt0.00025".replace('.','p')
            record=fx.TRIALS/terrain/args.candidate/f'{tag}.json'
            assert record.with_suffix('.npz').exists(),record
            records.append(record)
        path=out/f'{terrain}_three_views.mp4'
        render.make(records,path,end=6.)
        paths[terrain]=str(path)
        if terrain=='flat':
            path=out/'flat_tip_local_three_views.mp4'
            render.make(records,path,end=6.,local=True)
            paths['flat_local']=str(path)
    probe=module('probe_side_contact.py','delivery_probe')
    probe.K=candidate['stiffness_nm_rad']
    probe.DAMPING_SCALE=candidate['damping_scale']
    probe.C=candidate['damping_nms_rad']
    probe.OUT=out/'side_contact'
    tests=[probe.run(axis) for axis in ('old_radial','local_z')]
    probe.render(tests)
    paths['side_contact']=str(probe.OUT/'side_contact_axis_comparison_three_views.mp4')
    # Verify each actual media stream, not just the existence of a file.
    import imageio.v2 as imageio
    media={}
    for name,path in paths.items():
        reader=imageio.get_reader(path)
        media[name]=dict(path=path,frames=reader.count_frames(),shape=list(reader.get_data(0).shape),fps=reader.get_meta_data()['fps'])
        reader.close()
        assert media[name]['frames']>0
    payload=dict(candidate=candidate,media=media,side_contact=tests,
        flat_video_source='first 6 s of unsmoothed 12 s validation trace',
        rough_video_status='diagnostic; check gate ledger, not a success demonstration',
        erasure='OFF for all locomotion demos',default_model_replaced=False)
    (out/'delivery_manifest.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2),flush=True)


if __name__=='__main__':main()
