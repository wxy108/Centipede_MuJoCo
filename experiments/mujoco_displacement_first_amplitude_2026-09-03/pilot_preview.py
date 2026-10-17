"""Render an early, explicitly provisional speed-first comparison for wave 6/7."""
from __future__ import annotations
import json
from pathlib import Path
import study as s


def previous_trace(wave):
    old=s.read(s.OLD/'outputs/data/validated_selection.json')['waves'][f'{wave:g}']
    chosen=old['selected'];row=chosen['nominal'] if chosen else old['diagnostic']
    stage='delivery' if chosen else 'delivery_diagnostic'
    path=s.OLD/'outputs/trials'/stage/row['id']/f"{row['id']}.json"
    assert path.exists() and path.with_suffix('.npz').exists()
    return path,bool(chosen)


def main():
    selection=s.read(s.DATA/'interim_validated_selection.json')
    renderer=s.e.load(s.ROOT/'render_compliant_comparison.py','speed_first_preview_renderer')
    videos=[]
    for wave in (6.,7.):
        selected=selection['waves'][f'{wave:g}']['selected']
        assert selected and selected['qualified']
        base=s.read(selected['nominal']['record'])['task']
        task=dict(base,stage='pilot_preview_delivery',trace=True)
        raw=s.e.run_one(task)
        for key,value in selected['nominal']['metrics'].items():
            assert s.np.array_equal(s.np.asarray(value),s.np.asarray(raw['metrics'][key]),equal_nan=True),key
        old,old_ok=previous_trace(wave)
        labels=['PREVIOUS VALIDATED' if old_ok else 'PREVIOUS DIAGNOSTIC / NOT QUALIFIED',
                'PROVISIONAL DISPLACEMENT-FIRST / LONG-STEP VALIDATED']
        tag=f'{wave:g}'.replace('.','p')
        for local in (False,True):
            name=f"pilot_w{tag}_{'local' if local else 'three_views'}"
            videos.append(renderer.make([old,raw['record_path']],s.OUT/'videos'/f'{name}.mp4',local=local,labels=labels))
    result=dict(provisional=True,waves=[6,7],videos=videos,
        note='Candidates passed 12 s and half-step validation; exhaustive search is not complete.')
    s.write(s.DATA/'pilot_preview.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
