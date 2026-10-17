"""Read-only recovery of comparable historical amplitude evidence."""
import collections,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'mujoco_displacement_first_amplitude_2026-09-03'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def load(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
policy=load(OLD/'policy.py','recovery_old_policy')

if __name__=='__main__':
    mf=read(OLD/'outputs/data/amplitude_manifest.json')
    for f in mf['frozen_files']:assert sha(f['path'])==f['sha256']
    paths={r['record']:r['sha256'] for r in mf['source_records']}
    for p in (OLD/'outputs/assessments').glob('*.json'):
        a=read(p);paths[a['record']]=a['raw_record_sha256']
    rows={};incomplete=[];reject=[]
    for p,h in paths.items():
        assert sha(p)==h,p
        r=read(p);t=r['task'];w=f"{t['wave']:g}"
        if r['model_sha256']!=mf['plant_sha256'] or t['config']!=mf['PD_profiles_frozen'][w]['config']:reject.append(p);continue
        if t['duration']!=6 or t['dt_scale']!=1:continue
        if not r.get('metrics'):incomplete.append(dict(id=r['id'],task=t,record=p));continue
        a=policy.assess(r);rows[r['id']]=dict(id=r['id'],wave=t['wave'],amplitudes=t['amplitudes'],task=t,
            D=a['D'],feasible=a['feasible'],failed=a['failed'],metrics=r['metrics'],record=p,sha256=h)
    selected=read(OLD/'outputs/data/interim_validated_selection.json');summary={}
    for w in mf['waves']:
        group=[r for r in rows.values() if r['wave']==w];ok=sorted([r for r in group if r['feasible']],key=lambda r:r['D'],reverse=True)
        old=selected['waves'].get(f'{w:g}',{}).get('selected')
        near=sorted(group,key=lambda r:(len(r['failed']),-r['D']))[:4]
        summary[f'{w:g}']=dict(full=len(group),feasible=len(ok),top=[{k:r[k] for k in ['id','D','amplitudes','record']} for r in ok[:5]],
            previous_validated_D=old['nominal']['D'] if old and old['qualified'] else None,
            near_feasible=[{k:r[k] for k in ['id','D','amplitudes','failed']} for r in near],
            failed_gate_counts=dict(collections.Counter(k for r in group for k in r['failed'])))
    out=ROOT/'outputs/data';out.mkdir(parents=True,exist_ok=True)
    value=dict(plant=mf['plant'],plant_sha256=mf['plant_sha256'],profiles=mf['PD_profiles_frozen'],waves=mf['waves'],
        full_rows=list(rows.values()),incomplete=incomplete,excluded_incomparable=reject,summary=summary,
        historical_frozen_files_verified=len(mf['frozen_files']),old_manifest_sha256=sha(OLD/'outputs/data/amplitude_manifest.json'))
    (out/'history.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(dict(full=len(rows),incomplete=len(incomplete),incomparable=len(reject),waves=summary)))
