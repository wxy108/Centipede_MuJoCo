"""Record the completed bounded study and all local evidence hashes."""
from pathlib import Path
import study as s

if __name__=='__main__':
    frozen=s.verify();audit=s.read(s.DATA/'final_audit.json')
    assert audit['primary_audit']['passed'] and audit['primary_audit']['full_trajectories_recomputed']==236
    assert audit['persistent_audit']['passed'] and audit['persistent_audit']['full_trajectories_recomputed']==27
    stages={k:s.read(s.DATA/f'{k}_complete.json') for k in ['main_accepted9','independent_confirmation','resolution_duration_check','persistent_diagnostic']}
    assert sum(r['count'] for r in stages.values())==263
    assert all(r['completed']==r['count'] for r in stages.values())
    videos=[s.read(p) for p in (s.OUT/'videos').glob('*.json')]
    assert len(videos)==2 and all(r['frames']==180 and r['recorded_qpos_replayed_exactly'] for r in videos)
    target=s.DATA/'FINAL_RECEIPT.json'
    files=[dict(path=str(p.relative_to(s.ROOT)),bytes=p.stat().st_size,sha256=s.sha(p)) for p in sorted(s.ROOT.rglob('*')) if p.is_file() and p!=target and p.suffix!='.tmp']
    s.write(target,dict(created=s.now(),bounded_study_complete=True,biological_robustness_certified=False,
        clean_adoption=False,model_only=True,frozen_inputs=frozen,stages=stages,
        visual_qa='Both standalone figures and the no-loss, high-loss boundary and mid-burst video frames were visually inspected. Labels and paired red masks are visible. Fixed first confirmation seed 100 is shown, including its low-wave win, not selected for favorable performance.',
        interpretation='Independent 6 s paired-seed displacement crossing observed for waves 1.5 and 7; small strong-loss effect with body-ground contact and collision-restoration artifacts. No general or long-duration optimal-wave switch established.',
        videos=videos,file_count=len(files),files=files))
    print({'receipt':str(target),'evidence_files':len(files),'completed_primary_and_diagnostic_conditions':263,'frozen_inputs':frozen})
