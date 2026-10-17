"""Resume the predeclared screen after the wave 2.5 baseline exclusion."""
import study as s

WAVES=[1.5,2,3,3.5,4,4.5,5,6,7]

if __name__=='__main__':
    for w in WAVES:
        r=s.read(s.TRIALS/s.condition_id(s.task(w,0))/'result.json')
        assert r['completed'] and r['nominal_current_runtime_accepted']
    note=dict(created=s.now(),accepted_waves=WAVES,excluded_wave=2.5,
        reason='Wave 2.5 failed the previously declared 1% historical displacement reproduction bound; its current raw displacement is 178.4982589786465 mm vs 175.0807019459 mm historical. This is a calibration exclusion, not a simulation divergence.',
        action='Preserve the original 200-condition plan and all results. Resume the same loss levels and seeds for nine accepted waves, reusing exact completed conditions. No profile or bound changes.',
        source_sha256=s.sha(__file__))
    path=s.DATA/'accepted_screen_amendment.json'
    if not path.exists():s.write(path,note)
    tasks=[s.task(w,nu,seed,stage='main_accepted9') for nu in s.manifest()['nu'] for seed in ([0] if nu in (0,1) else range(6)) for w in WAVES]
    s.bank(tasks,'main_accepted9',12)
