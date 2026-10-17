"""Preserved pose renderer with an explicit speed-first viability label."""
from pathlib import Path
SOURCE=Path(__file__).resolve().parent.parent/'mujoco_compliant_relaxed_amplitude_2026-09-02'/'render_compliant_comparison.py'
code=SOURCE.read_text(encoding='utf-8')
old="status='ABSOLUTE GATES PASSED' if rec['absolute_feasible'] else 'DIAGNOSTIC: GATES FAILED'"
new="status='SPEED-FIRST VIABILITY PASSED' if rec.get('speed_first_feasible',rec.get('absolute_feasible')) else 'DIAGNOSTIC: VIABILITY FAILED'"
assert old in code
exec(compile(code.replace(old,new),str(SOURCE),'exec'))
