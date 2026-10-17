"""Use the audited comparison report with the local viability-labelled renderer."""
from pathlib import Path
SOURCE=Path(__file__).resolve().parent.parent/'mujoco_amplitude_convergence_2026-09-03'/'report.py'
code=SOURCE.read_text(encoding='utf-8')
old="renderer=e.load(s.OLD/'render_compliant_comparison.py','convergence_renderer')"
new="renderer=e.load(s.ROOT/'render_compliant_comparison.py','speed_first_renderer')"
assert old in code
code=code.replace(old,new).replace('Same model, phase law, gates, and per-wave PD; no monotonicity constraint',
    'Same model, phase law, speed-first viability, and per-wave PD; no monotonicity constraint')
code=code.replace('三幅值收敛研究：有限测试集结果，不是连续全局最优证明',
    '位移优先三幅值研究：有限测试集结果，不是连续全局最优证明')
exec(compile(code,str(SOURCE),'exec'))
