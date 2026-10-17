"""Strict-versus-relaxed amplitudes, unsmoothed demos, and final provenance audit."""
from __future__ import annotations
import argparse
import datetime
import inspect
import json
from pathlib import Path
import runpy
import sys
import xml.etree.ElementTree as ET
import study as s
e, np = s.e, s.np


def freeze_reporting_sources():
    path = e.DATA / 'reporting_and_asset_sources.json'
    if path.exists():
        return s.read(path)
    xml = ET.parse(s.manifest()['plant']).getroot()
    meshdir = Path(xml.find('compiler').get('meshdir', ''))
    if not meshdir.is_absolute():
        meshdir = Path(s.manifest()['plant']).parent / meshdir
    assets = []
    for mesh in xml.findall('./asset/mesh'):
        filename = mesh.get('file')
        if filename:
            asset = Path(filename)
            if not asset.is_absolute():
                asset = meshdir / asset
            assets.append(dict(path=str(asset.resolve()), sha256=s.sha(asset)))
    sources = [e.PARENT / 'joint_qa.py', e.PARENT / 'render.py',
               e.PREVIOUS / 'render_flexion_demo.py']
    result = dict(captured_at=datetime.datetime.now().astimezone().isoformat(),
        timing='Additional asset/reporting snapshot after initial search launch; not claimed as a pre-launch asset snapshot',
        cold_physics_replay_check=str(e.DATA / 'verification.json'), assets=assets,
        reporting=[dict(path=str(p), sha256=s.sha(p)) for p in sources])
    s.write(path, result)
    return result


def previous_trace(wave, strict, manifest):
    previous = strict['waves'][f'{wave:g}']['selected']
    if previous:
        tid = previous['nominal']['id']
        kind = 'previous strict optimum'
        folder = 'delivery'
    else:
        task = next(t for t in manifest['references'] if t['wave'] == wave)
        tid = e.task_id(task)
        kind = 'original fixed amplitude; previous strict search found no validated candidate'
        folder = 'delivery_reference'
    path = e.PARENT / 'outputs/trials' / folder / tid / f'{tid}.json'
    assert path.exists() and path.with_suffix('.npz').exists()
    return path, kind, bool(previous)


def strict_checks(choice):
    wave = choice['nominal']['wave']
    if wave in (2.5, 3.):
        assert e.fx.pd_config(wave) == s.manifest()['PD_profiles_frozen'][f'{wave:g}']['config'], 'Historical-axis diagnostic PD mapping changed'
    old = e.load(e.PARENT / 'search.py', 'relaxed_report_old_policy')
    old.load_reference = s.load_reference
    results = []
    for label, row in zip(('nominal_6s', 'long_12s', 'half_step_6s'),
                          [choice['nominal'], *choice['checks']]):
        raw = s.read(row['record'])
        raw = dict(raw, gates=raw.get('strict_absolute_gates', raw['gates']))
        assessment = old.assess(raw)
        results.append(dict(condition=label, feasible=assessment['feasible'],
                            failed=assessment['failed']))
    return results


def make_plots(table, motion):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = e.OUT / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    waves = [r['wave'] for r in table]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    for ax, key, title in zip(axes, s.KEYS, ('Body yaw', 'Leg yaw', 'Leg pitch')):
        old = [r['old_amplitudes'][key] if r['old_strict_available'] else np.nan for r in table]
        new = [r['amplitudes'][key] if r['amplitudes'] else np.nan for r in table]
        ax.plot(waves, [r['original_fixed_amplitudes'][key] for r in table], ':',
                color='#648ca7', alpha=.7, label='Original fixed setting (may fail gates)')
        ax.plot(waves, old, 's--', color='.55', label='Previous strict optimum')
        ax.plot(waves, new, 'o-', color='#007e96', label='New relaxed optimum')
        ax.set(title=title, xlabel='Coupled wave number', ylabel='Command amplitude (deg)')
        ax.set_xticks(waves)
        ax.tick_params(axis='x', labelrotation=60)
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .89),
               ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Amplitude-only optimization | same compliant tip and frozen PD per wave\nNo amplitude-distance reward; missing qualified results remain blank')
    fig.tight_layout(rect=(0, 0, 1, .82))
    fig.savefig(out / 'amplitudes_strict_vs_relaxed.png', dpi=170)
    plt.close(fig)

    matrix = np.array([[r['amplitude_delta_deg'][key] if r['amplitude_delta_deg'] else np.nan
                        for key in s.KEYS] for r in table])
    vmax = max(5., float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(3), ['Body yaw', 'Leg yaw', 'Leg pitch'])
    ax.set_yticks(range(len(table)), [f"{r['wave']:g}" + (' *' if not r['old_strict_available'] else '') for r in table])
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if np.isfinite(value):
                ax.text(j, i, f'{value:+.2f}°', ha='center', va='center',
                        color='white' if abs(value) > .55*vmax else 'black')
    ax.set(ylabel='Coupled wave number', title='New minus previous amplitude\n* No prior validated candidate: comparison uses original fixed amplitude')
    fig.colorbar(im, ax=ax, label='Amplitude difference (deg)')
    fig.tight_layout()
    fig.savefig(out / 'amplitude_differences.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(waves, [r['original_fixed_D_mm'] for r in table], ':', color='#648ca7',
            alpha=.8, label='Original fixed setting (may fail gates)')
    failed_fixed = [r for r in table if not r['original_fixed_nominal_feasible']]
    ax.scatter([r['wave'] for r in failed_fixed], [r['original_fixed_D_mm'] for r in failed_fixed],
               marker='x', color='#648ca7', s=40, label='Fixed setting fails 6 s gates')
    ax.plot(waves, [r['old_D_mm'] if r['old_strict_available'] else np.nan for r in table],
            's--', color='.55', label='Previous strict optimum')
    ax.plot(waves, [r['D_mm'] if r['D_mm'] is not None else np.nan for r in table],
            'o-', color='#007e96', label='New relaxed optimum')
    ax.set(xlabel='Coupled wave number', ylabel='Forward displacement (mm)',
           title='Nominal displacement, t=1.5–5.999 s\nOptima validated under their respective gates; fixed reference may fail')
    ax.set_xticks(waves)
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'displacement_strict_vs_relaxed.png', dpi=170)
    plt.close(fig)

    metrics = [('coherent_roll_p95_deg', 'Whole-body roll P95 (deg)'),
               ('front_roll_p95_deg', 'Front-body roll P95 (deg)'),
               ('center_height_ptp_mm', 'Mean body height range (mm)'),
               ('ground_force_peak_bw', 'Peak ground force (BW)'),
               ('loaded_foot_slip_p95_mm_s', 'Loaded foot slip P95 (mm/s)'),
               ('body_roll_speed_peak', 'Peak body roll joint speed (rad/s)')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.plot(waves, [r['old_metrics'][key] if r['old_strict_available'] else np.nan for r in table],
                's--', color='.55', ms=3, label='Previous strict optimum')
        ax.plot(waves, [r['new_metrics'][key] if r['new_metrics'] else np.nan for r in table],
                'o-', color='#007e96', ms=4, label='New relaxed optimum')
        ax.set(title=title, xlabel='Coupled wave number')
        ax.set_xticks(waves)
        ax.tick_params(axis='x', labelrotation=60)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Stability/contact trade-offs, not a claim that every metric improves\nIndividual-body roll envelopes are separately included in the report')
    fig.tight_layout()
    fig.savefig(out / 'stability_strict_vs_relaxed.png', dpi=170)
    plt.close(fig)


def audit(selection, table, videos):
    manifest = s.manifest()
    setup = s.audit_setup()
    reporting_check = s.read(e.DATA / 'reporting_policy_verification.json')
    assert reporting_check['passed'] and reporting_check['policy_sha256'] == s.gates.POLICY_SHA
    presentation = s.read(e.DATA / 'presentation_corrections.json')
    assert presentation['state_source_identical'] and presentation['make_source_identical_except_legend']
    assert s.sha(presentation['new_renderer']) == presentation['new_renderer_sha256']
    assert s.sha(presentation['preserved_parent_renderer']) == presentation['preserved_parent_renderer_sha256']
    assert setup['passed'] and selection['validation_complete']
    assert selection['policy_sha256'] == s.gates.POLICY_SHA
    sources = freeze_reporting_sources()
    for source in [*sources['assets'], *sources['reporting']]:
        assert s.sha(source['path']) == source['sha256'], source['path']
    rows = s.collect()
    tasks = manifest['tasks']
    tasks = tasks + s.read(e.DATA / 'coarse_manifest.json')['tasks'] + s.read(e.DATA / 'polish_manifest.json')['tasks']
    assert len(tasks) == len(rows) == len({e.task_id(t) for t in tasks})
    assert {r['id'] for r in rows} == {e.task_id(t) for t in tasks}
    for task in tasks:
        ref = next(t for t in manifest['references'] if t['wave'] == task['wave'])
        for key in ('config', 'topology', 'terrain', 'duration', 'dt_scale'):
            assert task[key] == ref[key], (key, task)
        assert set(task['amplitudes']) == set(s.KEYS)
        assert all(lo <= task['amplitudes'][key] <= hi for key, lo, hi in zip(s.KEYS, s.LOW, s.HIGH))
        if task.get('source_record'):
            assert s.sha(task['source_record']) == task['source_sha256']
    for row in rows:
        assert s.sha(row['record']) == row['raw_record_sha256']
    checks = []
    for wave, result in selection['waves'].items():
        selected = result['selected']
        tested = {r['nominal']['id']: r for r in result['candidates']}
        eligible = [r for r in rows if r['wave'] == float(wave) and r['feasible']]
        if selected:
            assert selected['qualified'] and all(r['feasible'] for r in selected['checks'])
            for candidate in eligible:
                if candidate['D'] > selected['nominal']['D']:
                    assert candidate['id'] in tested and not tested[candidate['id']]['qualified']
        else:
            assert all(r['id'] in tested for r in eligible)
            assert not any(r['qualified'] for r in tested.values())
        comparison = next(r for r in table if r['wave'] == float(wave))
        assert s.sha(comparison['original_fixed_record']) == comparison['original_fixed_record_sha256']
        if comparison['old_strict_available']:
            assert selected and comparison['D_mm'] >= comparison['old_D_mm'] - 1e-9
        checks.append(dict(wave=float(wave), status=result['status'],
                           nominal_feasible=len(eligible), validated=len(tested)))
    for video in videos:
        assert Path(video['video']).exists()
        assert video['frames'] == 180 and video['shape'] == [880, 1920, 3]
        assert video['unsmoothed'] and not video['physics_changes']
        assert all(max(p['body_error_m'], p['foot_error_m']) < 1e-7 for p in video['pose_checks'])
    for path in e.DATA.glob('*_errors.json'):
        assert s.read(path)['status'] == 'resolved_by_complete_rerun'
    result = dict(passed=True, distinct_amplitude_conditions=len(rows), wave_checks=checks,
        videos_verified=len(videos), only_three_amplitudes_varied=True,
        physical_model_and_PD_unchanged=True, original_strict_results_unchanged=True,
        every_faster_nominal_candidate_checked=True, original_strict_feasible_set_retained=True,
        no_amplitude_difference_reward_or_gate=True, global_optimum_proved=False,
        gate_policy_sha256=s.gates.POLICY_SHA)
    s.write(e.DATA / 'final_delivery_audit.json', result)
    return result


def report(table, motion, videos, audit_result):
    def link(label, path):
        return f'[{label}]({Path(path).as_posix()})'
    def amps(value):
        return ' / '.join(f'{value[key]:.2f}' for key in s.KEYS) if value else '—'
    qualified = [r for r in table if r['amplitudes']]
    noticeable = [r for r in qualified if r['noticeable_amplitude_difference']]
    lines = ['# Compliant tip：放宽门槛后的幅值优化', '',
        f"检查 {audit_result['distinct_amplitude_conditions']:,} 个不同幅值组合；{len(qualified)}/{len(table)} 个波数通过六秒、12 秒和半时间步复核。",
        f"{len(noticeable)} 个合格波数出现明确幅值变化（任一幅值变化 ≥5°，或相对变化 ≥25%，分母至少 5°）。这仅是报告指标，没有用它强迫优化结果变化。", '',
        '仍为同一个 local-Z compliant-tip Model；仅优化 body yaw、leg yaw、leg pitch 三个幅值。body/leg 波数相同、频率 1 Hz、每波数 PD 固定、相位与模型不变、contact erasure 关闭。不同波数的历史 PD 并非完全相同。Robot Step 18/19 和上一轮严格实验未修改。', '',
        '参数文件中的 PD 配置值是相对参考模型的倍率，不是 SI 单位的绝对刚度/阻尼。roll 倍率作用于被动 body-roll 弹簧/阻尼，其余倍率作用于主动 PD。历史 PD 来源的原始状态标签予以保留；本轮验证的是相应 PD 与新幅值的完整组合，不把历史标签误作本轮验证结论。', '',
        '## 新旧幅值直接对比', '',
        '三个幅值顺序为 body yaw / leg yaw / leg pitch，单位为度。位移 D 是 21 个身体节点几何中心沿固定前向轴的位移（非质量加权质心），统计 t=1.5–5.999 s。带 * 的波数旧严格搜索未找到合格解，旧值仅为原固定幅值，不是旧合格最优。', '',
        '图中的蓝色点线仅作原始固定幅值参照，不保证通过当前门槛，也不把这条参考曲线整体标为合格最优。表中原固定 D 带 †、位移图中的蓝色叉号，均表示该固定设置在当前六秒门槛下不合格；未标叉也不等于已通过本轮全部长时复核。', '',
        '| Wave | 旧幅值（°） | 新幅值（°） | D 旧→新（mm） | 原固定 D（mm） | 最大幅值变化（°） | 状态 |',
        '|---:|---|---|---|---:|---:|---|']
    for row in table:
        wave = f"{row['wave']:g}" + (' *' if not row['old_strict_available'] else '')
        d = f"{row['old_D_mm']:.2f} → {row['D_mm']:.2f}" if row['D_mm'] is not None else f"{row['old_D_mm']:.2f} → —"
        change = f"{max(abs(v) for v in row['amplitude_delta_deg'].values()):.2f}" if row['amplitude_delta_deg'] else '—'
        status = '合格，幅值明显变化' if row['noticeable_amplitude_difference'] else '合格' if row['amplitudes'] else '未解决'
        fixed = f"{row['original_fixed_D_mm']:.2f}" + (' †' if not row['original_fixed_nominal_feasible'] else '')
        lines.append(f"| {wave} | {amps(row['old_amplitudes'])} | {amps(row['amplitudes'])} | {d} | {fixed} | {change} | {status} |")
    lines += ['', link('三幅值新旧对比图', e.OUT / 'figures/amplitudes_strict_vs_relaxed.png') + '；' +
        link('幅值变化热图', e.OUT / 'figures/amplitude_differences.png') + '；' +
        link('位移对比', e.OUT / 'figures/displacement_strict_vs_relaxed.png'), '',
        '## 放宽了什么、保留了什么', '',
        '抬腿高度 P95：5.5→7 mm；腿 yaw/pitch 速度 P99：8/7→10/9 rad/s；身体/腿 yaw/腿 pitch 波形残差：.40/.18/.40→.60/.25/.55。滑移和左右支撑力不平衡允许达到同模型参考的 1.35 倍，单位位移功耗 1.50 倍，10 ms 冲量 1.25 倍。旧 tip 轴参考与额外的参考相关姿态约束只作诊断。接触占空比等次级步态限制也适度放宽，完整阈值见 '+link('冻结的实验协议', e.ROOT / 'PROTOCOL.md')+'。', '',
        '防翻、整体/前段/root roll、body 关节速度和跳变、严重撞地、穿透、力矩、tip 限位和跟踪误差等绝对底线保持不变。body yaw 最大搜索幅值仍为 36°，不跨模型关节限位。放宽门槛不代表改了刚度、阻尼、质量或惯性。', '',
        '旧完整数据重新判定；不再成立的旧早停结论重新计算。加入每波数 64 个新 Sobol 样本，再做两轮多起点细化。合格候选按六秒位移从高到低复核，全部更快名义合格候选均已检查。所得是已测试集合内、放宽后的门槛下最高合格位移，不是连续空间的数学全局最优。', '',
        '## 稳定性与代价', '',
        '| Wave | 整体/前段 roll P95（°） | 最差身体节点 roll 峰值（°） | 高度波动（mm） | 峰值地面力（BW） | tip P95/峰值（°） | 半步长 D 差异 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in qualified:
        m = row['new_metrics']
        j = motion[row['wave']]
        lines.append(f"| {row['wave']:g} | {m['coherent_roll_p95_deg']:.2f}/{m['front_roll_p95_deg']:.2f} | {max(j['world_roll_abs_peak_deg_per_body']):.2f} | {m['center_height_ptp_mm']:.3f} | {m['ground_force_peak_bw']:.3f} | {m['tip_angle_p95_deg']:.2f}/{m['tip_angle_peak_deg']:.2f} | {100*row['halfstep_D_relative_difference']:.2f}% |")
    lines += ['', '稳定不等于 roll 为零。这里同时报告逐节 world-roll 极值，避免平均值掩盖局部偏转；实际关节半幅也单独保存。单步角加速度仍作为诊断保留，合格并不意味着所有瞬时加速度已得到生物学校准。', '',
        link('稳定性与接触新旧对比图', e.OUT / 'figures/stability_strict_vs_relaxed.png'), '',
        '接近门槛的情况需要单独注意：', '']
    for row in qualified:
        m = row['new_metrics']
        if row['halfstep_D_relative_difference'] >= .08 or m['front_roll_p95_deg'] >= 10.8 or m['coherent_roll_p95_deg'] >= 9.:
            lines.append(f"- wave={row['wave']:g}：整体/前段 roll P95 为 {m['coherent_roll_p95_deg']:.2f}°/{m['front_roll_p95_deg']:.2f}°（上限 10°/12°）；半步长位移差异 {100*row['halfstep_D_relative_difference']:.2f}%（上限 10%）。通过不等于余量充足，也不等于消除了低波数 wobble。")
    lines += ['', '具体收益与代价（相对旧严格最优，不是原固定幅值）：', '']
    for row in qualified:
        if row['wave'] in (1.5, 2., 3.5) and row['old_strict_available']:
            old, new = row['old_metrics'], row['new_metrics']
            gain = 100*(row['D_mm']/row['old_D_mm']-1)
            lines.append(f"- wave={row['wave']:g}：位移 +{gain:.2f}%；载荷足滑移 P95 {old['loaded_foot_slip_p95_mm_s']:.2f}→{new['loaded_foot_slip_p95_mm_s']:.2f} mm/s；整体 roll P95 {old['coherent_roll_p95_deg']:.2f}°→{new['coherent_roll_p95_deg']:.2f}°。")
    lines += ['', '新候选在旧严格规则下仍未通过的项目（六秒、12 秒、半步长分别检查）：', '']
    for row in qualified:
        failed = [f"{check['condition']}: {', '.join(check['failed'])}" for check in row['strict_policy_checks'] if not check['feasible']]
        lines.append(f"- wave={row['wave']:g}: " + ('；'.join(failed) if failed else '旧严格规则也通过；本轮搜索找到了更好的候选或保留原解。'))
    unresolved = [row for row in table if not row['amplitudes']]
    if unresolved:
        lines += ['', '## 尚未解决的波数', '',
            '以下仅列出用于 Demo 的完整诊断轨迹及其失败项，不是合格优化结果，也不代表该波数的所有候选都因同一原因失败。有限搜索未找到合格解，不等于证明不存在合格解。', '',
            '| Wave | 诊断位移（mm，不计入合格成绩） | 该轨迹未通过的六秒检查 |',
            '|---:|---:|---|']
        for row in unresolved:
            diagnostic = row['diagnostic']
            lines.append(f"| {row['wave']:g} | {diagnostic['D']:.2f} | {', '.join(diagnostic['failed'])} |")
    lines += ['', '## Demo', '',
        '每个全景视频均有前视、侧视、斜视，上排为旧严格最优（旧搜索未找到合格解时显示原固定幅值），下排为新结果。未解决波数明确标为 NOT QUALIFIED。使用原始运动记录，不做动作平滑。', '']
    for video in videos:
        role = '局部放大' if video.get('local') else '三视图'
        status = '合格' if video['selection_status'] == 'qualified' else '仅供诊断'
        lines.append(f"- wave={video['wave']:g}，{status}：" + link(role, video['video']))
    lines += ['', '## 全精度结果与证据', '',
        '- ' + link('合格参数及固定 PD', e.DATA / 'selected_amplitude_profiles.json'),
        '- ' + link('新旧幅值、位移与严格规则复查', e.DATA / 'comparison_table.json'),
        '- ' + link('实际关节运动及逐节 roll', e.DATA / 'achieved_joint_motion.json'),
        '- ' + link('完整验证结果', e.DATA / 'validated_selection.json'),
        '- ' + link('最终审计', e.DATA / 'final_delivery_audit.json'), '',
        '- ' + link('画面抽查范围、局部偏转与边界提醒', e.ROOT / 'VISUAL_QA.md'), '',
        '本轮结论仅适用于所测试的平地工况、固定控制器和模型；不等于生物学参数标定，也不保证 rough terrain、扰动或 random-contact 条件下成立。', '']
    (e.ROOT / 'RESULTS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    selected = s.read(e.DATA / 'validated_selection.json')
    assert selected['validation_complete']
    reporting_verifier = e.load(e.ROOT / 'verify_reporting.py', 'relaxed_reporting_verifier')
    reporting_verifier.verify(strict_checks)
    strict = s.read(e.PARENT / 'outputs/data/validated_selection.json')
    manifest = s.manifest()
    render = e.load(e.PARENT / 'render.py', 'relaxed_three_view_renderer')
    local_render = e.load(e.ROOT / 'render_compliant_comparison.py', 'relaxed_local_legend_renderer')
    assert inspect.getsource(render.State) == inspect.getsource(local_render.State)
    normalized_make = inspect.getsource(local_render.make).replace(
        "info.get('local_legend_version')==3", "info.get('local_legend_version')==2").replace(
        'orange: compliant tip; cyan dashed: neutral; BOTH ROWS COMPLIANT',
        'orange: flexible; cyan dashed: neutral; cyan solid: rigid').replace(
        'local_legend_version=3 if local else None', 'local_legend_version=2 if local else None')
    assert inspect.getsource(render.make).strip() == normalized_make.strip()
    s.write(e.DATA / 'presentation_corrections.json', dict(
        captured_at=datetime.datetime.now().astimezone().isoformat(),
        timing='Post-validation presentation correction; not a pre-search source snapshot',
        change='Only two local-demo legends clarified: both rows are compliant. No trajectory, camera, panel or physics change.',
        state_source_identical=True, make_source_identical_except_legend=True,
        new_renderer=str(e.ROOT / 'render_compliant_comparison.py'),
        new_renderer_sha256=s.sha(e.ROOT / 'render_compliant_comparison.py'),
        preserved_parent_renderer=str(e.PARENT / 'render.py'),
        preserved_parent_renderer_sha256=s.sha(e.PARENT / 'render.py')))
    table, videos, paths_by_wave = [], [], {}
    for wave in e.WAVES:
        result = selected['waves'][f'{wave:g}']
        choice = result['selected']
        row = choice['nominal'] if choice else result['diagnostic']
        before, kind, old_qualified = previous_trace(wave, strict, manifest)
        old = s.read(before)
        fixed_task = next(t for t in manifest['references'] if t['wave'] == wave)
        fixed_raw, fixed_path = s.load_reference(fixed_task)
        fixed_assessment = s.assess(fixed_raw)
        after = e.TRIALS / ('delivery' if choice else 'delivery_diagnostic') / row['id'] / f"{row['id']}.json"
        replay = s.read(after)
        original = s.read(row['record'])
        for path in (before, after):
            with np.load(path.with_suffix('.npz')) as trace:
                assert float(trace['t'][-1]) >= 5.998, (str(path), 'incomplete six-second replay')
                assert np.isfinite(trace['xpos']).all(), (str(path), 'nonfinite replay')
        for key, value in original['metrics'].items():
            assert np.array_equal(np.asarray(value), np.asarray(replay['metrics'][key]), equal_nan=True), key
        assert old['config'] == replay['config']
        assert fixed_raw['config'] == replay['config']
        assert fixed_raw['model_sha256'] == manifest['plant_sha256']
        for key in ('wave', 'duration', 'dt_scale', 'terrain', 'topology'):
            assert fixed_raw['task'][key] == replay['task'][key], (wave, key)
        assert old['model_sha256'] == replay['model_sha256'] == manifest['plant_sha256']
        old_a = old['task']['amplitudes']
        new_a = row['amplitudes'] if choice else None
        delta = {key: new_a[key] - old_a[key] for key in s.KEYS} if choice else None
        noticeable = bool(delta and any(abs(delta[key]) >= 5. or abs(delta[key]) / max(old_a[key], 5.) >= .25 for key in s.KEYS))
        comparison = dict(wave=wave, status=result['status'], old_reference_kind=kind,
            original_fixed_amplitudes=fixed_raw['task']['amplitudes'],
            original_fixed_D_mm=fixed_raw['metrics']['displacement_mm'],
            original_fixed_nominal_feasible=fixed_assessment['feasible'],
            original_fixed_nominal_failed=fixed_assessment['failed'],
            original_fixed_record=str(fixed_path), original_fixed_record_sha256=s.sha(fixed_path),
            amplitude_delta_vs_original_fixed_deg={key: new_a[key]-fixed_raw['task']['amplitudes'][key] for key in s.KEYS} if choice else None,
            old_strict_available=old_qualified, old_amplitudes=old_a,
            amplitudes=new_a, amplitude_delta_deg=delta,
            noticeable_amplitude_difference=noticeable,
            old_D_mm=old['metrics']['displacement_mm'], D_mm=row['D'] if choice else None,
            old_metrics=old['metrics'], new_metrics=row['metrics'] if choice else None,
            halfstep_D_relative_difference=choice['halfstep_D_relative_difference'] if choice else None,
            strict_policy_checks=strict_checks(choice) if choice else [],
            before_record=str(before), after_record=str(after),
            diagnostic=row if not choice else None)
        table.append(comparison)
        prefix = 'relaxed_optimized' if choice else 'relaxed_unresolved'
        tag = f'{wave:g}'.replace('.', 'p')
        labels = ['PRIOR STRICT OPTIMUM' if old_qualified else 'PRIOR FIXED / STRICT UNRESOLVED',
                  'RELAXED / VALIDATED' if choice else 'RELAXED DIAGNOSTIC / NOT QUALIFIED']
        video = render.make([str(before), str(after)], e.OUT / 'videos' / f'{prefix}_w{tag}_three_views.mp4', labels=labels)
        video.update(wave=wave, selection_status=result['status'], local=False)
        videos.append(video)
        paths_by_wave[wave] = ([str(before), str(after)], labels)
    for comparison in sorted((r for r in table if r['amplitudes']),
            key=lambda r: max(abs(v) for v in r['amplitude_delta_deg'].values()), reverse=True)[:2]:
        wave = comparison['wave']
        paths, labels = paths_by_wave[wave]
        tag = f'{wave:g}'.replace('.', 'p')
        video = local_render.make(paths, e.OUT / 'videos' / f'relaxed_optimized_w{tag}_local.mp4', local=True, labels=labels)
        video.update(wave=wave, selection_status='qualified', local=True)
        videos.append(video)
    s.write(e.DATA / 'comparison_table.json', table)
    s.write(e.DATA / 'video_manifest.json', videos)
    profiles = dict(model=manifest['plant'], model_sha256=manifest['plant_sha256'],
        gate_policy=manifest['gate_policy'], policy_sha256=s.gates.POLICY_SHA,
        frequency_hz=1., body_leg_wave_coupled=True, erasure=False,
        PD_parameter_semantics='Dimensionless multipliers, not SI stiffness/damping: roll scales passive body-roll stiffness/damping; other entries scale active kp/kv. Apply the frozen apply_impedance helper once to a fresh model/index.',
        validation_scope='The complete frozen-PD plus amplitude combination is validated. Historical source-PD status labels are preserved as provenance, not reused as the new combination status.',
        canonical_baseline_replaced=False, global_optimum_proved=False, selected_profiles={})
    for wave, result in selected['waves'].items():
        if result['selected']:
            choice = result['selected']
            profiles['selected_profiles'][wave] = dict(amplitudes_deg=choice['nominal']['amplitudes'],
                validation_status='combination passed nominal 6 s, long 12 s, and half-step 6 s checks',
                frozen_PD=manifest['PD_profiles_frozen'][wave], nominal_record=choice['nominal']['record'],
                validation_records=[r['record'] for r in choice['checks']],
                displacement_mm=choice['nominal']['D'])
    s.write(e.DATA / 'selected_amplitude_profiles.json', profiles)
    sources = freeze_reporting_sources()
    for source in sources['reporting']:
        assert s.sha(source['path']) == source['sha256']
    assert sys.modules['engine'] is e
    # Run the unchanged plotting script against this already imported isolated
    # engine module. A separate subprocess in the old folder would be unsafe.
    runpy.run_path(str(e.PARENT / 'joint_qa.py'), run_name='__main__')
    motion = {r['wave']: r for r in s.read(e.DATA / 'achieved_joint_motion.json')}
    make_plots(table, motion)
    result = audit(selected, table, videos)
    report(table, motion, videos, result)
    print(json.dumps(dict(audit=result, report=str(e.ROOT / 'RESULTS_ZH.md')), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-sources', action='store_true')
    args = parser.parse_args()
    if args.freeze_sources:
        r = freeze_reporting_sources()
        print(json.dumps(dict(assets=len(r['assets']), reporting=len(r['reporting']), captured=r['captured_at'])))
    else:
        main()
