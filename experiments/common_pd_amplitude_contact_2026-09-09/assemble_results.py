"""Assemble completed evidence only; pending experiments cannot produce a final report."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
link=lambda label,p:f'[{label}](<{Path(p).as_posix()}>)'

def channel_report(name,title):
    out=ROOT/'outputs'/name;data=out/'data';manifest=read(data/'manifest.json')
    assert read(data/'p0_verification.json')['passed']
    discovery=read(data/'discovery_analysis.json')
    incomplete_groups=sum(not r['complete'] for r in discovery['summary'])
    figroot=out if name.endswith('transient') else ROOT
    lines=[f'## {title}','',f"发现阶段有 {len(discovery['candidates'])} 个探索性交叉组合；全部波数的均值保留失败的完整轨迹，不按幸存者筛选。{len(discovery['summary'])} 个波数×概率组合中，{incomplete_groups} 个因样本不完整而没有估计均值。",'',
           link('全波数位移与保持率',figroot/'contact_discovery.png')]
    inputs=[data/n for n in ('manifest.json','p0_verification.json','discovery_analysis.json','completed/discovery.json')]
    choice=discovery['selected_for_confirmation']
    if not choice:
        lines+=['','在本次波数、概率网格和样本下，没有选出满足既定条件的交叉组合。这不证明所有连续概率或其他模型中都不存在交叉。']
        return lines,inputs,dict(channel=name,status='NO_ELIGIBLE_DISCOVERY_CROSSING',incomplete_discovery_groups=incomplete_groups)
    frozen=read(data/'crossing_choice.json');confirmation=read(data/'confirmation_analysis.json')
    inputs += [data/'crossing_choice.json',data/'confirmation_analysis.json',data/'confirmation_declaration.json']
    low,high=frozen['low_wave'],frozen['high_wave'];p0,p1=frozen['bracket']
    lines+=['',f"固定比较 wave {low:g} 与 {high:g}；探索性交叉区间 p∈[{p0:g}, {p1:g}]。两者冻结 body amplitude 分别为 {frozen['baseline_body_amplitudes_deg'][0]:.3f}°、{frozen['baseline_body_amplitudes_deg'][1]:.3f}°。",
            '', '使用20组新的配对随机掩码确认；p=0、p=1为确定性端点。下表区间为低波减高波的97.5%配对区间，不能仅凭折线相交判定成功。','',
            '| p | 低波 D mm | 高波 D mm | 低−高的区间 mm | 工程门槛通过率：低／高 |','|---:|---:|---:|---|---|']
    for point in confirmation['points']:
        if not point['complete']:
            lines.append(f"| {point['p']:g} | 不完整 | 不完整 | 未估计 | 原始失败保留 |");continue
        a,b=point['group'];ci=point['paired_difference_CI97p5']
        lines.append(f"| {point['p']:g} | {a['mean_D']:.3f} | {b['mean_D']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {a['engineering_fraction']:.0%}／{b['engineering_fraction']:.0%} |")
    lines+=['',link('独立配对确认图',figroot/'contact_confirmation.png')]
    lines+=['',link('振幅、探索与独立确认总图',out/'final_crossing.png')]
    for n in ('figure_receipt.json','mechanism_analysis.json'):
        receipt_path=data/n
        if receipt_path.exists():inputs.append(receipt_path)
    if (out/'crossing_mechanism.png').exists():
        lines+=['',link('同一确认样本的运动与支撑诊断',out/'crossing_mechanism.png')+'。此图只描述已记录的路径、方向效率与接触情况，不构成对单一机制的因果证明。']
    lines+=['','同一批确认样本的接触与姿态诊断：','','| p | 实际擦除比例：低／高 | 身体触地时间比例：低／高 | 保持直立比例：低／高 |','|---:|---|---|---|']
    for point in confirmation['points']:
        if not point['complete']:
            lines.append(f"| {point['p']:g} | 样本不完整 | 未估计 | 未估计 |");continue
        a,b=point['group']
        lines.append(f"| {point['p']:g} | {a['mean_erased_fraction']:.1%}／{b['mean_erased_fraction']:.1%} | {a['chassis_contact_fraction']:.1%}／{b['chassis_contact_fraction']:.1%} | {a['upright_fraction']:.0%}／{b['upright_fraction']:.0%} |")
    lines+=['','持续通道的实际擦除比例为腿对占比，短时通道为腿×时间占比。该表与确认位移使用相同样本，不是另一组探索样本。']
    outcome=dict(channel=name,low_wave=low,high_wave=high,bracket=[p0,p1],
                 body_amplitude_descends=frozen['baseline_body_amplitude_descends'],
                 nominal_sign_confirmed=confirmation['confirmed_at_nominal_step'])
    later_reversals=[]
    for probability in manifest['probabilities']:
        if probability<=p1:continue
        groups=[next(g for g in discovery['summary'] if g['wave']==wave and g['p']==probability) for wave in (low,high)]
        if all(g['complete'] and g['mean_D']>=5 for g in groups) and groups[0]['mean_D']>groups[1]['mean_D']:
            later_reversals.append(dict(p=probability,low_minus_high_mm=groups[0]['mean_D']-groups[1]['mean_D']))
    outcome['exploratory_later_reversals_with_both_means_above_5mm']=later_reversals
    if later_reversals:
        text='、'.join(f"p={v['p']:g}（低减高{v['low_minus_high_mm']:.3f}mm）" for v in later_reversals)
        lines+=['',f'完整探索曲线在更高概率{text}又出现低波领先；这尚未独立确认。冻结区间内的反超即使得到确认，也不能外推为高波在所有更高概率都占优。']
    if not confirmation['confirmed_at_nominal_step']:
        lines+=['','新的随机掩码没有确认区间两端的相反符号；该组合不能作为已确认的交叉点。']
        outcome['status']='NOT_CONFIRMED_ON_HELD_OUT_MASKS'
    else:
        robustness=read(data/'robustness_analysis.json');inputs += [data/'robustness_analysis.json',data/'step_duration_declaration.json']
        signs=robustness['crossing_sign_supported_all_conditions'];posture=robustness['clean_posture_all_conditions'];engineering=robustness['all_engineering_gates_all_conditions']
        outcome.update(robust_signs=signs,robust_posture=posture,robust_all_engineering_gates=engineering)
        if signs and engineering:
            outcome['status']='SIGN_SUPPORTED_AND_ALL_OBSERVED_ENGINEERING_GATES_PASS'
            lines+=['','交叉符号在名义步长、减半步长和12秒复验中均获得支持，且这些复验观察均通过工程门槛。']
        elif signs and posture:
            outcome['status']='SIGN_AND_POSTURE_SUPPORTED_WITH_ENGINEERING_FAILURES'
            lines+=['','交叉符号和姿态在复验中获得支持，但部分轨迹未通过全部工程门槛，不能称作全面合格的稳健交叉。']
        elif signs:
            outcome['status']='SIGN_SUPPORTED_WITH_POSTURE_FAILURES'
            lines+=['','交叉符号在复验中获得支持，但包含姿态或身体支撑失败；这属于性能失效参与形成的交叉，不能当作两者均稳定行走的证据。']
        else:
            outcome['status']='NOMINAL_SIGN_CONFIRMED_BUT_ROBUSTNESS_INCONSISTENT'
            lines+=['','20组新种子在名义条件下确认了交叉，但相同8组掩码的名义／减半步长／12秒比较未在每个条件下都获得区间两端相反符号的支持，不能报告为稳健交叉。区间跨零可能来自样本较少，也可能伴随实际排序变化；需要分别看点估计与区间，不能直接等同于数值失稳。']
        sensitivity=robustness['timestep_sensitivity']
        if sensitivity:
            worst=max(r['absolute_mean_difference_as_no_loss_fraction'] for r in sensitivity)
            lines+=['',f"擦除条件下，减半步长与名义步长的最大单元均值差为对应无损位移的 {worst:.1%}。符号一致与数值收敛是不同判断；详见逐种子差异和区间。"]
            outcome['largest_mean_timestep_difference_as_no_loss_fraction']=worst
        lines+=['',link('步长与时长复验图',figroot/'contact_robustness.png')]
    if name.endswith('transient'):
        inputs.append(data/'exposure_analysis.json')
        selected_exposure=data/'selected_exposure_receipt.json'
        display=read(selected_exposure);assert display['heldout_analysis_present'] and not display['changed_statistics']
        for p,digest in display['source_hashes'].items():assert sha(p)==digest
        inputs.append(selected_exposure)
        lines+=['','此通道每0.5秒刷新期望失联；逐腿离地至少50µm后才恢复。实际失联时间可能因步态而不同，相同期望p不等于相同实际暴露。',
                link('选中两波的实际失联时间与恢复延迟',out/'selected_transient_exposure.png')]
    return lines,inputs,outcome

def main():
    workflow=read(ROOT/'outputs/contact_workflow_status.json')
    assert workflow['stage']=='mujoco_amplitude_and_two_contact_channels_complete','Required experiments are still pending.'
    data=ROOT/'outputs/physical_geometry_v2/data';selection=read(data/'validated_selection.json')
    assert selection['complete'] and selection['selected']
    amplitudes=read(data/'validated_amplitudes.json');assert amplitudes['status']=='VALIDATED_FINITE_SEARCH'
    seal=read(ROOT/'outputs/data/precontact_analysis_seal.json')
    for p,digest in seal['source_hashes'].items():assert sha(p)==digest,('Analysis seal changed; disclose and resolve before final assembly.',p)
    inputs=[data/'validated_selection.json',data/'validated_amplitudes.json',ROOT/'outputs/contact_workflow_status.json',ROOT/'outputs/data/precontact_analysis_seal.json']
    amendment_note=[]
    if 'amendment_receipt' in seal:
        amendment_path=Path(seal['amendment_receipt'])
        assert sha(amendment_path)==seal['amendment_sha256']
        amendment=read(amendment_path)
        assert not any(amendment[k] for k in ('changed_physics','changed_pd','changed_selection','changed_random_schedule','changed_statistics'))
        inputs.append(amendment_path)
        parity_path=ROOT/'outputs/data/p0_metadata_repair_dynamic_parity.json'
        assert read(parity_path)['passed'];inputs.append(parity_path)
        amendment_note=['','p=0首次运行发现结果保存时重复传入擦除类型字段，14项失败输出和原封存源码均已归档。仅删除重复字段后重新封存并重跑p=0；物理、PD、参数选点、随机计划和统计规则未改。修复前后的14条已保存轨迹逐数组完全一致。',link('修复与原始记录',amendment_path)]
    for item in seal.get('additional_amendments',[]):
        path=Path(item['receipt']);assert sha(path)==item['sha256']
        value=read(path);assert value['all_discovery_statistics_and_candidate_order_identical']
        assert not any(value[k] for k in ('changed_physics','changed_pd','changed_selection','changed_random_schedule','changed_statistics'))
        inputs.append(path)
        amendment_note+=['','探索分析保存比较对象时又发现重复interpretation文字字段。原分析已先保存全部统计量和选中的组合；修复保存方法后，全部统计量、候选顺序与选点完全一致，252项仿真未重跑。此修复发生在任何新种子确认之前。',link('分析记录修复证据',path)]
    convergence=read(data/'local_convergence_audit.json')
    inputs += [data/'local_convergence_audit.json',data/'convergence_extension_handover.json']
    polish=read(data/'joint_polish_audit.json')
    inputs += [data/'joint_polish_audit.json',data/'joint_polish_handover.json',data/'joint_polish_initial.json']
    finish_decision=read(data/'joint_finish_decision.json')
    inputs += [data/'joint_finish_decision.json',data/'joint_finish_declaration.json']
    finish_audit=None
    if finish_decision['extension_needed']:
        finish_audit=read(data/'joint_finish_audit.json')
        inputs += [data/'joint_finish_audit.json',data/'joint_finish_handover.json',data/'joint_finish_initial.json']
    qualification_audit=read(data/'qualification_audit.json')
    assert qualification_audit['status']=='COMPLETED_QUALIFICATION_AUDIT'
    assert qualification_audit['input_hashes'][str(data/'validated_selection.json')].lower()==sha(data/'validated_selection.json')
    inputs.append(data/'qualification_audit.json')
    evidence_path=ROOT/'outputs/data/final_evidence_check.json'
    evidence=read(evidence_path)
    assert evidence['passed'] and evidence['final_stage_check']
    assert all(v['started'] and v['assessments']>0 for v in evidence['channels'])
    inputs += [evidence_path,ROOT/'outputs/inference_static_checks/receipt.json']
    schedule_path=ROOT/'outputs/data/contact_schedule_final_audit.json'
    schedule=read(schedule_path);assert schedule['passed'] and schedule['final']
    for p,digest in schedule['input_hashes'].items():assert sha(p)==digest
    assert sha(ROOT/'audit_contact_schedules.py')==schedule['source_sha256']
    inputs.append(schedule_path)
    hist=amplitudes['historical_rl_range'];ext=amplitudes['extended_range']
    lines=['# 同一套PD：振幅优化与随机地面接触擦除','',
           f"完成 {len(selection['selected'])} 个波数的无损资格验证；未解决波数：{selection['unresolved']}。所有选点使用同一套完整PD，分别通过6秒、12秒、减半步长复验；无损减半步长位移差不超过10%。这是已测有限候选集的最优合格值，不是连续全局最优证明。",'']
    lines += [f"局部搜索的 {convergence['starts']} 个起点中，{convergence['converged']} 个达到预设坐标探查停止条件；{len(convergence['budget_limited'])} 个到达轮数上限。停止条件为归一化步长0.005时改进不超过0.2mm，不等于任意方向或全局收敛。原8轮终点仍有起点改善，因此在接触实验前统一延长所有未收敛起点，未按振幅趋势决定延长对象。",'']
    lines += [f"随后联动探查改善了 {polish['starts']} 个波数的候选，统一再做14方向、归一化步长0.005的小步细化；{polish['converged']} 个达到该停止条件，{len(polish['budget_limited'])} 个到达预算上限。两次细化均只依据位移改善与原工程门槛，未看过contact结果。",'']
    if finish_audit:
        lines += [f"上述预算上限触发了预先声明的条件续算：仅对仍有超过0.2mm改进的 {finish_audit['starts']} 个起点，保留14方向细化并沿上次成功方向增加3个较大探查距离，最多再做12轮。其中 {finish_audit['converged']} 个达到停止条件，{len(finish_audit['budget_limited'])} 个仍达到第二次上限。原预算记录保留；续算仍不使用趋势或contact反馈。",'']
    if hist['available']:
        lines.append(f"历史比较范围1.5–4.5的 body amplitude 与wave描述性ρ={hist['spearman_rho']:.3f}，相邻上升{len(hist['adjacent_increases'])}处；扩展范围0.5–9的ρ={ext['spearman_rho']:.3f}。总体下降与严格单调必须区分，优化未加入形状或交叉奖励。")
    lines+=['','比较对象是各wave分别优化并冻结的整套步态，body、leg yaw和leg pitch振幅一起变化。若出现交叉，它支持这些优化步态的性能排序随失联概率改变，不能仅凭本实验把原因单独归给body amplitude或波数本身。']
    lines+=['',link('最终振幅与验证图',ROOT/'validated_amplitudes.png'),'','| wave | body目标 ° | body实际1Hz中位幅值 ° | leg yaw目标 ° | leg pitch目标 ° | 无损 D mm | 减半步长差 |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in amplitudes['selected']:
        a=r['amplitude'];lines.append(f"| {r['wave']:g} | {a['body']:.3f} | {r['realized_body_median_deg']:.3f} | {a['leg_yaw']:.3f} | {a['leg_pitch']:.3f} | {r['D']:.3f} | {r['half_relative_difference']:.1%} |")
    lines+=['','优化变量是控制目标振幅；实际body列来自保存body yaw关节角的1Hz拟合，排除首个锁定参考的yaw后取中位数。它不是身体中心线的横向位移幅度，目标为零也不保证身体在外力下完全不弯曲。']
    definition_path=ROOT/'outputs/data/amplitude_definition_audit.json'
    if definition_path.exists():
        definition=read(definition_path)
        assert definition['physics_steps']==0 and not definition['changed_optimization']
        for p,digest in definition['source_hashes'].items():assert sha(p)==digest
        inputs.append(definition_path)
        lines+=['',link('目标角度、实际关节幅值与身体横向半宽',ROOT/'amplitude_definition_audit.png')+'。中心线半宽来自保存姿态的运动学重建，去除刚体平移和方向后计算；它是事后几何诊断，不替换优化变量，也不参与选点。']
    lines+=['',link('全精度冻结参数与完整资格记录',data/'validated_selection.json')]
    changed=sum(r.get('rank_in_eligible_nominal_bank',1)>1 for r in qualification_audit['rows'])
    lines += ['',f"有 {changed} 个波数最终选择了名义排名并非第一的候选，因为更快候选未通过完整资格条件。幅度变化、位移代价和逐项拒绝原因见 "+link('资格验证审计',ROOT/'qualification_audit.md')+'；'+link('选点变化图',ROOT/'qualification_audit.png')]
    outcomes=[]
    for name,title in [('contact_erase','持续随机失联'),('contact_erase_transient','短时随机失联与安全恢复')]:
        section,sources,outcome=channel_report(name,title);lines+=['',*section];inputs+=sources;outcomes.append(outcome)
    lines+=['','## 差异与根因记录','',
            '已确认历史PD、振幅搜索范围、选点方法和碰撞模型不一致。旧身体形状门槛对低波数大振幅有偏；修正为实际非相邻身体碰撞检查。匹配实验中，内部碰撞开关可反转wave1.5两组振幅的排名；单独降低摩擦未能反转该排名。身体与腿目标波形排查未发现大的幅值倍率或半节段相位遗漏。',
            '',link('根因摘要',ROOT/'ROOT_CAUSE_SUMMARY.md')+'；'+link('完整证据记录',ROOT/'ROOT_CAUSE_RECORD.md')+'；'+link('摩擦×自碰撞对照',ROOT/'friction_collision_diagnostic.png'),
            '', '原版Isaac Lab作为独立分支保留其61-spine正式指标、不同热身时间、额外头部自由度和碰撞配置，不能与MuJoCo直接混成同一条曲线。Clean未修改。',
            '', '图中均值与区间基于完整预定样本；失败不被删去以抬高性能。不完整数值轨迹没有补造完整位移。视频只可作为所标随机样本的运动核对，不能替代配对统计。','']
    native=ROOT.parents[2]/'Centipede_IsaacLab_RL/outputs/analysis/common_pd_amplitude_contact_20260909/search_cold/data/live_status.json'
    if native.exists():
        state=read(native)
        snapshot=ROOT/'outputs/data/final_native_status_snapshot.json'
        snapshot.write_text(json.dumps(dict(source=str(native),state=state),indent=2,ensure_ascii=False),encoding='utf-8')
        label='搜索筛选完成，仍需独立资格验证' if state['stage']=='search_screen_complete' else '该分支尚未记录完成的独立优化资格结果'
        lines += [f"原版分支在 {state.get('time','记录时刻')} 的快照：{label}；不能把其阶段性成绩称作最终最优。此状态不改变本报告的MuJoCo结论。",''];inputs.append(snapshot)
    lines += amendment_note
    report=ROOT/'FINAL_RESULT.md';report.write_text('\n'.join(lines),encoding='utf-8')
    receipt=dict(time=datetime.datetime.now().astimezone().isoformat(),experiment_workflow_complete=True,
                 desired_scientific_pattern_is_not_assumed=True,unresolved_waves=selection['unresolved'],channels=outcomes,
                 local_convergence=convergence,
                 joint_probe_polish=polish,
                 conditional_joint_finish=dict(decision=finish_decision,audit=finish_audit),
                 fixed_pd_sha256=amplitudes['all_pd_sha256'],report=str(report),report_sha256=sha(report),source_sha256=sha(__file__),
                 input_hashes={str(p):sha(p) for p in inputs})
    (ROOT/'FINAL_RECEIPT.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
