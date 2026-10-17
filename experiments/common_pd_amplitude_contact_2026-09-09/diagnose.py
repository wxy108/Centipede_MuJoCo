"""Read-only matched bridge analysis. Does not change acceptance or select gaits."""
import collections,json,math
import core as c
from prepare import anchors
np=c.np


def natural_diagnostic(m):
    # August 31 naturality limits, applied to the present observer's metrics.
    # Metric implementations/windows differ: this is NOT an exact old-policy replay.
    bounds={'body_wave_residual_p95':.40,'leg_yaw_wave_residual_p95':.18,
      'leg_pitch_wave_residual_p95':.40,'leg_yaw_speed_p99':8.,
      'leg_pitch_speed_p99':7.,'leg_yaw_reversals_p95':3.,
      'leg_pitch_reversals_p95':6.25,'leg_pitch_tracking_rms_deg':10.,
      'swing_foot_height_p95_mm':5.5,'left_right_duty_difference':.10}
    failed=[k for k,v in bounds.items() if m[k]>v]
    if not .15<=m['contact_duty_mean']<=.65:failed.append('contact_duty_mean')
    return failed


def harmonic(t,y):
    X=np.stack([np.sin(2*np.pi*t),np.cos(2*np.pi*t),np.ones_like(t)],axis=1)
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    return np.degrees(np.hypot(b[0],b[1]))


def main():
    tasks=c.read(c.DATA/'banks/bridge.json');history=anchors();rows=[];missing=[]
    for task in tasks:
        key=c.identity(task);path=c.OUT/'assessments'/f'{key}.json'
        if not path.exists():missing.append(key);continue
        r=c.read(path);a=[r['amplitudes'][k] for k in c.KEYS]
        family=next(k for k,v in history.items() if task['wave'] in v and np.allclose(a,v[task['wave']],rtol=0,atol=1e-10))
        z=np.load(r['motion_path']);q=(z['t']>=2)&(z['t']<6)
        body=harmonic(z['t'][q],z['bq'][q,1:,0])
        rr=dict(id=key,wave=r['wave'],family=family,plant=task['plant'],phase=task['phase'],
            amplitude=r['amplitudes'],D=r['D'],failed=r['failed'],feasible=r['feasible'],
            metrics=r['metrics'],pd_sha=r['pd_readback']['active_pd_sha256'],
            realized_body_harmonic_median_deg=float(np.median(body)),
            realized_body_harmonic_min_deg=float(np.min(body)),realized_body_harmonic_max_deg=float(np.max(body)),
            naturality_limit_diagnostic_failed=natural_diagnostic(r['metrics']),record=r['record'],motion=r['motion_path'])
        rows.append(rr)
    paired=[]
    for treatment,levels,fixed in [('phase',['legacy','midpoint'],['wave','family','plant']),
                                  ('plant',['august31','current'],['wave','family','phase'])]:
        groups=collections.defaultdict(dict)
        for r in rows:groups[tuple(r[k] for k in fixed)][r[treatment]]=r
        for group,levelsdata in groups.items():
            if not all(x in levelsdata for x in levels):continue
            a,b=[levelsdata[x] for x in levels]
            assert a['pd_sha']==b['pd_sha']
            paired.append(dict(treatment=treatment,**dict(zip(fixed,group)),reference=a['id'],changed=b['id'],
              levels=levels,reference_D=a['D'],changed_D=b['D'],difference_mm=b['D']-a['D'],
              reference_failed=a['failed'],changed_failed=b['failed']))
    payload=dict(status='COMPLETE_MATCHED_DIAGNOSTIC' if not missing else 'PARTIAL',missing=missing,rows=rows,paired=paired,
      caveats=['Naturality limits use current observer metrics, not an exact historical metric replay.',
        'One amplitude vector per historical family/wave is a transfer test, not a reoptimization.',
        'Command amplitudes are not independent evidence for an optimized trend.'])
    c.write(c.DATA/'bridge_analysis.json',payload)
    text=['# 统一 PD 对照：根因记录','',f"状态：{payload['status']}；{len(rows)}/{len(tasks)} 个完整对照。",
      '', '所有对照使用同一组实际 Kp/Kv；只改变标明的振幅、被动足尖模型或腿相位。位移单位 mm。',
      '', '| 波数 | 振幅来源 | 足尖模型 | 腿相位 | body 指令/实测一阶幅值 ° | 位移 | 工程失败项 |',
      '|---:|---|---|---|---:|---:|---|']
    for r in sorted(rows,key=lambda r:(r['wave'],r['family'],r['plant'],r['phase'])):
        text.append(f"| {r['wave']:g} | {r['family']} | {r['plant']} | {r['phase']} | {r['amplitude']['body']:.2f} / {r['realized_body_harmonic_median_deg']:.2f} | {r['D']:.2f} | {', '.join(r['failed']) or '通过'} |")
    text+=['','## 已确认与待验证','',
      '1. 历史图不是相同实验：RL v4 每个波数的 PD 不同，并含平滑选点/其他环境筛选；9 月 8 日恢复也用了 8 组 PD。8 月 31 日才是固定 PD 的振幅优化，但 PD、足尖、筛选与当前研究仍不同。',
      '2. 几何证据：两个 MuJoCo 模型的腿根都在对应体关节前半节段。RL 的腿根相位补偿 +0.5Δ 与此相符，原 MuJoCo 没有这项。成对位移差见 bridge_analysis.json；修正不保证每个波数改善。',
      '3. 原 spine≤1.2 限制是弧长/首尾距离。正常低波数大摆幅也会增加这个比值。RL 的 wave 1.5、body 32.27° 移入当前系统后，完整轨迹仅在此项失败；这证明筛选存在形状偏好，但不能证明解除它会得到速度最优趋势。',
      '4. body 实测幅值使用 2–6 s 的 1 Hz 正弦/余弦最小二乘，排除锁定的首关节。它与形状的横向摆幅分别记录，避免混淆。',
      '5. 历史自然性门槛按当前指标进行诊断复算，未用于更改本次主选择。前进位移、物理稳定和波形要求会偏好不同步态。',
      '', '主优化仍独立最大化各波数通过工程门槛的位移；不按振幅单调性筛选。对照结果不是最终优化结果。','']
    (c.ROOT/'ROOT_CAUSE_RECORD.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(dict(status=payload['status'],rows=len(rows),pairs=len(paired),missing=len(missing))))


if __name__=='__main__':main()
