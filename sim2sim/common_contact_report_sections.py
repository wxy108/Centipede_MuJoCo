"""Render measured current status, including limits beyond forward displacement."""
import json
import shared as s
R=s.ROOT
def load(p):return json.loads((R/p).read_text(encoding='utf-8'))
def sections():
    centered=(R/'CENTERED_COMMON_CONTACT_CANDIDATE_PROFILE.json').exists()
    j=load('audit/common_contact_centered_summary.json' if centered else 'audit/common_contact_summary.json');accepted=all((R/name).exists() for name in ['COMMON_CONTACT_OPERATING_PROFILE.json','COMMON_CONTACT_VALIDATION_RECEIPT.json','COMMON_CONTACT_REPRODUCTION_MANIFEST.json']) and not j['missing'] and not j['failed']
    if accepted:accepted=load('COMMON_CONTACT_OPERATING_PROFILE.json')['status']=='validated_common_contact_cpu_gpu' and (not centered or load('audit/common_contact_centered_motion_gates.json')['status']=='PASSED')
    state='已通过本报告限定的物理参数、前进性能、XY 轨迹、方向和半步长验收' if accepted else f"仍在验证：缺失 {len(j['missing'])} 项，超限 {len(j['failed'])} 项"
    intro=f'**共用物理模型、同一套 PD 和原点布局下的共用柔顺接触基准，{state}。** 原生接触方案 PGS / TGS 均未通过完整数值验收；当前结果依赖明确更换后的共用接触力律，不能解释为原生刚性接触求解器已经等价。'
    metrics=[]
    for key,label in [('cpu_gap_pct','MuJoCo / PhysX CPU 最大前进位移差'),('gpu_gap_pct','MuJoCo / PhysX GPU 最大前进位移差'),('cpu_gpu_gap_pct','PhysX CPU / GPU 最大前进位移差'),('mu_refinement_pct','MuJoCo 半步长最大变化'),('cpu_refinement_pct','CPU 半步长最大变化'),('gpu_refinement_pct','GPU 半步长最大变化')]:
        values=[r[key] for r in j['rows'] if r[key] is not None];value=f'{max(values):.5f}%' if len(values)==8 else f'待完成（{len(values)}/8）';metrics.append(f'| {label} | {value} |')
    overview='\n\n| 当前共用接触基准 | 八组实测最大值 |\n|---|---:|\n'+'\n'.join(metrics)
    overview+='\n\n表中跨后端前进位移差使用主步长 31.25 µs；半步长变化比较各后端自身 31.25 与 15.625 µs 的结果。'
    if all(r[f'{engine}_dt{grid}_mm'] is not None for r in j['rows'] for engine in ['mu','cpu','gpu'] for grid in [31,15]):
        comparisons=[]
        for grid in [31,15]:
            for first,second in [('mu','cpu'),('mu','gpu'),('cpu','gpu')]:
                for r in j['rows']:
                    x,y=r[f'{first}_dt{grid}_mm'],r[f'{second}_dt{grid}_mm'];comparisons.append((100*abs(x-y)/max((abs(x)+abs(y))/2,10),grid,first,second,r['wave'],r['family']))
        worst=max(comparisons)
        overview+=f"\n\n将两档步长的全部后端成对比较都计入，最大前进位移差为 **{worst[0]:.5f}%**，出现在 {worst[2].upper()} / {worst[3].upper()}、{31.25 if worst[1]==31 else 15.625} µs、wave {worst[4]} / {worst[5]}。"
    overview+='\n\n统一模型总质量为 2.536051087 g，251 个刚体、173 个转动关节；三种后端共用 SI 参数、PD、幅值和指令。主步长 31.25 µs，细步长 15.625 µs，指令保持 250 µs。位移定义为 21 个身体原点平均位置沿 −X 在 2–6 s 的变化。下图的横纵坐标完全一致，曲线接近重合；差异另有放大图。这是固定幅值步态验证，不是新一轮幅值优化。原八组被逐方案复用，应称固定验证集，不能声称是从未查看过的盲测。\n\n'
    overview+=f"![共用物理后的完整对照]({(R/('figures/23_common_contact_centered_full_bank.png' if centered else 'figures/19_common_contact_full_bank.png')).as_posix()})\n\n"
    overview+=f"![放大后的实际差异]({(R/('figures/25_common_contact_centered_difference_detail.png' if centered else 'figures/21_common_contact_difference_detail.png')).as_posix()})\n\n"
    if centered:
        gates=load('audit/common_contact_centered_motion_gates.json')
        overview+=f"独立环境都使用相同的原点位置，保持关节链和外力计算彼此独立，不做运行中的位置复位。额外要求跨引擎及各自半步长的 XY 轨迹差 ≤0.5 mm、头尾弦线方向差 ≤0.25°；共 72 组逐轨迹比较。当前运动检查：{gates['status']}，完成 {len(gates['rows'])}/72。\n\n![额外运动验收]({(R/'figures/26_common_contact_centered_motion_checks.png').as_posix()})\n\n"
    detail='### 轨迹、接触变形与计算代价\n\n'
    motion_path='audit/common_contact_centered_motion_comparison.json' if centered else 'audit/common_contact_motion_comparison.json'
    if (R/motion_path).exists():
        motion=load(motion_path)['rows']
        detail+='前进位移之外还逐帧检查 XY 轨迹、身体方向及关节角。方向是头尾身体弦线的平面角度，不是刚性根方向；轨迹均相对 2 s 位置。以下列出已完成数据，未把横向残差藏入前进指标。\n\n| 后端 / 步长 | 最大 XY 差 (mm) | 最大方向差 (°) | 最大关节角差 (°) |\n|---|---:|---:|---:|\n'
        for device in ['cpu','gpu']:
            for grid in [31.25,15.625]:
                rr=[r for r in motion if r['device']==device and r['grid_us']==grid]
                if rr:detail+=f"| {device.upper()} / {grid} µs | {max(r['max_centroid_xy_path_difference_mm'] for r in rr):.6f} | {max(r['max_heading_difference_deg'] for r in rr):.6f} | {max(r['max_joint_position_difference_deg'] for r in rr):.6f} |\n"
    if (R/'audit/common_contact_coordinate_precision.json').exists():
        coord=load('audit/common_contact_coordinate_precision.json');rr=coord['gait_rows']
        detail+='\n针对最大横向差的 4.5/A 做了完整六秒因果对照：只将全局 Y 从 1.75 m 移到 0；高 Y 副本与原整组对照逐位一致。\n\n| 步长 (µs) | 全局 Y (m) | 最大 XY 差 (mm) | 横向位移：CPU / Mu (mm) |\n|---|---:|---:|---:|\n'
        for r in rr:detail+=f"| {r['grid_us']} | {r['global_y_m']} | {r['max_xy_difference_mm']:.6f} | {r['lateral_displacement_mm']:.6f} / {r['mu_lateral_displacement_mm']:.6f} |\n"
        if coord['translation_rows']:
            detail+='\n独立无重力、无接触、零关节运动的匀速平移进一步复现位置依赖。两份机器人都保持 3 mm/s，0.5 s 解析位移应为 1.5 mm：\n\n| 步长 (µs) | 全局 Y (m) | 实测位移 (mm) | 简单 float32 逐步相加预测 (mm) |\n|---|---:|---:|---:|\n'
            for r in coord['translation_rows']:detail+=f"| {r['grid_us']} | {r['global_y_m']} | {r['measured_displacement_mm']:.9f} | {r['float32_repeated_addition_displacement_mm']:.9f} |\n"
            detail+='\n1.75 m 处 float32 坐标间距约 119.2 nm，单步运动量已与它相当或更小。实测证明坐标位置会影响积分结果；简单的“每整步加一次 float32”并不能解释所有实测值，因此不把该简化算式当成 PhysX 内部实现。公开的 PhysX 代码在根位姿积分后还有链式姿态传播、整体 COM 和动量修正，说明完整 articulation 更新不止一次相加；这里只作为实现背景，不冒充已确认安装二进制的精确版本。[PhysX articulation 更新源码](https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/main/physx/source/lowleveldynamics/src/DyFeatherstoneForwardDynamic.cpp)。这个诊断也不证明历史原生接触差异仅由精度造成。主测试数据及验收门槛保持原样，靠近原点的诊断没有替换较差的原数据。\n'
        detail+='\n因此，前进性能接近不代表任意场景偏移下所有轨迹都收敛。微小机器人后续批量环境须独立核查坐标尺度；单纯缩小步长可能恶化位置量化。\n'
        detail+='\n在输出观测时减去环境原点，不能补回积分过程中已经损失的位移。若波数或优化候选编号与环境位置相关，位置偏差可能混入比较；这是本次证据支持的风险机制，尚未追溯证明历史每个最优幅值都受它影响。当前修复作用于实际物理坐标布局。\n'
        if (R/'audit/common_contact_translation_step_arithmetic.json').exists():detail+='\n追加的逐物理步记录中，采用实际质量、COM 位置和速度，独立加入整体质心位置修正后，dt31 远处测试的 64 步根 COM 更新全部逐位吻合；仅根位置相加则仅 1 步吻合。这为该无接触平移异常提供了更具体的机制证据。原点处仍有约 0.015 nm 的算术模型残差，未将近似运算当成完整 PhysX 重实现。细步长完整步态对照也完成：移到原点后 4.5/A 最大 XY 差由 3.550056 mm 降到 0.024404 mm。\n'
        detail+=f"\n![场景坐标位置的独立对照]({(R/'figures/22_common_contact_coordinate_precision.png').as_posix()})\n"
    quality=[];meshes=[];full=[];runtimes=[]
    for grid in [31,15]:
        for dev in ['cpu','gpu']:
            name=f'native_common_contact_{"centered_" if centered else ""}validation_{dev}_dt{grid}_000';base=f'runs/{name}/'
            if not (R/(base+'complete.json')).exists():continue
            res=load(base+'results.json');runtimes.append((dev,grid,res['wall_s']/60));full.append(res['max_contact_depth_m']*1e6)
            if (R/(base+'quality_audit.json')).exists():quality+=load(base+'quality_audit.json')['rows']
            if (R/(base+'compliance_geometry_audit.json')).exists():meshes+=load(base+'compliance_geometry_audit.json')['rows']
    if quality:
        detail+=f"\n已完成原生数据中，2–6 s 采样的最大球面压缩为 {max(-q['min_foot_sphere_surface_z_m']*1e6 for q in quality):.3f} µm，网格支持压缩最大 {max(r['maximum_mesh_compression_um'] for r in meshes):.3f} µm；完整 0–6 s 含启动阶段的最大候选点深度为 {max(full):.3f} µm。这些刚体几何重叠用于表示等效柔顺变形。身体最低向上投影 {min(q['min_spine_up_dot'] for q in quality):.6f}，未出现翻转；姿态重建最大拟合残差 {max(q['max_pose_fit_residual_m'] for q in quality)*1e6:.3f} µm。网格压缩仅是采样几何量，不冒充接触概率。\n"
    if runtimes:
        detail+='\n当前实现每步在主机计算共用接触力，再通过原生接口施加，计算较慢。已完成九环境六秒运行的实际耗时：'+ '；'.join(f'{d.upper()} {31.25 if g==31 else 15.625} µs：{t:.1f} 分钟' for d,g,t in runtimes)+'。这是审计与机械验证入口；没有把它包装成已具备大规模 RL 训练吞吐的系统。\n'
    detail+='\nGPU 运行回读确认 GPU dynamics、GPU broadphase 和 cuda:0 张量后端。日志仍包含安装环境的 Warp / CUDA 入口、插件与弃用提示；有效运行未发现导致中断的物理错误，不宣称原生日志零警告。\n'
    detail+='\n本轮验收针对列出的两档步长和六秒固定步态。GPU 在细步长下的方向残差更大；通过绝对误差门槛，不等于所有误差都随步长减小而单调收敛。其他 PD、幅值、限位冲击、地形及更长轨迹未由本次结果覆盖。\n'
    if (R/'audit/hardware_identity.json').exists():
        hardware=load('audit/hardware_identity.json')['driver_devices'][0]
        detail+=f"\n实际驱动查询与 Torch 均识别为 {hardware['name']}，驱动 {hardware['driver_version']}。启动图形日志的设备名称却显示 GTX 1050 Ti；这处名称不一致已原样记录，尚未诊断其原因。GPU 后端判断依据实际驱动、CUDA 张量设备及物理场景回读。证据见 `audit/hardware_identity.json`；GPU 复现入口另核对设备和驱动版本。\n"
    if (R/'audit/common_joint_limit_readback.json').exists():
        limits=load('audit/common_joint_limit_readback.json');detail+=f"\n另核对全部 173 个关节限位，CPU/GPU 与 Mu 的最大范围回读差为 {max(r['maximum_bounded_error_rad'] for r in limits['rows']):.3e} rad。已完成轨迹的 100 Hz 采样在 2–6 s 内距限位最近仍有 {min(r['minimum_sampled_limit_clearance_deg'] for r in limits['trajectory_limit_clearances']):.3f}°。这是采样范围核查，未将其表述为逐子步限位冲击已验证；超范围驱动的限位碰撞行为不属于这次固定步态验证。\n"
    usage=('正式采用文件为 `COMMON_CONTACT_OPERATING_PROFILE.json`，参数与代码校验和见 `COMMON_CONTACT_REPRODUCTION_MANIFEST.json`，完整验收回执见 `COMMON_CONTACT_VALIDATION_RECEIPT.json`。' if accepted else '正式共用接触版本尚未采用；复现入口在验收完成前不会启动仿真。')
    usage+=' 两个原项目只增加日期化的可选入口。旧 PGS / TGS 候选保留拒绝状态和原始轨迹；旧视频不作为本次成功证据。新 `common_contact_*` 视频使用保存的关节角，以及从身体原点位置拟合的根位姿，用共用几何生成回放；拟合残差在上文报告。没有重新求解物理，也不是 Isaac 原生渲染。\n\n原 RL 观测、奖励、接触传感器、reset、terrain 和 policy 训练管线尚未迁移到新接触定义；直接禁用原生接触后，原生 contact sensor 不能代表这套外加接触力。当前可用范围是已验证的开环平地机械基准。'
    runtime=f'## Current numerical profile\n\nShared compliant-contact benchmark: {state}. Native rigid-contact PGS and TGS candidates remain rejected. Use COMMON_CONTACT_OPERATING_PROFILE.json only when present and validated. Contact forces use COMMON_CONTACT_V2.json in both engines; native ground contact is disabled.\n\nMain physics step is 31.25 microseconds; refinement is 15.625 microseconds. Command period remains 250 microseconds. Native PGS16/4, two CPU workers, TGS-only external iteration flag false. Mu uses the physically derived M+hD+h^2K adapter; numeric armature is not physical rotor inertia.\n\nThe tested bank uses Y offsets from zero to two metres. Origin-dependent lateral drift is measured and reported; acceptance concerns the stated forward metric, not arbitrary world-position invariance. GPU and CPU are independently validated. No RL training pipeline has been migrated. See REPORT.md, common_contact_motion_comparison.csv and the coordinate-precision audit.\n'
    if centered:runtime=runtime.replace('The tested bank uses Y offsets from zero to two metres. Origin-dependent lateral drift is measured and reported; acceptance concerns the stated forward metric, not arbitrary world-position invariance.','The current bank places all independent articulations at the same numerical origin. Native collisions remain disabled and contact forces are separate per environment. Additional XY-path and chord-heading gates are required across engines and across time-step refinement. The displaced-origin bank remains preserved as a rejected coordinate layout, despite its excellent forward agreement.')
    if centered:runtime=runtime.replace('common_contact_motion_comparison.csv','common_contact_centered_motion_comparison.csv')
    return dict(intro=intro,overview=overview,detail=detail,usage=usage,runtime=runtime,state=state,accepted=accepted)
