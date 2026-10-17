def run_task(task, *, initialize=None, case_override=None, output_dir=None):
    cfg,wave,duration,dt_scale,save_trace=task
    t0=time.monotonic()
    ev=opt.FastEvaluator(); m,d,ix=ev.model,ev.data,ev.index
    apply_impedance(m,ix,cfg)
    m.opt.timestep*=dt_scale
    dt=float(m.opt.timestep); stride=max(1,int(round(.001/dt)))
    c=wave_case(wave) if case_override is None else dict(case_override, wave=wave)
    output_dir=OUT if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    control=opt.ThreeAmplitudeController(m,ix,wave,c['body'],c['leg_yaw'],c['leg_pitch'])
    body_ids=np.stack((ix.active_joint_ids[ix.body_positions],ix.active_joint_ids[ix.pitch_positions],ix.roll_joint_ids),axis=-1)
    qa=m.jnt_qposadr[body_ids]; va=m.jnt_dofadr[body_ids]
    foot_sites=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,f'site_foot_{s}_{side}') for s in range(19) for side in range(2)])
    foot_map={int(g):i for i,g in enumerate(ev.foot_geoms)}
    body_map={int(b):i for i,b in enumerate(ix.chassis_body_ids)}
    legva=ix.dof_addresses[ix.leg_positions.reshape(-1)]; legqa=ix.qpos_addresses[ix.leg_positions.reshape(-1)]
    ground_categories=np.full(m.ngeom,59,dtype=np.int64)
    for g in range(m.ngeom):
        if g in foot_map:ground_categories[g]=foot_map[g]
        elif int(m.geom_bodyid[g]) in body_map:ground_categories[g]=38+body_map[int(m.geom_bodyid[g])]
    n=int(round(duration/dt)); weight=float(m.body_mass.sum()*9.81)
    force=np.zeros((n,60),dtype=np.float64) # 38 feet, 21 chassis, other
    pen=np.zeros(n); all_speed=np.zeros(n); all_tip_speed=np.zeros(n); torque=np.zeros(n)
    abswork=np.zeros(n); poswork=np.zeros(n); tipmax=np.zeros(n); body_speed=np.zeros((n,3))
    accmax=np.zeros((n,3)); foot_mask=np.zeros((n,38),dtype=bool)
    times=[]; bq=[]; bv=[]; om=[]; xpos=[]; rootmat=[]; worldroll=[]; lq=[]; lv=[]; lt=[]; tq=[]; fq=[]; fs=[]
    prevel=np.empty(m.nv); cf=np.zeros(6)
    mujoco.mj_resetData(m,d); mujoco.mj_forward(m,d)
    initialization=initialize(ev) if initialize is not None else {'method':'original_xml_reset'}
    self_contact_steps=0;self_contact_points=0;self_penetration_peak=0.
    finite=True
    channel=CONTACT_CHANNEL(ev,dt,output_dir)
    for k in range(n):
        channel.before(k)
        prevel[:]=d.qvel
        if k%stride==0:
            times.append(k*dt); bq.append(d.qpos[qa].copy()); bv.append(d.qvel[va].copy())
            lq.append(d.qpos[legqa].copy()); lv.append(d.qvel[legva].copy()); tq.append(d.qpos[ix.tip_qpos_addresses].copy())
        control.step(d)
        mujoco.mj_step(m,d)
        channel.after(k)
        torque[k]=np.max(np.abs(d.actuator_force[ix.active_actuator_ids]))
        power=d.actuator_force[ix.active_actuator_ids]*prevel[ix.dof_addresses]
        abswork[k]=np.sum(np.abs(power))*dt; poswork[k]=np.sum(np.maximum(power,0))*dt
        body_speed[k]=np.max(np.abs(d.qvel[va]),axis=0)
        accmax[k]=np.max(np.abs(d.qacc[va]),axis=0)
        all_speed[k]=np.max(np.abs(d.qvel[ix.dof_addresses]))
        all_tip_speed[k]=np.max(np.abs(d.qvel[ix.tip_dof_addresses]))
        tipmax[k]=np.max(np.abs(d.qpos[ix.tip_qpos_addresses]))
        pen[k]=ACCUMULATE_GROUND(d.contact.geom1,d.contact.geom2,d.contact.efc_address,
            d.contact.dim,d.contact.dist,d.efc_force,ev.ground,ground_categories,
            int(m.opt.cone)==0,force[k],foot_mask[k])
        if k*dt>=1.5:
            count,depth=OBSERVE_SELF_CONTACT(d.contact.geom1,d.contact.geom2,d.contact.dist,ground_categories)
            self_contact_steps+=int(count>0);self_contact_points+=count
            self_penetration_peak=max(self_penetration_peak,depth)
        if k%stride==0:
            lt.append(control.target[ix.leg_positions.reshape(-1)].copy())
            om.append(d.cvel[ix.chassis_body_ids,:3].copy()); xpos.append(d.xpos[ix.chassis_body_ids].copy())
            rootmat.append(d.xmat[int(ix.chassis_body_ids[0])].reshape(3,3).copy())
            worldroll.append(opt.tip.signed_world_roll_deg(d,ix.chassis_body_ids))
            fq.append(d.site_xpos[foot_sites].copy()); fs.append(k)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            finite=False; break
    a=dict(t=np.asarray(times),bq=np.asarray(bq),bv=np.asarray(bv),om=np.asarray(om),xpos=np.asarray(xpos),rootmat=np.asarray(rootmat),worldroll=np.asarray(worldroll),
           lq=np.asarray(lq),lv=np.asarray(lv),lt=np.asarray(lt),tq=np.asarray(tq),fq=np.asarray(fq),fs=np.asarray(fs,dtype=int))
    good=a['t']>=1.5; startstep=int(round(1.5/dt)); fine=slice(startstep,k+1)
    t=a['t'][good]; x=a['xpos'][good]; center=x.mean(axis=1)
    path=float(np.linalg.norm(np.diff(center[:,:2],axis=0),axis=1).sum())
    displacement=float(opt.base.FORWARD_SIGN*(center[-1,0]-center[0,0]))
    # No MuJoCo calls or parameter updates during postprocessing.
    bq1=a['bq'][good]; bv1=a['bv'][good]; om1=a['om'][good]
    lag=int(round(.01/(dt*stride)))
    dq=np.degrees(np.abs(bq1[lag:]-bq1[:-lag]))
    dom=np.linalg.norm(om1[lag:]-om1[:-lag],axis=-1)
    rm=a['rootmat'][good]; rootroll=np.degrees(np.arctan2(rm[:,2,1],rm[:,2,2]))
    wr=np.radians(a['worldroll'][good])
    coherent=np.degrees(np.arctan2(np.sin(wr).mean(axis=1),np.cos(wr).mean(axis=1)))
    front=np.degrees(np.arctan2(np.sin(wr[:,:7]).mean(axis=1),np.cos(wr[:,:7]).mean(axis=1)))
    f=force[fine]; total=f.sum(axis=1); chassis=f[:,38:59].sum(axis=1)
    loaded=f[:,:38]>1e-7
    sample_idx=a['fs'][good]
    contacts=foot_mask[sample_idx]
    legq=a['lq'][good].reshape(-1,19,2,2); legv=a['lv'][good].reshape(-1,19,2,2)
    target=a['lt'][good].reshape(-1,19,2,2); tips=np.degrees(np.abs(a['tq'][good]))
    # Original shape/reversal diagnostics use 100Hz, including their declared
    # 50ms reversal smoothing; no hidden rescaling of that filter.
    oldstride=max(1,int(round(.01/(dt*stride)))); tt=t[::oldstride]
    yaw=legq[...,0].reshape(len(t),-1); pitch=legq[...,1].reshape(len(t),-1)
    yawv=legv[...,0].reshape(len(t),-1); pitchv=legv[...,1].reshape(len(t),-1)
    cycle_d=[]
    for begin in np.arange(2.,duration-.999,1.):
        i=int(np.argmin(np.abs(t-begin))); j=int(np.argmin(np.abs(t-(begin+1))))
        cycle_d.append(float(opt.base.FORWARD_SIGN*(center[j,0]-center[i,0])))
    fv=np.diff(a['fq'][good],axis=0)/(dt*stride)
    slip=np.linalg.norm(fv[:,:,:2],axis=-1)[loaded[sample_idx[1:]-startstep]]
    sampled_chassis=(chassis>1e-7)
    mtr=dict(displacement_mm=displacement*1000,path_efficiency=displacement/max(path,1e-15),
             min_upright=float(rm[:,2,2].min()),spine_tortuosity_max=float(np.max(np.linalg.norm(np.diff(x,axis=1),axis=2).sum(axis=1)/np.maximum(np.linalg.norm(x[:,-1]-x[:,0],axis=1),1e-15))),
             root_roll_abs_peak_deg=float(np.abs(rootroll).max()),root_roll_abs_p95_deg=pct(np.abs(rootroll),.95),
             coherent_roll_p95_deg=pct(np.abs(coherent),.95),front_roll_p95_deg=pct(np.abs(front),.95),
             center_height_ptp_mm=float(np.ptp(center[:,2])*1000),center_vertical_speed_p99_m_s=pct(np.abs(np.diff(center[:,2])/(dt*stride)),.99),
             body_yaw_speed_peak=float(body_speed[fine,0].max()),body_pitch_speed_peak=float(body_speed[fine,1].max()),body_roll_speed_peak=float(body_speed[fine,2].max()),
             body_deltaomega_10ms_peak=float(dom.max()),body_pitch_delta10ms_deg=float(dq[:,:,1].max()),body_roll_delta10ms_deg=float(dq[:,:,2].max()),
             body_accel_peak_rad_s2=accmax[fine].max(axis=0).tolist(),
             ground_force_peak_bw=float(total.max()/weight),ground_force_p99_bw=pct(total/weight,.99),chassis_force_peak_bw=float(chassis.max()/weight),foot_force_peak_bw=float(f[:,:38].max()/weight),
             normal_impulse_10ms_bw_s=window_integral(total/weight,dt),chassis_impulse_10ms_bw_s=window_integral(chassis/weight,dt),
             chassis_contact_fraction=float(np.mean(sampled_chassis)),
             mean_contacting_feet=float(contacts.sum(axis=1).mean()),ballistic_gap_s=max_run(~loaded.any(axis=1))*dt,
             penetration_p95_mm=pct(pen[fine][pen[fine]>0]*1000,.95),penetration_peak_mm=float(pen[fine].max()*1000),
             active_speed_peak=float(all_speed[fine].max()),tip_speed_peak=float(all_tip_speed[fine].max()),tip_angle_peak_deg=float(np.degrees(tipmax[fine].max())),
             tip_angle_p95_deg=pct(tips,.95),tip_stop_fraction=float(np.mean(tips>=37.75)),
             torque_peak_nm=float(torque.max()),torque_steady_peak_nm=float(torque[fine].max()),
             absolute_work_mJ=float(abswork[fine].sum()*1000),positive_work_mJ=float(poswork[fine].sum()*1000),work_per_distance_n=float(abswork[fine].sum()/max(displacement,1e-15)),
             cycle_displacement_mm=[v*1000 for v in cycle_d],cycle_displacement_cv=float(np.std(cycle_d)/max(np.mean(cycle_d),1e-15)) if cycle_d else float('nan'),
             loaded_foot_slip_p95_mm_s=pct(slip*1000,.95),
             left_right_force_imbalance_rms_bw=float(np.sqrt(np.mean((f[:,:38:2].sum(axis=1)-f[:,1:38:2].sum(axis=1))**2))/weight),
             body_wave_residual_p95=pct(pilot.sinusoid_residual(tt,bq1[::oldstride,1:,0]),.95),
             leg_yaw_wave_residual_p95=pct(pilot.sinusoid_residual(tt,yaw[::oldstride]),.95),leg_pitch_wave_residual_p95=pct(pilot.sinusoid_residual(tt,pitch[::oldstride]),.95),
             leg_yaw_speed_p99=pct(np.abs(yawv),.99),leg_pitch_speed_p99=pct(np.abs(pitchv),.99),
             leg_yaw_reversals_p95=pct(pilot.reversal_rate(tt,yawv[::oldstride]),.95),leg_pitch_reversals_p95=pct(pilot.reversal_rate(tt,pitchv[::oldstride]),.95),
             leg_pitch_tracking_rms_deg=float(np.degrees(np.sqrt(np.mean((legq[...,1]-target[...,1])**2)))),
             leg_yaw_tracking_rms_deg=float(np.degrees(np.sqrt(np.mean((legq[...,0]-target[...,0])**2)))),
             swing_foot_height_p95_mm=pct(a['fq'][good,:,2][~contacts]*1000,.95),
             contact_duty_mean=float(contacts.mean()),left_right_duty_difference=float(np.abs(contacts.mean(axis=0).reshape(19,2)[:,0]-contacts.mean(axis=0).reshape(19,2)[:,1]).mean()))
    mtr.update(nonadjacent_chassis_contact_steps=int(self_contact_steps),
        nonadjacent_chassis_contact_points=int(self_contact_points),
        nonadjacent_chassis_penetration_peak_mm=float(self_penetration_peak*1000))
    warnings=opt.tip.stage1.warning_counts(d)
    rest_mask=(a['t']>=.9)&(a['t']<1.)
    rest=dict(body_speed_peak_rad_s=float(np.abs(a['bv'][rest_mask]).max()),
              tip_angle_peak_deg=float(np.degrees(np.abs(a['tq'][rest_mask]).max())),
              center_height_ptp_mm=float(np.ptp(a['xpos'][rest_mask].mean(axis=1)[:,2])*1000),
              normal_force_peak_bw=float(force[int(round(.9/dt)):int(round(1./dt))].sum(axis=1).max()/weight))
    gates=dict(finite_warning_free=finite and not warnings,positive_forward=displacement>0,path_efficiency=mtr['path_efficiency']>=.8,upright=mtr['min_upright']>=math.cos(math.radians(50)),
        spine=mtr['spine_tortuosity_max']<=1.2,root_roll_peak=mtr['root_roll_abs_peak_deg']<=20,height=mtr['center_height_ptp_mm']<=2,vertical_speed=mtr['center_vertical_speed_p99_m_s']<=.1,
        coherent_roll=mtr['coherent_roll_p95_deg']<=10,front_roll=mtr['front_roll_p95_deg']<=12,
        body_yaw_speed=mtr['body_yaw_speed_peak']<=12,body_pitch_speed=mtr['body_pitch_speed_peak']<=8,body_roll_speed=mtr['body_roll_speed_peak']<=5,body_velocity_jump=mtr['body_deltaomega_10ms_peak']<=10,
        pitch_10ms=mtr['body_pitch_delta10ms_deg']<=2,roll_10ms=mtr['body_roll_delta10ms_deg']<=2,
        total_force=mtr['ground_force_peak_bw']<=2.5,chassis_force=mtr['chassis_force_peak_bw']<=1,foot_force=mtr['foot_force_peak_bw']<=1,chassis_contact=mtr['chassis_contact_fraction']<=.03,
        feet_count=6<=mtr['mean_contacting_feet']<=28,no_ballistic=mtr['ballistic_gap_s']<=.08,penetration=mtr['penetration_p95_mm']<=.1,
        active_speed=mtr['active_speed_peak']<=100,tip_speed=mtr['tip_speed_peak']<=1000,tip_peak=mtr['tip_angle_peak_deg']<=40,tip_p95=mtr['tip_angle_p95_deg']<=33,tip_stops=mtr['tip_stop_fraction']<=.03,
        torque=mtr['torque_peak_nm']<=2e-4,cycle_consistency=mtr['cycle_displacement_cv']<=.15,
        body_wave=mtr['body_wave_residual_p95']<=.4,leg_yaw_wave=mtr['leg_yaw_wave_residual_p95']<=.18,leg_pitch_wave=mtr['leg_pitch_wave_residual_p95']<=.4,
        leg_yaw_speed=mtr['leg_yaw_speed_p99']<=8,leg_pitch_speed=mtr['leg_pitch_speed_p99']<=7,leg_yaw_reversals=mtr['leg_yaw_reversals_p95']<=3,leg_pitch_reversals=mtr['leg_pitch_reversals_p95']<=6.25,
        pitch_tracking=mtr['leg_pitch_tracking_rms_deg']<=10,swing_height=mtr['swing_foot_height_p95_mm']<=5.5,duty=.15<=mtr['contact_duty_mean']<=.65,lr_duty=mtr['left_right_duty_difference']<=.1)
    tag=f"{cfg['name']}_w{wave:g}_t{duration:g}_dt{dt:g}".replace('.','p')
    result=dict(id=tag,config=cfg,case=c,dt=dt,duration=duration,metrics=mtr,gates=gates,warnings=warnings,
                startup=dict(torque_peak_nm=float(torque[:startstep].max()),body_speed_peak=body_speed[:startstep].max(axis=0).tolist(),ground_force_peak_bw=float(force[:startstep].sum(axis=1).max()/weight)),
                elapsed_s=time.monotonic()-t0,model_sha256=opt.tip.sha256(opt.MODEL),initialization=initialization,rest_before_gait=rest,
                baseline_relative_gates='added during bank comparison, never assumed passed')
    channel.finish(result)
    if save_trace:
        np.savez_compressed(output_dir/f'{tag}.npz',**{key:value.astype(np.float32) if value.dtype.kind=='f' else value for key,value in a.items()},force=force.astype(np.float32),torque=torque.astype(np.float32))
    (output_dir/f'{tag}.json').write_text(json.dumps(result,indent=2))
    return result
