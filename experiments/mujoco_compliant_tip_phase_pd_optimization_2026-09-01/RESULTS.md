# Compliant-tip phase + PD optimization

结论：柔性 tip 始终保留且其被动 k/c 未改变。body-yaw 与 leg cluster 的空间采样已拆开，body/leg wave number 仍完全一致，控制器没有加入反馈、滤波、限幅或特例稳定器。两组非零相位方案都在 6 s nominal 下通过全部登记门槛，但没有同时通过 12 s 与半时间步复核，因此不覆盖上一版已认证 profile。

## Phase definition

`theta_leg[i] = theta_body[i] + f*(2*pi*wave/18)`；几何中点为 `f=0.5`。本轮 nominal winners 分别为 wave=2.5: `f=0.4853` (24.27 deg)，wave=3: `f=0.35` (21 deg)。

## Nominal profiles and validation

| wave | phase offset | 6 s nominal | 12 s | 6 s half-dt | decision |
|---:|---:|---:|---:|---:|---|
| 2.5 | 24.27 deg | 140.2 mm, PASS | 325.6 mm, PASS | 139.9 mm, FAIL: relative_lr_force_imbalance | nominal diagnostic only |
| 3 | 21 deg | 123.7 mm, PASS | 287 mm, FAIL: active_speed, tip_speed, tip_peak, relative_lr_force_imbalance | 123.8 mm, FAIL: relative_lr_force_imbalance | nominal diagnostic only |

## Exact impedance values

| wave | body yaw Kp/Kd | body pitch Kp/Kd | leg yaw Kp/Kd | leg pitch Kp/Kd | passive body roll k/c | tip k/c |
|---:|---|---|---|---|---|---|
| 2.5 | 0.000186754/1.5312e-06 | 0.000933857/4.03383e-06 | 9.29627e-05/4.13005e-07 | 0.000102835/1.39795e-07 | 0.000800909/5.71797e-06 | 4.5e-06/1.15376e-09 |
| 3 | 0.000453033/4.41626e-06 | 0.000262805/3.66097e-06 | 0.00012894/1.88472e-06 | 0.000476/7.60253e-08 | 0.000695843/1.40802e-05 | 4.5e-06/1.15376e-09 |

## Decision

- 默认仍是上一版柔性-tip、共享空间相位但已通过 nominal/12 s/half-dt 的两个 wave-specific PD profiles。
- 本轮证明：几何相位拆分不是只改一行 phase 就能免费获得的修正；它会改变足端冲击和左右载荷，需要新的 robustness 标定。
- rigid-tip 搜索已明确中止并标记为不得使用；Robot 目录与 Step 18/19 未写入。
- Demo 使用 1 kHz 记录的真实物理姿态、无平滑、contact erasure OFF；画面明确标注新方案仅 nominal pass。
