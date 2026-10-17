# Rigid-tip ablation

## 设计

移除 38 个 distal tip-roll 关节、38 个 tip body 和 38 个 foot body；distal capsule、foot geom 和 foot site 全部直接归入 leg-pitch body，是真正的一体 leg，而不是 fixed-child 近似。没有恢复 knee。七组随机关节姿态下，全部 geom 世界位置和足端位置最大误差为 1.39e-17 m；投影到保留自由度的质量矩阵最大误差为 6.78e-21。总质量、腿长、惯性、接触参数、控制器、PD 和幅值均不变。

## 成对结果

| wave | topology | D mm | Δω 10ms rad/s | pitch/roll speed rad/s | chassis force BW | accel P99 rad/s² | head+link1 force BW | slip P95 mm/s | gates |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2.5 | compliant tip | 149.962 | 8.628 | 5.616/2.566 | 0.420 | 826.462 | 0.000 | 32.312 | reference pass |
| 2.5 | rigid tip | 145.006 | 12.265 | 6.845/3.628 | 0.474 | 1592.478 | 0.000 | 39.043 | body_yaw_speed, body_velocity_jump, leg_yaw_speed, leg_pitch_speed, swing_height |
| 3 | compliant tip | 129.274 | 5.353 | 3.955/2.551 | 0.000 | 474.237 | 0.000 | 21.459 | reference pass |
| 3 | rigid tip | 129.333 | 8.400 | 7.108/3.681 | 0.000 | 959.747 | 0.000 | 23.688 | pitch_10ms, active_speed, leg_yaw_speed, leg_pitch_speed |

## 半时间步复验（0.125 ms）

| wave | topology | D mm | Δω 10ms rad/s | accel P99 rad/s² | failed gates |
|---:|---|---:|---:|---:|---|
| 2.5 | compliant tip | 149.737 | 4.898 | 556.993 | reference pass |
| 2.5 | rigid tip | 143.172 | 16.868 | 1222.976 | body_yaw_speed, body_velocity_jump, leg_yaw_speed, swing_height |
| 3 | compliant tip | 129.221 | 5.804 | 396.182 | reference pass |
| 3 | rigid tip | 126.870 | 7.807 | 645.929 | leg_yaw_speed |

## 判定

机器判定：`does_not_support_tip_as_major_twitch_source`。支持票 0，反对票 21。支持/反对票只来自预先列出的 body jump、pitch/roll speed、chassis force、加速度、jerk 与 head/link1 contact 指标，避免只凭视频印象。

原 tip-angle/tip-speed 四个门槛在 rigid topology 中没有对应自由度，因此明确标为 N/A，没有伪造为 pass。

## 工程结论

- wave=2.5：10 ms body 速度跳变由 8.628 增至 12.265 rad/s，body-joint acceleration P99 由 826.5 增至 1592.5 rad/s²；位移反而由 149.962 降至 145.006 mm。
- wave=3：10 ms body 速度跳变由 5.353 增至 8.400 rad/s，pitch/roll speed 由 3.955/2.551 增至 7.108/3.681 rad/s。
- 0.125 ms 半时间步没有救回结果：rigid wave=2.5 仍失败四项门槛，rigid wave=3 仍失败 leg-yaw speed；因此不是默认 0.25 ms 步长制造的假触动。
- 两组 head+link1 ground-contact peak 都是 0 BW，排除了“头部偶发撞地触发这些跳变”。

结论不是“tip 没有任何影响”，而是相反：当前 compliant tip 在这个模型里承担了机械低通/接触缓冲作用。将它完全刚性化会把足端接触高频直接传到 leg 和 body，触动更严重。当前 compliant-tip 模型应继续作为基线；一体刚性腿 XML 只保留为诊断变体，不升级为默认模型。

## Demo

- `outputs/videos/rigid_tip_w2p5_three_views.mp4`
- `outputs/videos/rigid_tip_w2p5_local.mp4`
- `outputs/videos/rigid_tip_w3p0_three_views.mp4`
- `outputs/videos/rigid_tip_w3p0_local.mp4`
