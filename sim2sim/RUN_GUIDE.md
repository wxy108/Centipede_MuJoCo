# 复现共用物理基准

这是单独的日期化入口，当前验收状态以 `REPORT.md` 和 `COMMON_CONTACT_OPERATING_PROFILE.json` 为准。验收文件尚未生成时，正式运行会停止；`--dry-run` 仅显示将使用的配置，不启动物理。

Windows PowerShell 示例：

```powershell
& 'C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe' -B 'C:\Users\wxy22\Documents\Centipede_Sim2Sim_2026-09-11\reproduce_common_contact.py' --engine mujoco --dry-run
& 'C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe' -B 'C:\Users\wxy22\Documents\Centipede_Sim2Sim_2026-09-11\reproduce_common_contact.py' --engine isaac-cpu --dry-run
& 'C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe' -B 'C:\Users\wxy22\Documents\Centipede_Sim2Sim_2026-09-11\reproduce_common_contact.py' --engine isaac-gpu --dry-run
```

验收通过后移除 `--dry-run` 即执行固定测试。每次自动创建唯一结果目录；也可用 `--name` 指定一个未使用的简单目录名。结果在 `reproductions/<名称>/results.json`，逐帧轨迹、回读参数及日志保存在对应运行目录。不要并发启动多个 GPU 物理进程。完整九环境运行较慢，实测耗时见报告。

两个原项目的便捷入口分别是：

- `Centipede_MUJOCO-main/experiments/sim2sim_alignment_2026-09-11/run_aligned.py`：MuJoCo。
- `Centipede_IsaacLab_RL/outputs/analysis/sim2sim_alignment_20260911/run_aligned.py`：原 Isaac Lab GPU；也可追加 `--engine isaac-cpu` 使用 CPU。

共同设置：四个 wave number，每个两组固定幅值，另有一组校准步态；251 个刚体、173 个转动关节；同一套物理 PD、重力和共用柔顺接触参数；主步长 31.25 µs、指令 250 µs、运行 6 s、评分窗口 2–6 s。独立环境都在相同数值原点，各自的关节链和接触力互不耦合，不做运行中的位置复位。没有按后端拟合物理量。

必须保留的边界：原生地面接触响应关闭，接触力由共用函数施加；原生接触传感器不能直接表示这套外加力。尚未迁移原 RL 奖励、观测、重置、terrain 或训练。旧 bank 的大坐标偏移会产生已记录的横向误差；当前原点布局额外检查 XY 路径和方向，但不等于任意世界坐标或任意长时间行进都已经验证。

原项目默认入口仍执行原来的物理设置。Clean edition 未改动。旧 `reproduce_alignment.py` 继续拒绝启动旧 PGS 失败候选。

复现前自动核对参数、代码和运行库校验和；GPU 入口还核对本次实测显卡与驱动版本。当前硬件记录在 `audit/hardware_identity.json`。更换设备、驱动或运行库后，应重新验证数值结果，不能直接继承本次验收状态。

配置含义按层区分：`COMMON_CONTRACT.json` 保存共用机器人、物理 PD、指令及最初验收定义；`COMMON_CONTACT_V2.json` 明确替换接触物理；通过后的 `COMMON_CONTACT_OPERATING_PROFILE.json` 指定实际步长、求解器、布局和入口。最初 contract 的 `candidate_numerics` 是历史起点，不是当前运行配置。原生材质与 Mu 结果中保留的旧 `contact_time_constant_s` 字段属于被关闭的原生接触响应，不改变实际固定的共用刚度和阻尼。回弹由新接触力律决定，不能把旧 restitution 字段当作其有效恢复系数。
