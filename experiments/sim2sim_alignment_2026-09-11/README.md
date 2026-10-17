# 2026-09-11 共用物理参数基准

这里是 MuJoCo 的共用柔顺接触对齐入口。共用模型、物理 PD、控制时钟与验证记录位于 `C:\Users\wxy22\Documents\Centipede_Sim2Sim_2026-09-11`。当前进展请查看该目录的 `REPORT.md`、`RUN_GUIDE.md` 和通过后才生成的 `COMMON_CONTACT_OPERATING_PROFILE.json`。旧 `OPERATING_PROFILE.json` 属于已拒绝的原生接触候选。

`run_aligned.py --dry-run` 可查看将要运行的固定基准。正式入口只有在整组验收完成、配置标记为已验证后才启动模拟；它会核对冻结输入的校验值，并创建独立输出目录。

运行时使用原环境 `C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe`。输出位于共用目录的 `reproductions` 和 `runs` 中。MuJoCo 的接触求解与速度积分共享由物理 PD 推导的有效惯量；数值系数与物理 rotor inertia 在记录中明确分开。

历史优化脚本、模型和结果保持原样。此入口用于共用物理定义的开放环路基准。

两边原生地面接触响应关闭，由同一个固定柔顺接触力律施加外力。这是明确的新物理接触模型，并非原生刚性接触算法已等价。报告单独记录坐标位置导致的横向残差、变形深度和计算耗时。
