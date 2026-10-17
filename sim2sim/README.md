# Centipede MuJoCo / Isaac Lab 物理对齐

先读 [REPORT.md](REPORT.md)：实际结果、通过与失败项目、根因证据及使用限制都在这里。初始审计是 [AUDIT.md](AUDIT.md)，决策记录是 [DECISION_LEDGER.md](DECISION_LEDGER.md)。

现已完成同一机器人、同一物理 PD 和同一柔顺接触力律的开环平地基准验收：八组步态、三种后端、两档步长全部完成；72 组运动比较全部通过。两档步长的所有后端成对比较中，最大前进位移差为0.048325%。旧原生接触 PGS / TGS 候选未通过完整验收。旧优化、contact erasure、RL 训练及 Clean edition 没有恢复或迁移。

- 当前完整数字：[common_contact_centered_validation_table.csv](common_contact_centered_validation_table.csv)。
- 图 23：八组性能与两档步长；图 24：完整门槛；图 25：放大后的真实差异；图 26：完整运动验收。
- 图 22：场景坐标位置对微小机器人积分的影响。
- `demos/common_contact_*`：保存位姿的对照回放；未重新计算物理，不是 Isaac 原生渲染。
- 复现方法：[RUN_GUIDE.md](RUN_GUIDE.md)。

当前采用版本把独立环境放在相同数值原点，以修复远处坐标下的微小位移损失。所有原生碰撞保持关闭，外加接触力按环境独立计算。第一轮偏移布局的图 19–21 和数据仍保留，未拿靠近原点的较好结果覆盖它们。

`COMMON_CONTACT_OPERATING_PROFILE.json` 只在全部结果和质量核查通过后创建。存在该文件时，还必须通过 `COMMON_CONTACT_REPRODUCTION_MANIFEST.json` 的文件校验。`COMMON_CONTACT_CANDIDATE_PROFILE.json` 和 `COMMON_CONTACT_V2.json` 保留创建时的冻结状态；它们自身不代表已通过验证。

旧 `OPERATING_PROFILE.json`、`REPRODUCTION_MANIFEST.json` 和 `reproduce_alignment.py` 属于被拒绝的 PGS 候选，保留作溯源。当前复现使用 `reproduce_common_contact.py`。
