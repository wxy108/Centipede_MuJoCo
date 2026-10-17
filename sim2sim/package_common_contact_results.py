"""Collect reviewed current artifacts only after complete acceptance."""
import json,shutil
import shared as s
R=s.ROOT
profile=json.loads((R/'COMMON_CONTACT_OPERATING_PROFILE.json').read_text())
assert profile['status']=='validated_common_contact_cpu_gpu'
out=R/'presentation_ready_common_contact';out.mkdir(exist_ok=False)
files=['REPORT.md','RUN_GUIDE.md','common_contact_centered_validation_table.csv','common_contact_centered_motion_comparison.csv','COMMON_CONTACT_OPERATING_PROFILE.json','COMMON_CONTACT_VALIDATION_RECEIPT.json','COMMON_CONTACT_REPRODUCTION_MANIFEST.json','COMMON_CONTACT_V2.json','audit/common_contact_centered_motion_gates.json']
files += [f'figures/{n}.{ext}' for n in ['23_common_contact_centered_full_bank','24_common_contact_centered_acceptance_checks','25_common_contact_centered_difference_detail','26_common_contact_centered_motion_checks','22_common_contact_coordinate_precision'] for ext in ['png','pdf','svg']]
files += [f'demos/common_contact_centered_{device}_w2p5_B_closeup_{suffix}' for device in ['cpu','gpu'] for suffix in ['replay.mp4','replay.json','preview.png']]
records=[]
for relative in files:
    dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/relative,dest);assert s.sha(dest)==s.sha(R/relative);records.append(dict(path=relative,sha256=s.sha(dest)))
(out/'README.md').write_text('''# 当前对齐结果展示包

这是共用柔顺接触模型的已验收平地机械基准，不是原生刚性接触算法等价的声明。

- 图 23：相同坐标下八组步态与三种后端、两档步长。
- 图 24：原定验收门槛；图 25：放大后的实际差异。
- 图 26：额外的完整 XY 轨迹和方向验收。
- 图 22：场景位置导致的坐标精度限制；应与性能图一起保留。
- demos：CPU / GPU 各一段与 MuJoCo 的仿真轨迹重建回放，1280×800、25 fps。使用保存的关节角与身体位置拟合根位姿；不是 Isaac 原生渲染，也没有重新求解物理。

PNG 可直接插入演示文稿；PDF / SVG 为独立科学图。没有用 PowerPoint 叠加文字。所有图内文字、图例和坐标已包含在图像中。

REPORT.md 说明原始差异、共用参数、失败候选和适用范围。幅值固定，未做新优化；原 RL 训练、contact erasure 和 Clean edition 均未迁移。完整实验、原始日志和复现入口位于上一级实验目录。
''',encoding='utf-8')
s.write(out/'MANIFEST.json',dict(status=profile['status'],files=records,source_directory=str(R)))
print(str(out))
