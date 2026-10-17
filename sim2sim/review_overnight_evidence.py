"""Read-only source audit, dated copies, and verbatim user-facing history export.

Never launches physics or changes the accepted experiment inputs.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import collections
import hashlib
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DOCS = ROOT.parent
OUT = ROOT / "history_and_backup_review_2026-09-11"
MU = DOCS / "Centipede_MUJOCO-main"
RL = DOCS / "Centipede_IsaacLab_RL"
LOG = Path(r"C:\Users\wxy22\.codex\sessions\2026\09\08\rollout-2026-09-08T10-38-17-01a08174-9a0d-7950-a0c8-8af5ec3f3214.jsonl")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


initial = json.loads((ROOT / "audit/static_inventory.json").read_text())
expected = initial["source_hashes"]
frozen = json.loads((ROOT / "COMMON_CONTACT_REPRODUCTION_MANIFEST.json").read_text())["required_sha256"]
source_check = [{"path": name, "initial_sha256": sha, "current_sha256": digest(Path(name))}
                for name, sha in expected.items()]
assert all(x["initial_sha256"] == x["current_sha256"] for x in source_check)
assert all(digest(ROOT / name) == sha for name, sha in frozen.items())

selected = {}


def add(path, category):
    path = path.resolve()
    assert path.is_file(), str(path)
    path.relative_to(DOCS)
    selected.setdefault(path, set()).add(category)


for name in expected:
    add(Path(name), "original_with_initial_audit_hash")

xml_path = MU / "experiments/mujoco_tip_passive_gliding_2026-09-01/models/centipede_flat_localz_k2em05_d3p2_r38.xml"
tree = ET.parse(xml_path).getroot()
compiler = tree.find("compiler").attrib
xml_dependencies = []
for element in tree.iter():
    if "file" not in element.attrib:
        continue
    file = Path(element.attrib["file"])
    directory = compiler.get("meshdir", "") if element.tag == "mesh" else compiler.get("texturedir", "") if element.tag == "texture" else ""
    path = file if file.is_absolute() else xml_path.parent / directory / file
    add(path, "original_mujoco_model_dependency_current_copy")
    xml_dependencies.append(str(path.resolve()))

for project in [MU, RL]:
    for path in project.glob("*.md"):
        add(path, "original_project_document_current_copy")
    for rel in ["configs", "docs"]:
        directory = project / rel
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    add(path, "original_config_or_document_current_copy")

historical = [
    MU / "experiments/common_pd_amplitude_contact_2026-09-09",
    MU / "experiments/mujoco_tip_passive_gliding_2026-09-01",
    MU / "experiments/mujoco_rigid_control_amplitude_2026-09-02",
    RL / "outputs/analysis/common_pd_amplitude_contact_20260909",
    RL / "outputs/analysis/mujoco_transfer_refinement_20260902",
]
for directory in historical:
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in {".md", ".py", ".json", ".yaml", ".yml", ".csv"}:
            add(path, "historical_experiment_code_parameters_document_current_copy")
    code = directory / "code"
    if code.exists():
        for path in code.rglob("*.py"):
            add(path, "historical_experiment_code_parameters_document_current_copy")
add(RL / "scripts/sanity/stage_a_vec_sweep_cell_compliance_2026-07-13.py", "historical_runtime_current_copy")

for name in frozen:
    add(ROOT / name, "accepted_branch_with_frozen_hash")
for path in ROOT.glob("*.md"):
    add(path, "alignment_document_current_copy")
for name in ["REPRODUCTION_MANIFEST.json", "COMMON_CONTACT_REPRODUCTION_MANIFEST.json", "audit/static_inventory.json", "audit/source_preservation.json", "audit/runtime_versions.json"]:
    add(ROOT / name, "alignment_audit_or_manifest_current_copy")
for directory in [MU / "experiments/sim2sim_alignment_2026-09-11", RL / "outputs/analysis/sim2sim_alignment_20260911"]:
    for path in directory.iterdir():
        if path.is_file():
            add(path, "new_opt_in_entrypoint_current_copy")

OUT.mkdir(exist_ok=False)
created = datetime.now(timezone.utc).isoformat()
rows = []
for source, categories in sorted(selected.items()):
    target = OUT / "backup" / source.relative_to(DOCS)
    target.parent.mkdir(parents=True, exist_ok=True)
    before = digest(source)
    shutil.copy2(source, target)
    assert digest(source) == before == digest(target), str(source)
    rows.append({"source": str(source), "copy": str(target.relative_to(OUT)), "bytes": target.stat().st_size,
                 "sha256": before, "categories": sorted(categories), "initial_audit_sha256": expected.get(str(source)),
                 "source_mtime_utc": datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat()})

messages = []
with LOG.open(encoding="utf-8") as stream:
    for line_no, line in enumerate(stream, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        stamp = record.get("timestamp", "")
        payload = record.get("payload", {})
        if not "2026-09-11T04:00" <= stamp < "2026-09-11T15:06":
            continue
        if record.get("type") != "response_item" or payload.get("type") != "message" or payload.get("role") not in {"user", "assistant"}:
            continue
        body = "\n".join(c.get("text", "") for c in payload.get("content", []) if c.get("type") in {"input_text", "output_text"})
        if not body.strip() or "<environment_context>" in body or "<recommended_plugins>" in body:
            continue
        eastern = datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone(timedelta(hours=-4)))
        messages.append({"timestamp_utc": stamp, "timestamp_edt": eastern.isoformat(), "role": payload["role"],
                         "phase": payload.get("phase"), "source_line": line_no, "text": body})

transcript = ["# 2026-09-11 凌晨至上午对话原文摘录", "",
              "从本任务本地 JSONL 记录按时间顺序提取用户消息和对用户显示的回复；不包含内部推理或工具输出。",
              "时间使用美东夏令时 EDT（UTC−04:00）。保留当时的中途判断；后续被否决的候选不能当作最终结果。",
              "这份摘录保存历史证据，不修复应用对话框显示。", "", f"来源：`{LOG}`", ""]
for message in messages:
    label = "用户" if message["role"] == "user" else "助手最终回复" if message["phase"] == "final_answer" else "助手进展"
    transcript += [f"## {message['timestamp_edt']} · {label}", "", f"原始文件第 {message['source_line']} 行。", "", message["text"].rstrip(), ""]
(OUT / "OVERNIGHT_CONVERSATION.zh-CN.md").write_text("\n".join(transcript), encoding="utf-8")
(OUT / "history_messages.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")

receipt = {"created_utc": created, "initial_audit_time": initial["time"], "original_sources_rechecked": len(source_check),
           "all_original_source_hashes_match": True, "accepted_frozen_files_rechecked": len(frozen),
           "all_accepted_frozen_hashes_match": True, "copied_files": len(rows), "copied_bytes": sum(x["bytes"] for x in rows),
           "all_copies_hash_verified": True, "history_messages": len(messages), "history_source": str(LOG),
           "original_source_verification": source_check, "xml_local_dependencies_copied": xml_dependencies,
           "scope": "Dated selective same-disk backup, not a complete project/environment/disk backup. Copies made now, not before the overnight work. Only the 41 initial-audit sources have an overnight original-state checksum claim. Other documents and dependencies are current copies. No simulator or optimizer launched.",
           "not_copied": ["complete raw runs, all videos and checkpoints", "complete Python/Isaac Sim/runtime installations", "Clean project", "full application conversation database"],
           "restore_note": "Original source copies retain bytes and original Documents-relative layout. The original MuJoCo XML still names its absolute mesh directory; restore its dependencies to original paths or explicitly adapt a working copy. The original USD geometry is embedded; OmniPBR.mdl is an external built-in rendering dependency, not bundled here.",
           "files": rows}
(OUT / "BACKUP_MANIFEST.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
readme = f"""# 本轮历史核对与补充备份

创建时间：{created}。这是一份**现在补做的选择性本地副本**，不是昨晚操作前创建的整项目备份，也不是异地备份。

- 原始输入重新校验：{len(source_check)}/{len(source_check)} 与凌晨审计一致。
- 最终对齐配置重新校验：{len(frozen)}/{len(frozen)} 与定版清单一致。
- 复制 {len(rows)} 个文件，共 {sum(x['bytes'] for x in rows)/1e6:.3f} MB；每份副本均与来源 SHA-256 一致。
- 已包括两份原始模型、MuJoCo XML 的 {len(xml_dependencies)} 个本地文件依赖、两个项目 configs/docs 中的文件、相关历史实验代码与文档、最终对齐模型与冻结配置。
- 只有原始清单中的 41 个文件有“与凌晨原版相同”的校验依据；其余文档与依赖标记为现在的副本，不能倒推它们昨晚之前的状态。
- 完整实验轨迹、视频、训练权重和整套运行环境未在此重复复制；它们仍依赖原保存位置。Clean 未复制或修改。

[逐文件来源、校验值与范围](BACKUP_MANIFEST.json)

[凌晨至上午用户可见对话原文](OVERNIGHT_CONVERSATION.zh-CN.md)

backup 下保留相对于 Documents 的原目录结构；没有自动覆盖或恢复操作。原 MuJoCo XML 使用绝对 meshdir，恢复时需连同依赖恢复到原路径，或在另一个工作副本中明确改路径。原 USD 的几何已嵌入；系统自带 OmniPBR.mdl 渲染材质未打包。
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
print(json.dumps({k: receipt[k] for k in ["created_utc", "copied_files", "copied_bytes", "history_messages", "original_sources_rechecked", "accepted_frozen_files_rechecked"]}, ensure_ascii=False))
print(str(OUT))
