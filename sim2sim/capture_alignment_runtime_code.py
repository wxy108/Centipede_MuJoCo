"""Current runtime code fingerprints for reproduction, not a retroactive audit."""
import importlib.util,json
from pathlib import Path
from datetime import datetime,timezone
import shared as s
R=s.ROOT;origin=Path(importlib.util.find_spec('isaaclab').origin).resolve();base=origin.parent
relative=['sim/simulation_context.py','sim/simulation_cfg.py','actuators/actuator_pd.py','actuators/actuator_base.py','app/app_launcher.py','sim/spawners/from_files/from_files_cfg.py','sim/spawners/from_files/from_files.py','sim/schemas/schemas.py','assets/asset_base.py','assets/articulation/articulation.py','assets/articulation/articulation_data.py']
modules=json.loads((R/'audit/loaded_physics_modules.json').read_text(encoding='utf-8-sig'));paths=[origin]+[base/n for n in relative]+[Path(m['FileName']) for m in modules]
rows=[dict(path=str(p),sha256=s.sha(p),bytes=p.stat().st_size) for p in paths]
s.write(R/'audit/runtime_code_snapshot.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),isaaclab_origin=str(origin),rows=rows,qualification='Captured during centered validation for exact future code checks. Loaded native physics module paths came from the owned centered GPU main worker. This is not a claim that these hashes were captured before the original audit.'))
print(json.dumps(dict(files=len(rows),total_bytes=sum(r['bytes'] for r in rows),isaaclab_origin=str(origin))))
