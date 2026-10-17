from pathlib import Path
import json,hashlib
import shared as s
audit=json.loads((s.ROOT/'audit/static_inventory.json').read_text());rows=[]
for name,expected in audit['source_hashes'].items():
    p=Path(name);actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    rows.append(dict(path=name,preserved=actual==expected,initial_sha256=expected,current_sha256=actual))
s.write(s.ROOT/'audit/source_preservation.json',dict(rows=rows,all_preserved=all(r['preserved'] for r in rows)));print(json.dumps(dict(files=len(rows),all_preserved=all(r['preserved'] for r in rows),differences=[r['path'] for r in rows if not r['preserved']])))
