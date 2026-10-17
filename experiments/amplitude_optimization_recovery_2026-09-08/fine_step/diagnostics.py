"""Choose explicit complete failed cases; never promote diagnostic scores."""
from feasibility import violation

def failed_cases(rows):
    full=[r for r in rows if r['full'] and not r['feasible'] and violation(r) is not None]
    if not full:return []
    candidates=[('least_violation','LEAST OBSERVED CONSTRAINT VIOLATION',min(full,key=lambda r:(violation(r),-r['D']))),
        ('fastest_failed','FASTEST COMPLETE FAILED CASE',max(full,key=lambda r:r['D']))]
    result=[];seen=set()
    for kind,label,row in candidates:
        if row['id'] in seen:continue
        seen.add(row['id'])
        result.append(dict(row,diagnostic_kind=kind,diagnostic_label=label,diagnostic_max_ratio=violation(row)))
    return result
