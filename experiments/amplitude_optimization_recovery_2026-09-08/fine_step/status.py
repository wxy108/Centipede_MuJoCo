"""Compact progress from task banks and actual atomic assessment receipts."""
import collections,datetime,json
import core as c

def snapshot():
    rows=c.records();by_id={r['id']:r for r in rows};nominal=c.nominal()
    validation=c.read(c.DATA/'validation_index.json') if (c.DATA/'validation_index.json').exists() else {}
    waves=[]
    for w in c.WAVES:
        rr=[r for r in nominal if r['wave']==w];good=[r for r in rr if r['feasible']]
        vv=[v for v in validation.values() if v.get('qualified') and v['nominal']['wave']==w]
        waves.append(dict(wave=w,n=len(rr),full=sum(r['full'] for r in rr),feasible=len(good),
            best_nominal=max([r['D'] for r in good],default=None),best_qualified=max([v['nominal']['D'] for v in vv],default=None),
            failed=collections.Counter(k for r in rr for k in r['failed'])))
    stages=[]
    for path in sorted((c.DATA/'banks').glob('*.json')):
        if '_sources_' in path.name:continue
        bank=c.read(path)
        if not isinstance(bank,list):continue
        complete=(c.DATA/'completed'/path.name).exists()
        stages.append(dict(stage=path.stem,assessments=sum(c.identity(t) in by_id for t in bank),planned=len(bank),completed_receipt=complete))
    return dict(time=datetime.datetime.now().astimezone().isoformat(),stages=stages,waves=waves,
        execution_errors=[dict(id=r['id'],error=r['error']) for r in rows if 'error' in r])

if __name__=='__main__':print(json.dumps(snapshot(),indent=2))
