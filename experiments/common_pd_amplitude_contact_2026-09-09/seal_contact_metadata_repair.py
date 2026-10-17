"""Record a one-line output-only repair before any nonzero contact trial."""
from pathlib import Path
import datetime, hashlib, json

ROOT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'outputs/amendments/p0_metadata_20260909_0923'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__=='__main__':
    receipt_path=ARCHIVE/'repair_receipt.json'
    assert not receipt_path.exists(), 'Repair already sealed; inspect instead of repeating.'
    old_seal=read(ARCHIVE/'precontact_analysis_seal.json')
    current_path=ROOT/'outputs/data/precontact_analysis_seal.json'
    assert read(current_path)==old_seal
    changed=[]
    for source,digest in old_seal['source_hashes'].items():
        saved=ARCHIVE/Path(source).name
        assert sha(saved)==digest
        if sha(source)!=digest:changed.append(source)
    assert changed==[str(ROOT/'contact_erase.py')]
    old=(ARCHIVE/'contact_erase.py').read_text(encoding='utf-8')
    new=(ROOT/'contact_erase.py').read_text(encoding='utf-8')
    before="dict(**self.task['erasure'],kind='persistent_ground_only_bilateral_pair',"
    after="dict(**self.task['erasure'],"
    assert old.count(before)==1 and old.replace(before,after)==new
    failed=ARCHIVE/'failed_p0_contact_erase'
    rows=[read(p) for p in (failed/'assessments').glob('*.json')]
    assert len(rows)==14 and all(r['task']['erasure']['probability']==0 for r in rows)
    assert all('multiple values for keyword argument' in r['error'] for r in rows)
    assert not list((failed/'data/completed').glob('*.json'))
    assert not (ROOT/'outputs/contact_erase').exists()
    assert not list((ROOT/'outputs/contact_erase_transient/assessments').glob('*.json'))
    regression=read(ARCHIVE/'serialization_regression.json')
    assert regression['passed'] and regression['corrected_code_sha256'].lower()==sha(ROOT/'contact_erase.py')
    assert read(ROOT/'outputs/data/channel_static_check.json')['passed']
    receipt=dict(time=datetime.datetime.now().astimezone().isoformat(),
        reason='p0 result serialization duplicated the already-present kind keyword; all14 p0 runs failed at finish.',
        changed_source=changed[0], old_source_sha256=sha(ARCHIVE/'contact_erase.py'),
        new_source_sha256=sha(ROOT/'contact_erase.py'), exact_one_line_replacement=dict(before=before,after=after),
        archived_seal=str(ARCHIVE/'precontact_analysis_seal.json'), archived_seal_sha256=sha(ARCHIVE/'precontact_analysis_seal.json'),
        archived_failed_assessments=14, before_any_nonzero_contact_dynamics=True,
        changed_physics=False, changed_pd=False, changed_selection=False, changed_random_schedule=False,
        changed_statistics=False, regression_sha256=sha(ARCHIVE/'serialization_regression.json'),
        new_static_check_sha256=sha(ROOT/'outputs/data/channel_static_check.json'),
        archive_file_hashes={str(p):sha(p) for p in ARCHIVE.rglob('*') if p.is_file()})
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    revised=dict(old_seal)
    revised.update(time=receipt['time'],before_any_contact_dynamics=False,
        before_any_nonzero_contact_dynamics=True,
        amendment_receipt=str(receipt_path), amendment_sha256=sha(receipt_path),
        reporting_update='Original analysis/statistics unchanged. One-line persistent-result metadata repair after archived p0 failure, before any nonzero contact trial.')
    revised['source_hashes']={p:sha(p) for p in old_seal['source_hashes']}
    current_path.write_text(json.dumps(revised,indent=2),encoding='utf-8')
    print(json.dumps(dict(repair_sealed=True,receipt=str(receipt_path),changed_sources=changed)))
