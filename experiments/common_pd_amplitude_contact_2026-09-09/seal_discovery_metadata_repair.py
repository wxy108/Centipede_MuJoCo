"""Seal an output-only analysis repair after discovery, before held-out trials."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'outputs/amendments/discovery_metadata_20260909_1007'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

if __name__=='__main__':
    receipt_path=ARCHIVE/'repair_receipt.json';assert not receipt_path.exists()
    seal_path=ROOT/'outputs/data/precontact_analysis_seal.json'
    previous=read(ARCHIVE/'precontact_analysis_seal.json');assert read(seal_path)==previous
    changed=[]
    for p,h in previous['source_hashes'].items():
        assert sha(ARCHIVE/Path(p).name)==h
        if sha(p)!=h:changed.append(p)
    assert changed==[str(ROOT/'analyze_contact.py')]
    original=(ARCHIVE/'analyze_contact.py').read_text(encoding='utf-8')
    corrected=(ROOT/'analyze_contact.py').read_text(encoding='utf-8')
    before="record=dict(**choice,confirmation_seeds=";after="record=dict(choice,confirmation_seeds="
    assert original.count(before)==1 and original.replace(before,after)==corrected
    old_result=read(ARCHIVE/'discovery_analysis.json')
    new_result=read(ROOT/'outputs/contact_erase/data/discovery_analysis.json')
    old_result.pop('code_sha256');new_result.pop('code_sha256');assert old_result==new_result
    selection=old_result['selected_for_confirmation'];frozen=read(ROOT/'outputs/contact_erase/data/crossing_choice.json')
    assert all(frozen[k]==v for k,v in selection.items() if k!='interpretation')
    rows=[read(p) for p in (ROOT/'outputs/contact_erase/assessments').glob('*.json')]
    assert len(rows)==252 and all(r['full'] and r['task']['erasure']['seed'] in range(4) for r in rows)
    assert not (ROOT/'outputs/contact_erase/data/confirmation_declaration.json').exists()
    assert not list((ROOT/'outputs/contact_erase_transient/assessments').glob('*.json'))
    manifest=read(ROOT/'outputs/contact_erase/data/manifest.json')
    for p,h in manifest['frozen_inputs'].items():assert sha(p)==h.lower(),p
    assert read(ARCHIVE/'serialization_regression.json')['passed']
    inputs={}
    for r in rows:
        record=Path(r['record']);assessment=ROOT/'outputs/contact_erase/assessments'/f"{r['id']}.json"
        for p in (assessment,record,record.parent/'contact_trajectory.npz'):inputs[str(p)]=sha(p)
    receipt=dict(time=datetime.datetime.now().astimezone().isoformat(),
        reason='Discovery selection already existed inside the saved analysis; freezing its record duplicated the interpretation keyword and failed.',
        changed_source=changed[0],old_source_sha256=sha(ARCHIVE/'analyze_contact.py'),new_source_sha256=sha(ROOT/'analyze_contact.py'),
        exact_replacement=dict(before=before,after=after),before_any_nonzero_contact_dynamics=False,
        before_any_heldout_contact_dynamics=True,changed_physics=False,changed_pd=False,changed_selection=False,
        changed_random_schedule=False,changed_statistics=False,all_discovery_statistics_and_candidate_order_identical=True,
        choice=dict(low_wave=frozen['low_wave'],high_wave=frozen['high_wave'],bracket=frozen['bracket']),
        unchanged_discovery_input_hashes=inputs,
        archive_file_hashes={str(p):sha(p) for p in ARCHIVE.rglob('*') if p.is_file()})
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    revised=dict(previous);revised.update(time=receipt['time'],before_any_nonzero_contact_dynamics=False,
        before_any_heldout_contact_dynamics=True,
        reporting_update='Second output-only repair: allow the intended interpretation text update when freezing the already-computed discovery choice. Statistics and selection identical; no held-out trials yet.')
    revised['additional_amendments']=[*previous.get('additional_amendments',[]),dict(receipt=str(receipt_path),sha256=sha(receipt_path))]
    revised['source_hashes']={p:sha(p) for p in previous['source_hashes']}
    seal_path.write_text(json.dumps(revised,indent=2),encoding='utf-8')
    print(json.dumps(dict(repair_sealed=True,identical_discovery_statistics=True,choice=receipt['choice'],raw_conditions=252)))
