from pathlib import Path
R=Path(__file__).resolve().parent
for device,source in [('cpu','render_common_contact_cpu_demo.py'),('gpu','render_common_contact_demo.py')]:
    text=(R/source).read_text(encoding='utf-8').replace('native_common_contact_validation_','native_common_contact_centered_validation_')
    old='common_contact_cpu_w2p5' if device=='cpu' else 'common_contact_w2p5'
    text=text.replace(old,f'common_contact_centered_{device}_w2p5').replace('Same plant, PD and shared compliant contact','Centered environments; same plant, PD and contact')
    (R/f'render_common_contact_centered_{device}_demo.py').write_text(text,encoding='utf-8')
