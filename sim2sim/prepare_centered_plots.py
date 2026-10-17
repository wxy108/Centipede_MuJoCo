from pathlib import Path
R=Path(__file__).resolve().parent
for source,target in [('plot_common_contact_results.py','plot_common_contact_centered_results.py'),('plot_common_contact_difference_detail.py','plot_common_contact_centered_difference_detail.py')]:
    text=(R/source).read_text(encoding='utf-8').replace('common_contact_summary.json','common_contact_centered_summary.json')
    text=text.replace('19_common_contact_full_bank','23_common_contact_centered_full_bank').replace('20_common_contact_acceptance_checks','24_common_contact_centered_acceptance_checks').replace('21_common_contact_difference_detail','25_common_contact_centered_difference_detail')
    text=text.replace('Shared compliant contact','Centered environments; shared compliant contact').replace('Shared compliant-contact acceptance checks','Centered common-contact forward acceptance checks')
    (R/target).write_text(text,encoding='utf-8')
