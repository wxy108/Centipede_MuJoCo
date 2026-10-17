from pathlib import Path
R=Path(__file__).resolve().parent
for source,target in [('evaluate_common_contact.py','evaluate_common_contact_centered.py'),('analyze_common_contact_motion.py','analyze_common_contact_centered_motion.py')]:
    text=(R/source).read_text(encoding='utf-8')
    text=text.replace('native_common_contact_validation_', 'native_common_contact_centered_validation_')
    text=text.replace('common_contact_summary.json','common_contact_centered_summary.json').replace('common_contact_validation_table.csv','common_contact_centered_validation_table.csv').replace('common_contact_motion_comparison','common_contact_centered_motion_comparison')
    (R/target).write_text(text,encoding='utf-8')
