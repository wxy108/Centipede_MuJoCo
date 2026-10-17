from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'benchmark_desaxce_warm_robot.py').read_text(encoding='utf-8')
src=src.replace('from desaxce_contact import step as cold','from desaxce_contact_anderson import step as cold').replace('from desaxce_contact_warm import step as warm','from desaxce_contact_anderson import step as warm')
src=src.replace("[('cold',cold),('warm',warm)]","[('anderson60',lambda m,d:cold(m,d,60)),('anderson120',lambda m,d:warm(m,d,120))]")
src=src.replace('desaxce_warm_robot_benchmark.json','desaxce_anderson_robot_benchmark.json')
(R/'benchmark_desaxce_anderson_robot.py').write_text(src,encoding='utf-8')
