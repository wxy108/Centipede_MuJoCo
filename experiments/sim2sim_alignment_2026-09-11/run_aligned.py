"""Opt-in shared compliant-contact benchmark; guarded until validation passes."""
from pathlib import Path
import subprocess,sys
study=Path(r'C:\Users\wxy22\Documents\Centipede_Sim2Sim_2026-09-11')
runtime=Path(r'C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe')
raise SystemExit(subprocess.call([str(runtime),'-B',str(study/'reproduce_common_contact.py'),'--engine','mujoco',*sys.argv[1:]]))
