"""Current finalizer: the centered layout supersedes the displaced-origin candidate."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent/'finalize_centered_common_contact.py'),run_name='__main__')
