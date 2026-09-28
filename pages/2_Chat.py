from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "2_Chat.py"))