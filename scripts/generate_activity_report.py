from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parent / "reporting" / "generate_activity_report.py"
runpy.run_path(str(TARGET), run_name="__main__")
