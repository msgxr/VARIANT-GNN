from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parent / "scripts" / "notebook" / "build_notebook.py"
runpy.run_path(str(TARGET), run_name="__main__")
