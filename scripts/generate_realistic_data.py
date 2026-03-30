from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parent / "data_generation" / "generate_realistic_data.py"
runpy.run_path(str(TARGET), run_name="__main__")
