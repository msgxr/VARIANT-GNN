import os
import re

MAPPING = {
    r'from src\.models': r'from src.core.models',
    r'import src\.models': r'import src.core.models',
    r'from src\.graph': r'from src.core.graph',
    r'import src\.graph': r'import src.core.graph',
    r'from src\.inference': r'from src.api',
    r'import src\.inference': r'import src.api',
    r'from src\.ui': r'from src.interface.dashboard',
    r'import src\.ui': r'import src.interface.dashboard',
    r'from src\.explainability': r'from src.scientific.xai',
    r'import src\.explainability': r'import src.scientific.xai',
    r'from src\.evaluation': r'from src.scientific.metrics',
    r'import src\.evaluation': r'import src.scientific.metrics',
    r'from src\.calibration': r'from src.scientific.calibration',
    r'import src\.calibration': r'import src.scientific.calibration',
    r'from src.data.schemas': r'from src.data.schemas',
    r'import src.data.schemas': r'import src.data.schemas',
}

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content
    for pattern, replacement in MAPPING.items():
        new_content = re.sub(pattern, replacement, new_content)

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {file_path}")

def main():
    root = os.getcwd()
    for dirpath, dirnames, filenames in os.walk(root):
        if any(ignored in dirpath for ignored in ['.git', '.venv', '__pycache__', '.pytest_cache']):
            continue
        for file in filenames:
            if file.endswith('.py'):
                process_file(os.path.join(dirpath, file))

if __name__ == '__main__':
    main()
