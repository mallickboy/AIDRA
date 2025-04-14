import os
from pathlib import Path  # Correct module

folder_str = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    "requirements.txt",       # fixed typo
    "setup.py",
    "app.py",
    "playground/test.ipynb"
]

for filepath in folder_str:
    filepath = Path(filepath)
    if filepath.parent != Path("."):
        os.makedirs(filepath.parent, exist_ok=True)
    if not filepath.exists():
        filepath.touch()
