import os
import sys
import subprocess
from pathlib import Path
from importlib.resources import files

def open_notebook_folder():
    """
    Opens the folder containing the specified notebook in the system's file explorer.
    """
    # notebook_path = files("image_to_image.model_interactions") / "eval_physgen_benchmark.ipynb"
    # notebook_path = "../model_interactions"  # /eval_physgen_benchmark.ipynb"
    notebook_path = Path(__file__).parent / "../model_interactions"
    notebook_path = notebook_path.resolve()

    if not notebook_path.exists():
        print(f"Folder not found: {notebook_path}")
        return

    print(f"Opening folder: {notebook_path}, there is the 'eval_physgen_benchmark.ipynb' notebook for running.")

    if sys.platform == "win32":
        os.startfile(notebook_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(notebook_path)])
    else:  # Linux and others
        subprocess.run(["xdg-open", str(notebook_path)])


