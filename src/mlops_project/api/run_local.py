import os
import subprocess
from pathlib import Path

os.environ.setdefault("MODEL_BUCKET", "mlops-project-models")
os.environ.setdefault("MODEL_BLOB", "modelweights.pkl")
os.environ.setdefault("FEATURE_COUNT", "7")

# Get the directory where this script is located
script_dir = Path(__file__).parent
main_py = script_dir / "main.py"

cmd = [
    "functions-framework",
    "--target", "logreg_classifier",
    "--source", str(main_py),
    "--debug",
    "--port", "8080",
]
print(main_py)
subprocess.run(cmd, check=True)

