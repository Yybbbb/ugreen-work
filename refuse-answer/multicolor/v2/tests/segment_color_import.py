import sys
from pathlib import Path


def add_repo_path():
    v2_dir = Path(__file__).resolve().parents[1]
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))
