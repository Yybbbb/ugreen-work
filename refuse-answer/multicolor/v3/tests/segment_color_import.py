import sys
from pathlib import Path


def add_repo_path():
    v3_dir = Path(__file__).resolve().parents[1]
    v2_dir = v3_dir.parent / "v2"
    for path in [v3_dir, v2_dir]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
