import importlib.util
import inspect
import sys
import traceback
from pathlib import Path


def main():
    tests_dir = Path(__file__).resolve().parent
    passed = 0
    failed = 0
    for test_file in sorted(tests_dir.glob("test_*.py")):
        module = _load_module(test_file)
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            try:
                func()
            except Exception:
                failed += 1
                print(f"FAIL {test_file.name}::{name}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"PASS {test_file.name}::{name}")
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


def _load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
