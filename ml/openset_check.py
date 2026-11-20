"""Run the original and open-set contract tests and save their bounded summary."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module_dir = Path(__file__).resolve().parent
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([loader.discover(str(module_dir), pattern="test_*.py"),
                                loader.discover(str(module_dir), pattern="openset_test.py")])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"tests_run": result.testsRun, "passed": result.wasSuccessful(),
              "failure_count": len(result.failures), "error_count": len(result.errors), "skipped_count": len(result.skipped),
              "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
              "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors],
              "scope": "20 original ML contracts plus10 open-set gate/prototype/selection tests; synthetic test inputs are not recognition accuracy evidence",
              "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in module_dir.glob("*.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
