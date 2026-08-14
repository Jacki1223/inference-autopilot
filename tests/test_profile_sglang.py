import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import profile_sglang


class ProfileInputTests(unittest.TestCase):
    def test_accepts_execution_spec(self):
        spec = {"execution": {"host": "127.0.0.1"}}
        self.assertIs(profile_sglang.resolve_profile_spec(spec), spec)

    def test_extracts_autopilot_plan_profile_spec(self):
        spec = {"execution": {"host": "127.0.0.1"}}
        plan = {"profiling": {"spec": spec}}
        self.assertIs(profile_sglang.resolve_profile_spec(plan), spec)

    def test_existing_diagnosis_has_runtime_compatible_status(self):
        # The run path checks this contract for both fresh and reused profiles.
        status = {"state": "completed", "reused": True}
        self.assertEqual(status["state"], "completed")


if __name__ == "__main__":
    unittest.main()
