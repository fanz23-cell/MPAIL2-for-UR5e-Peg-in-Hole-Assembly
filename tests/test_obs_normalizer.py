import unittest
import importlib.util
from pathlib import Path

import torch


_SPEC = importlib.util.spec_from_file_location(
    "obs_normalizer_under_test",
    Path(__file__).resolve().parents[1] / "mpail2" / "utils" / "obs_normalizer.py",
)
_NORMALIZER_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NORMALIZER_MODULE)
FixedObsNormalizer = _NORMALIZER_MODULE.FixedObsNormalizer


class FixedObsNormalizerTest(unittest.TestCase):
    def test_std_floor_limits_low_variance_dimensions(self):
        demos = {
            "policy": torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 2.0]],
                    [[1.0, 4.0], [1.0, 6.0]],
                ]
            )
        }

        normalizer = FixedObsNormalizer(demos, device="cpu", std_floor=0.05, clip_obs=None)

        self.assertGreaterEqual(normalizer.policy_std.min().item(), 0.05)
        out = normalizer({"policy": torch.tensor([[1.1, 4.0]])})["policy"]
        self.assertLess(abs(out[0, 0].item()), 3.0)


if __name__ == "__main__":
    unittest.main()
