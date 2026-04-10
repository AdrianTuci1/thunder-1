import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config_manager import THUNDER_CONFIG
from core.scheduler import ThunderScheduler
from reasoning.router import ThunderRouter


class TestThunderLogic(unittest.TestCase):
    def test_router_modes_remain_supported(self):
        router = ThunderRouter()
        route = router.route_query("Explain how diffusion language models denoise text.", forced_mode="thinking")
        self.assertEqual(route["mode"], "thinking")
        self.assertEqual(route["target"], "INTERNAL_THUNDER")

    def test_scheduler_targets_fast_budget(self):
        scheduler = ThunderScheduler()
        steps = scheduler.calculate_steps(mode="fast", anchor_len=512)
        self.assertGreaterEqual(steps, THUNDER_CONFIG["logic"]["min_steps"])
        self.assertLessEqual(steps, THUNDER_CONFIG["logic"]["modes"]["fast"]["max"])


if __name__ == "__main__":
    unittest.main()
