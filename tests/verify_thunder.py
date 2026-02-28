import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.tile_manager import FractalTiler
from reasoning.router import ThunderRouter
from core.scheduler import ThunderScheduler

class TestThunderLogic(unittest.TestCase):
    def test_fractal_tiling(self):
        tiler = FractalTiler()
        # Test tree building logic for 4000 tokens
        tree = tiler.build_tile_tree(4000)
        self.assertIsNotNone(tree)
        self.assertTrue(len(tree.children) > 0)
        print(f"✅ Recursive Tree verified (Branches: {len(tree.children)})")

    def test_routing_logic(self):
        router = ThunderRouter()
        # Test internal routing
        query = "How does parallel diffusion work?"
        route = router.route_query(query)
        self.assertEqual(route["target"], "INTERNAL_THUNDER")
        print("✅ Routing logic Verified.")

    def test_scheduler_steps(self):
        scheduler = ThunderScheduler()
        # Mock a leaf node for testing
        from core.tile_manager import ThunderTileNode
        leaf = ThunderTileNode(0, 512, 4)
        leaf.is_leaf = True
        
        steps = scheduler.calculate_steps(leaf)
        # Leaf nodes should get extra refinement steps
        from core.config_manager import THUNDER_CONFIG
        self.assertTrue(steps >= THUNDER_CONFIG["logic"]["default_steps"])
        print(f"✅ Dynamic Scheduler Verified (Steps: {steps})")

if __name__ == "__main__":
    unittest.main()
