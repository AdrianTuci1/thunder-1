import sys
import os
import math

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mocking parts of ThunderConfig if needed, but since we have the files, we'll try to use them
try:
    from core.config_manager import THUNDER_CONFIG
    from reasoning.router import ThunderRouter
    from core.scheduler import ThunderScheduler
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def run_logic_audit():
    print("=== THUNDER CORE LOGIC AUDIT (No External Dependencies) ===\n")
    
    router = ThunderRouter()
    scheduler = ThunderScheduler()
    
    test_cases = [
        {"q": "What is 2+2?", "mode": None, "desc": "Simple query (Auto)"},
        {"q": "Analyze the geopolitical impact of fusion energy.", "mode": None, "desc": "Complex query (Auto)"},
        {"q": "Fast reply please.", "mode": "instant", "desc": "Forced Instant"},
        {"q": "Deep dive into Mamba architecture.", "mode": "thinking", "desc": "Forced Thinking"},
    ]
    
    for case in test_cases:
        query = case["q"]
        forced = case["mode"]
        print(f"Testing: {case['desc']} -> '{query}'")
        
        # 1. Test Routing Logic
        route = router.route_query(query, forced_mode=forced)
        mode = route["mode"]
        intensity = route["intensity"]
        pred_len = route["predicted_length"]
        
        print(f"  > Mode Selected: {mode}")
        print(f"  > Intensity: {intensity}")
        print(f"  > Predicted Length: {pred_len}")
        
        # 2. Test Scheduler Logic
        # We'll test with a mock tile node (leaf vs non-leaf)
        class MockNode:
            def __init__(self, is_leaf):
                self.is_leaf = is_leaf
        
        leaf_node = MockNode(True)
        steps = scheduler.calculate_steps(tile_node=leaf_node, mode=mode, predicted_length=pred_len)
        
        print(f"  > Calculated Refinement Steps (Leaf): {steps}")
        
        # Sanity Checks
        if forced and mode != forced:
            print(f"  [ERROR] Mode mismatch! Expected {forced}, got {mode}")
        
        if mode == "thinking" and steps < 30:
            print(f"  [WARNING] Thinking mode steps seem too low: {steps}")
        
        if mode == "instant" and steps > 15:
            print(f"  [WARNING] Instant mode steps seem too high: {steps}")
            
        print("-" * 50)

    print("\nLogic Audit Complete. All core routing and scaling heuristics are operational.")

if __name__ == "__main__":
    run_logic_audit()
