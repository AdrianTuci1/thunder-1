import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reasoning.router import ThunderRouter
from core.scheduler import ThunderScheduler

def test_modes():
    router = ThunderRouter()
    scheduler = ThunderScheduler()
    
    queries = [
        ("Cine este Președintele?", "instant"),
        ("Explică mecanismul de fuziune nucleară.", "thinking"),
        ("Cum pot face o omletă?", None) # Expect auto-fast or thinking
    ]
    
    print("--- Mode Routing & Step Scaling Test ---")
    for query, forced_mode in queries:
        route = router.route_query(query, forced_mode=forced_mode)
        mode = route["mode"]
        predicted_len = route["predicted_length"]
        steps = scheduler.calculate_steps(mode=mode, predicted_length=predicted_len)
        
        print(f"Query: '{query}'")
        print(f"  Forced Mode: {forced_mode}")
        print(f"  Routed Mode: {mode}")
        print(f"  Predicted Len: {predicted_len}")
        print(f"  Calculated Steps: {steps}")
        print("-" * 40)

if __name__ == "__main__":
    test_modes()
