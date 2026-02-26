import math

def cosine_schedule(timesteps, s=0.008):
    alphas_cumprod = []
    steps = timesteps + 1
    for t in range(1, steps):
        val = math.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod.append(val)
    
    # Normalize by the first value (which would be t=0)
    # Cos(s/(1+s)*pi/2)^2
    base = math.cos(s / (1 + s) * math.pi * 0.5) ** 2
    return [a / base for a in alphas_cumprod]

def sigmoid_schedule(timesteps, start=-3, end=3):
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    v_start = sigmoid(start)
    v_end = sigmoid(end)
    alphas_cumprod = []
    
    for i in range(timesteps):
        x = start + (end - start) * i / (timesteps - 1)
        val = 1.0 - ((sigmoid(x) - v_start) / (v_end - v_start))
        alphas_cumprod.append(val)
    return alphas_cumprod

def linear_schedule(timesteps):
    alphas_cumprod = []
    current_prod = 1.0
    for i in range(timesteps):
        beta = 0.0001 + (0.02 - 0.0001) * i / (timesteps - 1)
        alpha = 1.0 - beta
        current_prod *= alpha
        alphas_cumprod.append(current_prod)
    return alphas_cumprod

def test_schedules():
    timesteps = 1000
    
    schedules = {
        "linear": linear_schedule,
        "cosine": cosine_schedule,
        "sigmoid": sigmoid_schedule
    }
    
    for name, func in schedules.items():
        print(f"\n--- Testing {name} schedule ---")
        alphas_cumprod = func(timesteps)
        
        # Check monotonicity
        is_monotonic = all(alphas_cumprod[i] <= alphas_cumprod[i-1] for i in range(1, len(alphas_cumprod)))
        print(f"Monotonic: {is_monotonic}")
        
        # Print values at key percentages
        for p in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            idx = int(p * (timesteps - 1))
            val = alphas_cumprod[idx]
            print(f"  Step {idx:4d} ({p*100:3.0f}%): alpha_cumprod = {val:.4f}")

if __name__ == "__main__":
    test_schedules()
