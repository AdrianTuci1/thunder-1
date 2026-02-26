import torch

class StreamOrchestrator:
    """
    Orchestrates the global diffusion steps on the full sequence.
    (Formerly managed tile streams with complex tree parallelism, now 
    simplified for full-attention context computation).
    """
    
    def __init__(self, stream_count=1):
        self.stream_count = stream_count
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

    def crystallize_global_field(self, global_latent, diffusion_engine, total_steps=50, prompt_embeds=None):
        """
        Executes the synchronized reverse diffusion loop across the entire latent sequence.
        global_latent: shape [B, L, D] representing the entire sequence.
        """
        print(f"⚡ Thunder: Starting Global Sequence Crystallization ({total_steps} steps)...")
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                result = diffusion_engine.crystallize_sequence(
                    initial_noise=global_latent,
                    steps=total_steps,
                    prompt_embeds=prompt_embeds
                )
            # Wait for execution to finish
            torch.cuda.current_stream().wait_stream(self.stream)
            current_state = result
        else:
            current_state = diffusion_engine.crystallize_sequence(
                initial_noise=global_latent,
                steps=total_steps,
                prompt_embeds=prompt_embeds
            )
            
        print("⚡ Thunder: Global Field Crystallization Complete.")
        return current_state
