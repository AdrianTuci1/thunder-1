import torch
import torch.nn.functional as F

from core.config_manager import THUNDER_CONFIG

class ThunderTokenSampler:
    """
    High-fidelity sampling for Diffusion-based language modeling.
    """
    
    def __init__(self):
        self.temperature = THUNDER_CONFIG["sampling"]["temperature"]
        self.top_k = THUNDER_CONFIG["sampling"]["top_k"]
        self.top_p = THUNDER_CONFIG["sampling"]["top_p"]

    def sample(self, logits):
        """
        Samples tokens from logits using the configured strategy.
        logits: [B, L, D] or [L, D]
        """
        if logits.dim() == 3:
            # Handle batched logits [B, L, D]
            return torch.stack([self.sample(l) for l in logits])
            
        # 1. Temperature scaling
        logits = logits / max(self.temperature, 1e-5)
        
        # 2. Top-K filtering
        if self.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
            
        # 3. Top-P (Nucleus) filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Shift the indices to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')
            
        # 4. Final Sampling
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token

    def crystallize_sampling(self, latent, noise_level):
        """
        Specialized sampling for diffusion where the 'noise' level 
        dictates the randomness of the sampling.
        Higher noise -> higher temperature/randomness.
        """
        dynamic_temp = self.temperature * (1.0 + noise_level)
        # In a real implementation, this would involve a reverse-diffusion step
        # mapped to token space via the embedding/unembedding layers.
        return self.sample(latent / dynamic_temp)
