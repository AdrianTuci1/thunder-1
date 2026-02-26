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

    def sample(self, latent_vectors, lm_head):
        """
        Samples tokens from the continuous latent field by projecting back to the vocabulary.
        latent_vectors: [B, L, D] or [L, D]
        lm_head: The prediction head (un-embedding layer) of the backbone model, projecting D -> V.
        """
        # Project Latent Space -> Vocabulary Logits
        logits = lm_head(latent_vectors)
        
        if logits.dim() == 3:
            # Handle batched logits [B, L, V]
            return torch.stack([self._sample_logits(l) for l in logits])
        return self._sample_logits(logits)

    def _sample_logits(self, logits):
        """Internal method performing the actual temperature/Top-K/Top-P scaling on logits."""
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
        next_tokens = torch.multinomial(probs, num_samples=1)
        
        return next_tokens

    def crystallize_sampling(self, latent_field, lm_head, noise_level):
        """
        Specialized sampling for diffusion where the 'noise' level 
        dictates the randomness (temperature) of the projection.
        Higher noise -> higher temperature/randomness.
        """
        # Dynamic temperature based on the estimated noise level remaining
        dynamic_temp = self.temperature * (1.0 + noise_level)
        
        # Temporarily override temperature for this sampling step
        original_temp = self.temperature
        self.temperature = dynamic_temp
        
        sampled_tokens = self.sample(latent_field, lm_head)
        
        # Restore original temperature
        self.temperature = original_temp
        
        return sampled_tokens
