import torch
import torch.nn as nn
from unsloth import FastLanguageModel

class TimestepEmbedder(nn.Module):
    """
    Projects scalar timesteps into a latent space vector.
    Used to inform the model about the current diffusion stage.
    """
    def __init__(self, latent_dim, timestep_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, timestep_dim),
            nn.SiLU(),
            nn.Linear(timestep_dim, latent_dim)
        )
    def forward(self, t):
        # t can be [B] or [B, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.mlp(t)

class DenoisingHead(nn.Module):
    """
    Specific projection head for predicting the noise component.
    Maps transformer hidden states back to latent (embedding) dimension.
    """
    def __init__(self, hidden_size, latent_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_size, latent_dim)
    def forward(self, x):
        return self.proj(x)

class ThunderModelAdapter:
    """
    Injects LoRA adapters and activates bilateral convergence capabilities
    for the Thunder diffusion engine.
    """
    
    def __init__(self, model):
        self.model = model

    def apply_lora(self, r=64, lora_alpha=128, target_modules=None):
        """
        Applies LoRA adapters optimized for parallel denoising.
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                             "gate_proj", "up_proj", "down_proj"]

        print(f"⚡ Thunder: Injecting LoRA (r={r}, alpha={lora_alpha})...")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        return self.model

    def adapt_for_diffusion(self, checkpoint_path=None):
        """
        Incorporate diffusion-specific layers into the backbone.
        If checkpoint_path is provided, try to load existing weights.
        """
        print("⚡ Thunder: Adapting architecture for Diffusion (Embedder + Head)...")
        hidden_size = self.model.config.hidden_size
        latent_dim = hidden_size 
        
        # Attach new components to the model object
        self.model.timestep_embedder = TimestepEmbedder(latent_dim).to(self.model.device).to(self.model.dtype)
        self.model.denoising_head = DenoisingHead(hidden_size, latent_dim).to(self.model.device).to(self.model.dtype)
        
        # Try to load existing weights if available
        if checkpoint_path:
            self.load_diffusion_layers(checkpoint_path)
        
        # Monkey-patch methods onto the model object for direct access
        import types
        self.model.get_timestep_embedding = types.MethodType(self.__class__.get_timestep_embedding, self.model)
        self.model.predict_noise = types.MethodType(self.__class__.predict_noise, self.model)
        
        # Ensure the model remembers it's adapted
        self.model.is_thunder_adapted = True
        
        self.enable_bidirectional_attention()
        return self.model

    def save_diffusion_layers(self, path):
        """
        Saves the weights of the custom diffusion layers.
        """
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.timestep_embedder.state_dict(), os.path.join(path, "timestep_embedder.pt"))
        torch.save(self.model.denoising_head.state_dict(), os.path.join(path, "denoising_head.pt"))
        print(f"⚡ Thunder: Diffusion layers saved to {path}")

    def load_diffusion_layers(self, path):
        """
        Loads the weights of the custom diffusion layers.
        """
        import os
        embed_path = os.path.join(path, "timestep_embedder.pt")
        head_path = os.path.join(path, "denoising_head.pt")
        
        if os.path.exists(embed_path):
            self.model.timestep_embedder.load_state_dict(torch.load(embed_path, map_location=self.model.device))
            print(f"⚡ Thunder: Loaded TimestepEmbedder from {embed_path}")
        if os.path.exists(head_path):
            self.model.denoising_head.load_state_dict(torch.load(head_path, map_location=self.model.device))
            print(f"⚡ Thunder: Loaded DenoisingHead from {head_path}")

    def enable_bidirectional_attention(self):
        """
        Activates global coupling across the latent field.
        Turns the backbone into a non-causal noise predictor where every 
        position can influence all others simultaneously (Full Sequence Attention).
        """
        print("⚡ Thunder: Activating Global Field Coupling (Non-Causal Diffusion)...")
        
        # In the context of a diffusion model, we disable causal masking.
        # This allows the model to treat the entire sequence as a single field.
        if hasattr(self.model.config, "is_causal"):
            self.model.config.is_causal = False
            
        # For Phi/Llama models, we often need to monkey patch the attention 
        # forward pass or rely on `is_causal=False` flowing down to the attention math.
        # Unsloth handles this mostly via config, but we ensure it's set globally.
        self.model.config.use_cache = False # Not useful for diffusion (we don't autoregress)

    def get_timestep_embedding(self, t, batch_size, seq_len):
        """
        Creates a temporal embedding field to be added to input embeddings.
        Includes a learnable scale factor to prevent gradient flooding.
        """
        # When monkey-patched, 'self' is the model
        if not hasattr(self, "timestep_scale"):
            # Initialize scale at 0.1 to avoid dominating the latent field initially
            self.timestep_scale = nn.Parameter(torch.tensor(0.1, device=self.device, dtype=self.dtype))
            
        t_embed = self.timestep_embedder(t) # [B, D]
        # Normalize and scale
        t_embed = (t_embed / (t_embed.norm(dim=-1, keepdim=True) + 1e-6)) * self.timestep_scale
        return t_embed.unsqueeze(1).expand(-1, seq_len, -1)

    def predict_noise(self, hidden_states, t=None, condition_len=0, condition=None):
        """
        Main interface for noise prediction (ε-prediction).
        Projects backbone hidden states back to noise space.
        """
        # When monkey-patched, 'self' is the model
        
        # If we have clean prompt conditioned prefix, slice it off
        if condition_len > 0:
            latent_hidden = hidden_states[:, condition_len:, :]
        else:
            latent_hidden = hidden_states
            
        # Project hidden states to noise prediction via DenoisingHead
        return self.denoising_head(latent_hidden)
