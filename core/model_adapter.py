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

class EmbeddingBridge(nn.Module):
    """
    Translates discrete token embeddings into a continuous, isotropic latent space
    suitable for Gaussian diffusion.
    """
    def __init__(self, hidden_size, latent_dim):
        super().__init__()
        # Optional projection if hidden_size != latent_dim
        self.proj = nn.Linear(hidden_size, latent_dim) if hidden_size != latent_dim else nn.Identity()
        # LayerNorm with NO learnable parameters to enforce N(0, I) distribution
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        
    def forward(self, x):
        x = self.proj(x)
        return self.norm(x)

class ThunderModelAdapter:
    """
    Injects LoRA adapters and activates bilateral convergence capabilities
    for the Thunder diffusion engine.
    """
    
    def __init__(self, model):
        self.model = model

    def freeze_bottom_layers(self, freeze_up_to=16):
        """
        Freezes the bottom layers of the model to act as a stable feature extractor.
        """
        print(f"⚡ Thunder: Freezing bottom {freeze_up_to} layers...")
        
        # Typically, Unsloth/HF models store layers in model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
            for i in range(min(freeze_up_to, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False
                
        # Also freeze the base embeddings as they are the source of truth for the bridge
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            for param in self.model.model.embed_tokens.parameters():
                param.requires_grad = False

    def apply_lora(self, r=128, lora_alpha=256, target_modules=None):
        """
        Applies LoRA adapters optimized for parallel denoising.
        Only applied to unfrozen (upper) layers.
        """
        if target_modules is None:
            # We want to target all linear projections to allow maximum plasticity
            # for the bidirectional attention shift.
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

    def adapt_for_diffusion(self, freeze_layers=16):
        """
        Incorporate diffusion-specific layers into the backbone and apply freezing.
        """
        print("⚡ Thunder: Adapting architecture for Continuous Diffusion...")
        hidden_size = self.model.config.hidden_size
        # Assuming latent dim is the same as hidden size for simplified diffusion
        latent_dim = hidden_size 
        
        # Attach new components to the model object
        self.model.embedding_bridge = EmbeddingBridge(hidden_size, latent_dim).to(self.model.device)
        self.model.timestep_embedder = TimestepEmbedder(latent_dim).to(self.model.device)
        self.model.denoising_head = DenoisingHead(hidden_size, latent_dim).to(self.model.device)
        
        self.freeze_bottom_layers(freeze_up_to=freeze_layers)
        self.enable_bidirectional_attention()
        return self.model

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
        """
        t_embed = self.model.timestep_embedder(t) # [B, D]
        return t_embed.unsqueeze(1).expand(-1, seq_len, -1)

    def predict_noise(self, hidden_states, t=None, condition_len=0):
        """
        Main interface for noise prediction (ε-prediction).
        Projects backbone hidden states back to noise space.
        """
        # If we have clean prompt conditioned prefix, slice it off
        if condition_len > 0:
            latent_hidden = hidden_states[:, condition_len:, :]
        else:
            latent_hidden = hidden_states
            
        # Project hidden states to noise prediction via DenoisingHead
        return self.model.denoising_head(latent_hidden)
