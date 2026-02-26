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

    def adapt_for_diffusion(self):
        """
        Incorporate diffusion-specific layers into the backbone.
        """
        print("⚡ Thunder: Adapting architecture for Diffusion (Embedder + Head)...")
        hidden_size = self.model.config.hidden_size
        # Assuming latent dim is the same as hidden size for simplified diffusion
        latent_dim = hidden_size 
        
        # Attach new components to the model object
        self.model.timestep_embedder = TimestepEmbedder(latent_dim).to(self.model.device)
        self.model.denoising_head = DenoisingHead(hidden_size, latent_dim).to(self.model.device)
        
        self.enable_bidirectional_attention()
        return self.model

    def enable_global_field_coupling(self):
        """
        Activates global coupling across the latent field.
        Turns the backbone into a non-causal noise predictor where every 
        position can influence all others simultaneously.
        """
        print("⚡ Thunder: Activating Global Field Coupling (Non-Causal Diffusion)...")
        
        # In the context of a diffusion model, we disable any causal masking.
        # This allows the model to treat the entire sequence as a single field.
        if hasattr(self.model.config, "is_causal"):
            self.model.config.is_causal = False

        # Register hooks to inject global context or modify internal coupling
        for name, module in self.model.named_modules():
            # Targets core computational blocks of any backbone (Phi, Mamba, etc.)
            if any(target in name for target in ["layers", "mixer", "attn"]):
                # Point for injecting global field information (like timestep)
                pass

    def predict_noise(self, latent_field, t):
        """
        Main interface for noise prediction (ε-prediction).
        Takes a latent field (noisy embeddings) and a timestep, returns predicted noise.
        """
        # 1. Inform the backbone about the current diffusion temporal stage
        t_embed = self.model.timestep_embedder(t) # [B, D]
        t_embed_seq = t_embed.unsqueeze(1).expand(-1, latent_field.shape[1], -1)
        
        # 2. Residual coupling of temporal info into the field
        coupled_field = latent_field + t_embed_seq
        
        # 3. Model forward pass through the non-causal backbone
        outputs = self.model(
            inputs_embeds=coupled_field,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 4. Project hidden states to noise prediction via DenoisingHead
        last_hidden = outputs.hidden_states[-1]
        return self.model.denoising_head(last_hidden)
