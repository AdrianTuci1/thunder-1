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

    def predict_noise(self, latent_field, t, condition=None):
        """
        Main interface for noise prediction (ε-prediction).
        Takes a latent field (noisy embeddings) and a timestep, returns predicted noise.
        Optionally takes a 'condition' (clean prompt embeddings) to prepend.
        """
        # 1. Inform the backbone about the current diffusion temporal stage
        t_embed = self.model.timestep_embedder(t) # [B, D]
        t_embed_seq = t_embed.unsqueeze(1).expand(-1, latent_field.shape[1], -1)
        
        # 2. Residual coupling of temporal info into the noisy field
        coupled_field = latent_field + t_embed_seq
        
        # 3. Concatenate prompt (clean) + z_t (noisy) if condition is provided
        if condition is not None:
            # condition: [B, L_prompt, D]
            # coupled_field: [B, L_latent, D]
            # Full input: [prompt | z_t]
            model_input = torch.cat([condition, coupled_field], dim=1)
        else:
            model_input = coupled_field
        
        # 4. Model forward pass through the non-causal backbone
        # The model "sees" the full sequence symmetrically: prompt + z_t together.
        outputs = self.model(
            inputs_embeds=model_input,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 5. Extract only the predictions corresponding to the noisy latent part
        last_hidden = outputs.hidden_states[-1]
        
        if condition is not None:
            # Slice off the prompt part: we only predict noise for z_t
            prompt_len = condition.shape[1]
            latent_hidden = last_hidden[:, prompt_len:, :]
        else:
            latent_hidden = last_hidden
            
        # 6. Project hidden states to noise prediction via DenoisingHead
        return self.model.denoising_head(latent_hidden)
