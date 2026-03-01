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
        
        # Normalize t to [0, 1] for stable embedding
        # num_train_timesteps is usually 2000
        # diffusion_steps (T) is usually 2000
        from core.config_manager import THUNDER_CONFIG
        max_t = THUNDER_CONFIG["diffusion"].get("diffusion_steps", 2000)
        t_norm = t.float() / max_t
        
        # Ensure t_norm is correct dtype (bfloat16 if supported, else float16)
        target_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        t_norm = t_norm.to(target_dtype)
        return self.mlp(t_norm)


class PrefixLMDiffusionAdapter:
    """
    Adapts a causal LLM (like Llama-3.2) for continuous diffusion
    by using PrefixLM (bidirectional attention) and x0-parametrization.
    """
    
    def __init__(self, model):
        self.model = model

    def apply_lora(self, r=64, lora_alpha=128, target_modules=None):
        """
        Applies LoRA adapters optimized for diffusion fine-tuning.
        """
        if target_modules is None:
            # Full linear layer targeting is recommended for major capability shifts
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                             "gate_proj", "up_proj", "down_proj"]

        print(f"⚡ Thunder PrefixLM: Injecting LoRA (r={r}, alpha={lora_alpha})...")
        
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
        Incorporate diffusion-specific layers and settings into the backbone.
        """
        print("⚡ Thunder PrefixLM: Adapting architecture for Diffusion (x0-parametrization)...")
        hidden_size = self.model.config.hidden_size
        
        # We need a timestep embedder
        self.model.timestep_embedder = TimestepEmbedder(hidden_size).to(self.model.device).to(self.model.dtype)
        
        # x0 parametrization: we predict the clean x0 directly. 
        # INITIALIZATION: Very important. We initialize to a small identity-like transform
        # so that the model starts by outputting something close to its inputs.
        self.model.x0_head = nn.Linear(hidden_size, hidden_size).to(self.model.device).to(self.model.dtype)
        nn.init.eye_(self.model.x0_head.weight)
        # Add a bit of noise to break symmetry but keep it stable
        self.model.x0_head.weight.data += torch.randn_like(self.model.x0_head.weight.data) * 0.01
        nn.init.zeros_(self.model.x0_head.bias)

        # Try to load existing weights if available
        if checkpoint_path:
            self.load_diffusion_layers(checkpoint_path)
        
        self.enable_bidirectional_attention()
        self.replace_forward_for_diffusion()
        
        # EXPERIMENTAL: End-to-End Embedding Training (Paper Section 4.1)
        # Unfreezing the embedding matrix so the model can learn to space out
        # mathematical tokens from natural language tokens in the continuous domain.
        self.model.get_input_embeddings().weight.requires_grad = True
        
        # Ensure the model remembers it's adapted
        self.model.is_thunder_adapted = True
        
        return self.model

    def save_diffusion_layers(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.timestep_embedder.state_dict(), os.path.join(path, "timestep_embedder.pt"))
        torch.save(self.model.x0_head.state_dict(), os.path.join(path, "x0_head.pt"))
        
        if self.model.get_input_embeddings().weight.requires_grad:
            torch.save(self.model.get_input_embeddings().weight.data.cpu(), os.path.join(path, "custom_embeddings.pt"))
            
        print(f"⚡ Thunder PrefixLM: Diffusion layers and custom embeddings saved to {path}")

    def load_diffusion_layers(self, path):
        import os
        embed_path = os.path.join(path, "timestep_embedder.pt")
        head_path = os.path.join(path, "x0_head.pt")
        custom_emb_path = os.path.join(path, "custom_embeddings.pt")
        
        if os.path.exists(embed_path):
            self.model.timestep_embedder.load_state_dict(torch.load(embed_path, map_location=self.model.device))
        if os.path.exists(head_path):
            self.model.x0_head.load_state_dict(torch.load(head_path, map_location=self.model.device))
        if os.path.exists(custom_emb_path):
            self.model.get_input_embeddings().weight.data.copy_(torch.load(custom_emb_path, map_location=self.model.device))

    def enable_bidirectional_attention(self):
        """
        Activates PrefixLM capability.
        Turning off the causal mask and KV-cache. Llama uses FlashAttention usually,
        so we have to ensure the causal flag is properly turned off in the config
        and pass a full 1s attention mask during forward.
        """
        print("⚡ Thunder PrefixLM: Activating Bidirectional Attention...")
        
        if hasattr(self.model.config, "is_causal"):
            self.model.config.is_causal = False
            
        # KV cache makes no sense for full-sequence parallel forward passes
        self.model.config.use_cache = False 


    def replace_forward_for_diffusion(self):
        """
        Replaces the standard model forward pass to support PrefixLM mask
        and continuous latent inputs + timestep injection.
        """
        import types
        
        def diffusion_forward(self_model, x_t, t, attention_mask=None, self_cond=None, **kwargs):
            """
            Custom forward pass with Self-Conditioning support.
            x_t: [batch, seq_len, hidden_size]
            t: [batch] or [batch, 1]
            self_cond: [batch, seq_len, hidden_size] - previous x0 estimate (optional)
            """
            device = x_t.device
            batch_size, seq_len, _ = x_t.shape
            
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
            
            # 1. Inject Timestep
            t_embed = self_model.timestep_embedder(t) # [B, D]
            inputs_embeds = x_t + t_embed.unsqueeze(1)
            
            # 2. Self-Conditioning Trick (Chen et al. 2022)
            # If we have a previous estimate of x0, we provide it as additional guidance.
            # In Diffusion-LM, this can be added or concatenated. We'll add it.
            if self_cond is not None:
                inputs_embeds = inputs_embeds + self_cond
            
            # 3. Backbone Forward Pass
            # We bypass the Embedding layer and go directly into the Transformer blocks
            # Llama models usually have `model.model` as the core Transformer, 
            # and `model.lm_head` as the vocabulary projector.
            # Depending on if it's PeftModel or LlamaForCausalLM, the path might be slightly different.
            
            # We pass inputs_embeds directly and get the hidden states.
            # We need to find the core Transformer module regardless of PEFT wrappers.
            if hasattr(self_model, "base_model") and hasattr(self_model.base_model, "model") and hasattr(self_model.base_model.model, "model"):
                transformer = self_model.base_model.model.model
            elif hasattr(self_model, "model"):
                # Usually it's model.model for LlamaForCausalLM, but 
                # actually LlamaForCausalLM has `.model` which IS the LlamaModel!
                if isinstance(self_model.model, torch.nn.Module) and hasattr(self_model.model, "layers"):
                    transformer = self_model.model
                elif hasattr(self_model.model, "model"):
                    transformer = self_model.model.model
                else:
                    transformer = self_model.model
            else:
                transformer = self_model

            outputs = transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                **kwargs
            )
            
            # The last hidden state
            last_hidden_state = outputs.last_hidden_state # [batch, seq_len, hidden_size]
            
            # 4. Predict x0
            # Parametrization: the model outputs its best guess for the clean x0
            x0_pred = self_model.x0_head(last_hidden_state)
            
            return x0_pred
            
        self.model.diffusion_forward = types.MethodType(diffusion_forward, self.model)
