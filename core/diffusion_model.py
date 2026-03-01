from unsloth import FastLanguageModel
import torch
import torch.nn as nn
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
        
        try:
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
        except Exception as e:
            print(f"⚠️ Unsloth LoRA failed ({e}). Falling back to standard PEFT...")
            
            # Prepare for k-bit training if it's already quantized
            if getattr(self.model, "is_loaded_in_4bit", False) or getattr(self.model, "is_loaded_in_8bit", False):
                try:
                    self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
                except Exception:
                    print("⚠️ Gradient checkpointing not supported by this model. Disabling for LoRA prep...")
                    self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
                
            config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM", # Close enough for non-causal diffusion bridge
            )
            self.model = get_peft_model(self.model, config)
        return self.model

    def adapt_for_diffusion(self, checkpoint_path=None):
        """
        Incorporate diffusion-specific layers and settings into the backbone.
        Optimized for LLaDA/Llama-based architectures.
        """
        print(f"⚡ Thunder PrefixLM: Adapting architecture for Diffusion (x0-parametrization)...")
        hidden_size = self.model.config.hidden_size
        
        # 1. Timestep Embedder
        self.model.timestep_embedder = TimestepEmbedder(hidden_size).to(self.model.device).to(self.model.dtype)
        
        # 2. x0 head (predict cleaner latents)
        # Using a specialized initialization for faster convergence
        self.model.x0_head = nn.Linear(hidden_size, hidden_size).to(self.model.device).to(self.model.dtype)
        nn.init.eye_(self.model.x0_head.weight)
        self.model.x0_head.weight.data += torch.randn_like(self.model.x0_head.weight.data) * 0.005
        nn.init.zeros_(self.model.x0_head.bias)

        # 3. Load weights if resuming/checkpointing
        if checkpoint_path and os.path.isdir(checkpoint_path):
            self.load_diffusion_layers(checkpoint_path)
        
        # 4. Attention & Flow Control
        # LLaDA is bidirectional by default, but we ensure it for all backbones
        self.enable_bidirectional_attention()
        self.replace_forward_for_diffusion()
        
        # 5. LLaDA-Specific VRAM Optimizations
        # If it's a LLaDA model, we can bypass the huge LM head to save VRAM.
        # Checkpointing is skipped here as it can be unstable with standard PEFT.
        model_type_str = str(type(self.model))
        if "LLaDA" in model_type_str:
            inner_model = self.model.model if hasattr(self.model, "model") else self.model
            
            # 5a. Bypass massive Logit Head (ff_out) to save VRAM
            # LLaDA Head: 3072 -> 128,256. Gradient allocation is huge.
            # We don't need it because we use our custom x0_head on hidden states.
            if hasattr(inner_model, "transformer") and hasattr(inner_model.transformer, "ff_out"):
                print("⚡ Thunder: Bypassing LLaDA ff_out head (Identity Patch) to save VRAM...")
                inner_model.transformer.ff_out = nn.Identity().to(self.model.device).to(self.model.dtype)
        
        # 6. Embedding Training
        self.model.get_input_embeddings().weight.requires_grad = True
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
            if self_cond is not None:
                inputs_embeds = inputs_embeds + self_cond
            
            # 3. Backbone Forward Pass
            # We need the module that accepts embeddings and contains 'layers'.
            def get_transformer_outputs(m, embeds, mask, **kwargs):
                # LLaDA and some other models might require 'input_ids' or use 'input_embeddings'
                import inspect
                sig = inspect.signature(m.forward)
                
                call_args = {"attention_mask": mask, **kwargs}
                
                # Determine embedding parameter name
                emb_param = "input_embeddings" if "input_embeddings" in sig.parameters else "inputs_embeds"
                call_args[emb_param] = embeds
                
                # Some models (like LLaDA) have input_ids as a mandatory first arg or keyword
                if "input_ids" in sig.parameters:
                    call_args["input_ids"] = None
                    
                return m(**call_args)

            def find_core_transformer(m):
                # For GSAI-ML LLaDA, model.model is the LLaDAModel
                if hasattr(m, "layers") and not isinstance(m, torch.nn.ModuleDict): return m
                if hasattr(m, "model") and hasattr(m.model, "layers"): return m.model
                return m

            if hasattr(self_model, "base_model"):
                base = self_model.base_model
                inner = base.model if hasattr(base, "model") else base
                transformer = find_core_transformer(inner)
            else:
                transformer = find_core_transformer(self_model)

            outputs = get_transformer_outputs(
                transformer,
                embeds=inputs_embeds,
                mask=attention_mask,
                output_hidden_states=True, # We need the hidden states
                use_cache=False,
                **kwargs
            )
            
            # Extract last hidden state
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                last_hidden_state = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                # hidden_states is a tuple (embeddings, layer_1, ..., layer_n)
                last_hidden_state = outputs.hidden_states[-1]
            elif isinstance(outputs, (tuple, list)):
                last_hidden_state = outputs[0]
            else:
                # Last resort: if it's a dict-like or standard output
                last_hidden_state = outputs.get("last_hidden_state", outputs.get("hidden_states", [None])[-1])

            if last_hidden_state is None:
                raise ValueError(f"Could not extract hidden states from model output type: {type(outputs)}")
                
            # 4. Predict x0
            x0_pred = self_model.x0_head(last_hidden_state)
            
            return x0_pred
            
        self.model.diffusion_forward = types.MethodType(diffusion_forward, self.model)
