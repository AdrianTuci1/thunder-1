import torch
import torch._dynamo
from core.config_manager import THUNDER_CONFIG
from training.noise_scheduler import ThunderNoiseScheduler
import torch.nn.functional as F

class PrefixLMDiffusionEngine:
    """
    The Continuous Diffusion Engine using PrefixLM and x0-parametrization.
    Implements the generation loops with the Clamping Trick.
    """
    
    def __init__(self, model):
        self.model = model
        # Using the Sqrt Noise Scheduler for text distribution stability
        self.noise_scheduler = ThunderNoiseScheduler()

    def clamp_to_vocabulary(self, x0_pred, embedding_matrix, logit_scale=1.0, temperature=1.0):
        """
        The Clamping Trick: Maps predicted standardized vectors to the nearest
        actual discrete token embedding and returns confidence scores.
        """
        # Calculate logits in standardized space
        logits = torch.matmul(x0_pred, embedding_matrix.t())
        
        if logit_scale != 1.0:
            logits = logits / logit_scale
            
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            # Sample from categorical distribution
            b, l, v = probs.shape
            closest_token_ids = torch.multinomial(probs.view(-1, v), 1).view(b, l)
            # For confidence, we take the max probability of the top token
            confidences, _ = torch.max(probs, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            confidences, closest_token_ids = torch.max(probs, dim=-1)

        clamped_x0 = F.embedding(closest_token_ids, embedding_matrix)
        
        return clamped_x0, closest_token_ids, confidences

    def process_single_step(self, x_t, step_idx, total_steps, embedding_matrix, apply_clamping=True, guidance_scale=1.0, 
                            uncond_prompt_embeds=None, logit_scale=1.0, temperature=1.0, anchor_len=0, self_cond=None,
                            clean_prompt_embeds=None):
        """
        Executes one reverse diffusion step with Grounding Fix (Zero-Drift Prompt).
        """
        device = x_t.device
        diffusion_steps = self.noise_scheduler.diffusion_steps
        
        t_now = int(( (total_steps - 1 - step_idx) / total_steps) * diffusion_steps)
        t_next = int(( (total_steps - 2 - step_idx) / total_steps) * diffusion_steps) if (total_steps - 2 - step_idx) >= 0 else 0
        
        alpha_now = self.noise_scheduler.alphas_cumprod[t_now].to(device).to(self.model.dtype)
        alpha_next = self.noise_scheduler.alphas_cumprod[t_next].to(device).to(self.model.dtype) if t_now > 0 else torch.tensor(1.0, device=device, dtype=self.model.dtype)

        with torch.no_grad():
            target_dtype = self.model.dtype
            x_t = x_t.to(target_dtype)
            t_tensor = torch.tensor([t_now], device=device).to(target_dtype)
            
            if hasattr(self.model, "diffusion_forward"):
                # Forward with Prompt (Conditional)
                x0_cond = self.model.diffusion_forward(x_t, t_tensor, attention_mask=None, self_cond=self_cond)
                
                if guidance_scale > 1.0:
                    # Create Unconditional (Null) state: mask out the anchor influence
                    # For PrefixLM, "unconditional" means the model sees no prefix or a padding prefix.
                    # We can simulate this by zeroing out the prompt part of the input.
                    x_uncond = x_t.clone()
                    if anchor_len > 0:
                        x_uncond[:, :anchor_len, :] = 0.0 # Null embedding
                    
                    x0_uncond = self.model.diffusion_forward(x_uncond, t_tensor, attention_mask=None, self_cond=self_cond)
                    
                    # Extrapolate: x0_pred = x0_uncond + scale * (x0_cond - x0_uncond)
                    x0_pred = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
                else:
                    x0_pred = x0_cond
                
                # --- Grounding Fix: Overwrite x0_pred for prompt with CLEAN embeds ---
                # This prevents the model from "misremembering" the question.
                if anchor_len > 0 and clean_prompt_embeds is not None:
                    x0_pred[:, :anchor_len, :] = clean_prompt_embeds[:, :anchor_len, :].to(target_dtype)

            else:
                raise AttributeError("Model must be adapted via PrefixLMDiffusionAdapter before inference.")

            # --- Standardized Clamping ---
            avg_conf = 0.0
            if apply_clamping:
                x0_clamped, token_ids, confs = self.clamp_to_vocabulary(x0_pred, embedding_matrix, logit_scale=logit_scale, temperature=temperature)
                if anchor_len < x0_pred.shape[1]:
                    # Confidence of the Assistant part only
                    avg_conf = confs[:, anchor_len:].mean().item()
                x0_for_posterior = x0_clamped
            else:
                # We still want confidence even if not clamping the actual state update
                _, token_ids, confs = self.clamp_to_vocabulary(x0_pred, embedding_matrix, logit_scale=logit_scale, temperature=temperature)
                if anchor_len < x0_pred.shape[1]:
                    avg_conf = confs[:, anchor_len:].mean().item()
                x0_for_posterior = x0_pred

            if t_now > 0:
                beta_t = 1.0 - (alpha_now / alpha_next)
                mean = (torch.sqrt(alpha_next) * beta_t / (1.0 - alpha_now)) * x0_for_posterior + \
                       (torch.sqrt(alpha_now / alpha_next) * (1.0 - alpha_next) / (1.0 - alpha_now)) * x_t
                
                variance = ((1.0 - alpha_next) / (1.0 - alpha_now)) * beta_t
                noise = torch.randn_like(x_t)
                x_t_minus_1 = mean + torch.sqrt(variance) * noise
            else:
                x_t_minus_1 = x0_for_posterior

            return x_t_minus_1, token_ids, x0_pred, avg_conf

    def generate(self, shape, embedding_matrix, steps=None, prompt_embeds=None, anchor_len=0, 
                 apply_clamping=True, guidance_scale=1.0, uncond_prompt_embeds=None, 
                 return_trajectory=False, early_stopping_patience=3):
        """
        Generation with Dynamic Decoding Steps (10 to 100) based on complexity.
        embedding_matrix: EXPECTS RAW EMBEDDINGS
        prompt_embeds: EXPECTS RAW EMBEDDINGS
        """
        # Dynamic Steps Logic (Paper Section C: Downsampling)
        # We scale steps between 10 (simple) and 100 (complex) depending on the prompt length
        max_dynamic_steps = 100
        min_dynamic_steps = 10
        if steps is None:
            # Heuristic: More context = harder diffusion problem, need more steps
            complexity_factor = min(1.0, anchor_len / 512.0) # Assume 512 is "max" complexity for this heuristic
            steps = int(min_dynamic_steps + (max_dynamic_steps - min_dynamic_steps) * complexity_factor)
            
        print(f"⚡ Thunder PrefixLM: Generating (Dynamic Steps: {steps}, CFG: {guidance_scale}, EarlyStop: {early_stopping_patience}, Grounded: True)...")

        device = self.model.device
        dtype = self.model.dtype
        logit_scale = (shape[-1] ** 0.5) + 1e-6
        
        # 1. Start with pure noise x_T (N(0, 1))
        current_state = torch.randn(shape, device=device, dtype=dtype)
        
        # 2. Apply Prompt (already standardized in the script)
        if anchor_len > 0 and prompt_embeds is not None:
            current_state[:, :anchor_len, :] = prompt_embeds[:, :anchor_len, :].to(dtype)

        trajectory = []
        final_tokens = None
        
        # Early stopping & Self-conditioning state
        last_response_tokens = None
        last_x0_pred = None # Zero initialized implicitly by None in forward usually, but let's be explicit
        stable_count = 0
        
        for step_idx in range(steps):
            # 1. Temperature & Logit Scale Annealing
            # Diffusion needs more exploration early on.
            # Start with high temperature (2.0) dropping to sharp (0.1)
            temp = 2.0 - (1.9 * (step_idx / steps))
            
            # Dynamic Logit Scale: Progressively increase from slightly scaled down to full sqrt(D) + epsilon
            # This restricts premature hard-commitment.
            current_logit_scale = (logit_scale * 0.5) + (logit_scale * 0.5) * (step_idx / steps)
            
            # 2. Iterative Clamping (Mercury 1 Adaptation)
            # Apply clamping earlier in the intermediate steps (50% onwards) rather than just the final 20%
            # This grounds the parallel generation and reduces accumulated rounding errors during coarse-to-fine refinement.
            use_clamp = apply_clamping and (step_idx > int(steps * 0.5))
            
            new_state_clamped, current_token_ids, x0_pred_val, step_conf = self.process_single_step(
                x_t=current_state, 
                step_idx=step_idx, 
                total_steps=steps, 
                embedding_matrix=embedding_matrix,
                apply_clamping=True, # Forced for monitoring
                guidance_scale=guidance_scale,
                uncond_prompt_embeds=uncond_prompt_embeds,
                logit_scale=current_logit_scale,
                temperature=temp,
                anchor_len=anchor_len,
                self_cond=last_x0_pred,
                clean_prompt_embeds=prompt_embeds # Grounding Fix
            )
            
            last_x0_pred = x0_pred_val

            if not use_clamp:
                actual_state, _, _, _ = self.process_single_step(
                    x_t=current_state, step_idx=step_idx, total_steps=steps, 
                    embedding_matrix=embedding_matrix, apply_clamping=False,
                    guidance_scale=guidance_scale, logit_scale=current_logit_scale, 
                    temperature=temp, anchor_len=anchor_len, 
                    self_cond=last_x0_pred, clean_prompt_embeds=prompt_embeds
                )
                current_state = actual_state
            else:
                current_state = new_state_clamped

            # Re-anchor prompt every step (Redundant but safe)
            if anchor_len > 0 and prompt_embeds is not None:
                current_state[:, :anchor_len, :] = prompt_embeds[:, :anchor_len, :].to(dtype)

            final_tokens = current_token_ids
            
            # Dynamic Early Stopping logic: MONITOR RESPONSE ONLY
            response_tokens = current_token_ids[:, anchor_len:]
            
            if last_response_tokens is not None and torch.equal(response_tokens, last_response_tokens):
                stable_count += 1
            else:
                stable_count = 0
            
            last_response_tokens = response_tokens
            
            # CONFIDENCE-BASED EXIT: If stable AND high confidence, exit early.
            # 0.9 is a high bar, meaning the model is very sure.
            is_high_conf = (step_conf > 0.9)
            
            if return_trajectory and (step_idx % max(1, steps // 20) == 0 or step_idx == steps - 1 or stable_count >= early_stopping_patience):
                trajectory.append({
                    "step": step_idx,
                    "tokens": current_token_ids[0].cpu().tolist()
                })

            if early_stopping_patience > 0 and stable_count >= early_stopping_patience:
                # Add confidence condition for even faster exit on simple prompts
                if step_idx > (steps // 5) or is_high_conf:
                    print(f"⚡ Early Exit: Output stabilized at step {step_idx} (Stability: {stable_count}, Conf: {step_conf:.2f})")
                    break
                
        # Final clamping to get the discrete tokens if not already done
        if final_tokens is None: # This case should ideally not happen if current_token_ids is always set
            _, final_tokens = self.clamp_to_vocabulary(current_state, embedding_matrix, logit_scale=logit_scale)
            
        if return_trajectory:
            return current_state, final_tokens, trajectory
            
        return current_state, final_tokens
