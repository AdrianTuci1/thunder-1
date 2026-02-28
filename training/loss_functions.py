import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionLMLoss:
    """
    Loss functions for continuous Diffusion-LM with x0-parametrization.
    """
    
    def __init__(self, t_round_penalty=0.01):
        self.t_round_penalty = t_round_penalty

    def calculate_x0_mse(self, x0_pred, x0_target, attention_mask=None):
        """
        Computes the L2 distance (MSE) between the predicted x0 and the real clean embeddings.
        If attention_mask is provided, we only average over the non-pad tokens.
        """
        if attention_mask is not None:
            # Squared error [B, L, H]
            sq_err = (x0_pred - x0_target)**2
            # Sum over dimension [B, L]
            # Then apply mask [B, L]
            masked_err = sq_err.sum(dim=-1) * attention_mask
            # Mean over non-masked tokens
            loss = masked_err.sum() / (attention_mask.sum() * x0_pred.size(-1) + 1e-6)
        else:
            loss = F.mse_loss(x0_pred, x0_target, reduction='mean')
        return loss

    def calculate_l_round(self, x0_pred, target_token_ids, embedding_weight, attention_mask=None, logit_scale=1.0):
        """
        L_round: Cross-Entropy loss at t=1 (or t=0).
        logit_scale: factor to divide/multiply logits (e.g., sqrt(hidden_size)).
        """
        batch_size, seq_len, hidden_size = x0_pred.shape
        
        # [B, L, V]
        logits = torch.matmul(x0_pred, embedding_weight.t())
        
        # Apply scaling for softmax stability
        if logit_scale != 1.0:
            logits = logits / logit_scale
        
        # Reshape for cross entropy
        logits = logits.view(-1, logits.size(-1))
        targets = target_token_ids.view(-1)
        
        if attention_mask is not None:
            targets_masked = targets.clone()
            mask_flat = attention_mask.view(-1)
            targets_masked[mask_flat == 0] = -100
            ce_loss = F.cross_entropy(logits, targets_masked, ignore_index=-100, reduction='mean')
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction='mean')
        return ce_loss

    def calculate_xt_regularization(self, x0_target, alpha_bar_T, attention_mask=None):
        """
        Regularization term from Equation 1219/1293: ||sqrt(alpha_bar_T) * x0||^2
        """
        if attention_mask is not None:
            norm_sq = torch.norm(x0_target, dim=-1)**2
            reg = torch.abs(alpha_bar_T) * (norm_sq * attention_mask).sum() / (attention_mask.sum() + 1e-6)
        else:
            reg = torch.abs(alpha_bar_T) * torch.mean(torch.norm(x0_target, dim=-1)**2)
        return reg

    def calculate_total_loss(self, x0_pred, x0_target, input_ids, embedding_weight, t_indices, alphas_cumprod, attention_mask=None, round_threshold=0.15, logit_scale=1.0):
        """
        Calculates the complete masked loss for a training step (Eq 1289).
        """
        # 1. Main Diffusion Loss (MSE on x0 for all t)
        mse_loss = self.calculate_x0_mse(x0_pred, x0_target, attention_mask=attention_mask)
        
        # 2. xT Regularization (t = T)
        alpha_bar_T = alphas_cumprod[-1]
        xt_reg = self.calculate_xt_regularization(x0_target, alpha_bar_T, attention_mask=attention_mask)
        
        # 3. L_round component (t near 0)
        num_timesteps = len(alphas_cumprod)
        t_normalized = t_indices.float() / num_timesteps
        
        l_round_loss = torch.tensor(0.0, device=x0_pred.device, dtype=x0_pred.dtype)
        
        low_noise_mask = (t_normalized < round_threshold)
        if low_noise_mask.any() and self.t_round_penalty > 0:
                round_attention_mask = attention_mask[low_noise_mask] if attention_mask is not None else None
                
                l_round_loss = self.calculate_l_round(
                    x0_pred[low_noise_mask], 
                    input_ids[low_noise_mask], 
                    embedding_weight,
                    attention_mask=round_attention_mask,
                    logit_scale=logit_scale
                )
        
        # Combined Loss
        total_loss = mse_loss + xt_reg + (self.t_round_penalty * l_round_loss)
            
        return total_loss, mse_loss, l_round_loss
