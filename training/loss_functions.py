import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionLMLoss:
    """
    Loss functions for continuous Diffusion-LM with x0-parametrization.
    """
    
    def __init__(self, t_round_penalty=0.01):
        self.t_round_penalty = t_round_penalty

    def calculate_denoising_diffusion_loss(self, x0_pred, target_token_ids, embedding_weight, t_indices, alphas_cumprod, attention_mask=None, logit_scale=1.0):
        """
        Mercury 1 Adaptation: Denoising Diffusion Loss
        -E_t [ gamma(t) * E_{z_t ~ q} log p_theta(x | z_t) ]
        This computes the token-level cross-entropy loss directly from the predicted continuous x0.
        """
        batch_size, seq_len, hidden_size = x0_pred.shape
        
        # Map predicted continuous states to vocabulary probabilities
        logits = torch.matmul(x0_pred, embedding_weight.t())
        
        # Apply scaling for softmax stability during early diffusion steps
        if logit_scale != 1.0:
            logits = logits / logit_scale
            
        logits = logits.view(-1, logits.size(-1))
        targets = target_token_ids.view(-1)
        
        if attention_mask is not None:
            targets_masked = targets.clone()
            mask_flat = attention_mask.view(-1)
            targets_masked[mask_flat == 0] = -100
            
            # Loss per position
            ce_loss_raw = F.cross_entropy(logits, targets_masked, ignore_index=-100, reduction='none')
            ce_loss_matrix = ce_loss_raw.view(batch_size, seq_len)
            
            # For gamma(t), Mercury uses a time-dependent weighting, but uniform or simplified SNR
            # weights can be applied. We start with uniform scaled by attention_mask.
            ce_loss = (ce_loss_matrix * attention_mask).sum() / (attention_mask.sum() + 1e-6)
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
        Calculates the complete masked loss for a training step using Mercury 1 objectives.
        """
        # 1. Main Diffusion Loss (Denoising Diffusion Loss on Cross Entropy for all t)
        denoising_loss = self.calculate_denoising_diffusion_loss(
            x0_pred=x0_pred, 
            target_token_ids=input_ids, 
            embedding_weight=embedding_weight, 
            t_indices=t_indices, 
            alphas_cumprod=alphas_cumprod, 
            attention_mask=attention_mask, 
            logit_scale=logit_scale
        )
        
        # 2. xT Regularization (t = T)
        alpha_bar_T = alphas_cumprod[-1]
        xt_reg = self.calculate_xt_regularization(x0_target, alpha_bar_T, attention_mask=attention_mask)
        
        # L_round_loss is now fully absorbed by Denoising Diffusion Loss across all timesteps
        l_round_loss = torch.tensor(0.0, device=x0_pred.device, dtype=x0_pred.dtype)
        
        # Combined Loss
        total_loss = denoising_loss + xt_reg
            
        return total_loss, denoising_loss, l_round_loss
