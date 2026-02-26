import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.loss_functions import HybridLoss

def test_hybrid_loss():
    print("Testing HybridLoss Improvements...")
    
    # Initialize loss function (boundary_weight=0.1 from config default)
    loss_fn = HybridLoss(boundary_weight=0.1)
    
    # Dummy data
    B, L, D = 2, 512, 128
    predicted_noise = torch.randn(B, L, D)
    target_noise = torch.randn(B, L, D)
    
    # Normalized timesteps: high t (0.9), low t (0.1)
    timesteps = torch.tensor([0.9, 0.1])
    
    # 1. Test Denoising Loss with SNR weighting
    print("\n1. Testing SNR Weighting...")
    loss_with_t = loss_fn.calculate_loss(predicted_noise, target_noise, timesteps=timesteps)
    loss_without_t = loss_fn.calculate_loss(predicted_noise, target_noise)
    
    print(f"Loss with SNR weights: {loss_with_t.item():.4f}")
    print(f"Loss without weights: {loss_without_t.item():.4f}")
    
    # 2. Test Boundary Coherence Loss with Overlap and Cosine Similarity
    print("\n2. Testing Boundary Coherence...")
    overlap_size = 64
    tile_a_overlap = torch.randn(B, overlap_size, D)
    # tile_b_overlap is slightly different
    tile_b_overlap = tile_a_overlap + 0.05 * torch.randn(B, overlap_size, D)
    
    boundaries = (tile_a_overlap, tile_b_overlap)
    
    total_loss = loss_fn.calculate_loss(predicted_noise, target_noise, timesteps=timesteps, tile_boundaries=boundaries)
    print(f"Total Loss (Denoising + Boundary): {total_loss.item():.4f}")
    
    # Test identical overlaps
    boundaries_identical = (tile_a_overlap, tile_a_overlap)
    boundary_loss_zero = loss_fn._calculate_boundary_discontinuity(boundaries_identical)
    print(f"Boundary Loss for identical overlaps: {boundary_loss_zero.item():.4f}")
    
    assert boundary_loss_zero < 1e-6, "Boundary loss should be zero for identical overlaps"
    
    # Test orthogonal overlaps (cosine loss should be high)
    tile_a_ortho = torch.zeros(B, overlap_size, D)
    tile_a_ortho[:, :, 0] = 1.0 # vectors in direction 0
    tile_b_ortho = torch.zeros(B, overlap_size, D)
    tile_b_ortho[:, :, 1] = 1.0 # vectors in direction 1 (orthogonal)
    
    boundary_loss_ortho = loss_fn._calculate_boundary_discontinuity((tile_a_ortho, tile_b_ortho))
    print(f"Boundary Loss for orthogonal overlaps: {boundary_loss_ortho.item():.4f}")
    # MSE = mean((1-0)^2 + (0-1)^2) = 2/D? No, mean of all elements.
    # Cosine = mean(1 - 0) = 1.0
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_hybrid_loss()
