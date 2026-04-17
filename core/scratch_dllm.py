from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    HAS_TE = True
except ImportError:
    HAS_TE = False


try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    print("⚡ Thunder: Standardizing on high-perf FlashAttention backend.")
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    print("⚡ Thunder: Standardizing on explicit flash_attn backend (FA3 ready).")
except ImportError:
    HAS_FLASH_ATTN = False

def build_bidirectional_attention_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Expands a [B, L] padding mask into the broadcastable bool mask expected by
    scaled_dot_product_attention. There is no causal triangle here: every valid
    token can attend to every other valid token.
    """
    if attention_mask is None:
        return None
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must have shape [batch, seq_len]")
    return attention_mask[:, None, None, :].bool()


@dataclass
class ScratchDLMConfig:
    vocab_size: int
    embedding_dim: int
    latent_dim: int
    ffn_hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int              # Added for GQA support
    max_seq_len: int
    pad_token_id: int = 0
    dropout: float = 0.0
    self_conditioning: bool = True
    use_rope: bool = True
    rope_theta: float = 100000.0
    use_fp8: bool = False             # Enable FP8 via Transformer Engine

    @property
    def hidden_size(self) -> int:
        return self.embedding_dim

    def to_dict(self):
        return asdict(self)

    def estimate_parameter_count(self) -> int:
        attn = 4 * self.latent_dim * self.latent_dim
        mlp = 3 * self.latent_dim * self.ffn_hidden_size
        per_layer = attn + mlp + (4 * self.latent_dim)
        embeddings = self.vocab_size * self.embedding_dim
        # RoPE nu adauga parametri antrenabili (spre deosebire de learned positions)
        bridges = (self.embedding_dim * self.latent_dim * 2) + (self.latent_dim * self.embedding_dim)
        return int((per_layer * self.num_layers) + embeddings + bridges)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if HAS_TE:
            self.norm = te.RMSNorm(dim, eps=eps)
        else:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_TE:
            return self.norm(x)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        linear_cls = te.Linear if HAS_TE else nn.Linear
        self.gate = linear_cls(dim, hidden_dim, bias=False)
        self.up = linear_cls(dim, hidden_dim, bias=False)
        self.down = linear_cls(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_attention_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        linear_cls = te.Linear if HAS_TE else nn.Linear
        self.q_proj = linear_cls(dim, dim, bias=False)
        self.k_proj = linear_cls(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = linear_cls(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = linear_cls(dim, dim, bias=False)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x shape: [B, L, H, D]
        # cos, sin shape: [1, L, 1, D]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # Real-space rotation formula:
        # [x1] * [cos -sin] = [x1*cos - x2*sin]
        # [x2]   [sin  cos]   [x1*sin + x2*cos]
        
        # We need to interleave the results back to [B, L, H, D]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor], rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if rope_cos is not None and rope_sin is not None:
            q = self._apply_rope(q, rope_cos, rope_sin)
            k = self._apply_rope(k, rope_cos, rope_sin)

        q_fa = q  # Save [B, S, H, D] format for explicit flash_attn
        k_fa = k
        v_fa = v

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat K/V heads if using GQA for SDPA fallback
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        dropout_p = self.dropout if self.training else 0.0

        if HAS_FLASH_ATTN and q.device.type == "cuda" and q.dtype in [torch.float16, torch.bfloat16]:
            # FlashAttention uses the [B, S, H, D] structures directly and supports GQA natively.
            attn_output = flash_attn_func(q_fa, k_fa, v_fa, dropout_p=dropout_p, causal=False)
            attn_output = attn_output.contiguous().view(batch_size, seq_len, self.dim)
        else:
            attn_mask = build_bidirectional_attention_mask(attention_mask)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
            
        return self.o_proj(attn_output)


class BidirectionalDiffusionBlock(nn.Module):
    def __init__(self, dim: int, ffn_hidden_size: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attn = BidirectionalSelfAttention(dim, num_heads, num_kv_heads, dropout=dropout)
        self.ffn = SwiGLU(dim, ffn_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor], rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), attention_mask, rope_cos=rope_cos, rope_sin=rope_sin))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        return x


        return self.net(timesteps.float() / max(max_t, 1))


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps: torch.Tensor, max_t: int) -> torch.Tensor:
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)
        return self.net(timesteps.float() / max(max_t, 1))


class BidirectionalRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Instead of torch.polar, we return cos and sin directly for better compatibility
        return torch.cos(freqs).view(seq_len, -1), torch.sin(freqs).view(seq_len, -1)


class ThunderScratchDiffusionLM(nn.Module):
    """
    Lightweight from-scratch bidirectional diffusion LM backbone.
    The transformer denoises in a compressed latent space, while logits and
    clamping stay tied to the token embedding space.
    """

    def __init__(self, config: ScratchDLMConfig, diffusion_steps: int = 256):
        super().__init__()
        self.config = config
        self.diffusion_steps = diffusion_steps
        self.is_thunder_adapted = True

        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id)
        
        if config.use_rope:
            head_dim = config.latent_dim // config.num_attention_heads
            self.rope = BidirectionalRotaryEmbedding(head_dim, theta=config.rope_theta)
            self.position_embeddings = None
        else:
            self.rope = None
            self.position_embeddings = nn.Embedding(config.max_seq_len, config.latent_dim)
            
        linear_cls = te.Linear if HAS_TE else nn.Linear
        self.latent_in = linear_cls(config.embedding_dim, config.latent_dim, bias=False)
        self.self_cond_proj = linear_cls(config.embedding_dim, config.latent_dim, bias=False)
        self.timestep_embedder = TimestepEmbedder(config.latent_dim)

        self.blocks = nn.ModuleList(
            [
                BidirectionalDiffusionBlock(
                    dim=config.latent_dim,
                    ffn_hidden_size=config.ffn_hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_kv_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = RMSNorm(config.latent_dim)
        linear_cls = te.Linear if HAS_TE else nn.Linear
        self.x0_head = linear_cls(config.latent_dim, config.embedding_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.01)
        if self.position_embeddings is not None:
            nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.latent_in.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.self_cond_proj.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.x0_head.weight, mean=0.0, std=0.01)

    @property
    def device(self):
        return self.token_embeddings.weight.device

    @property
    def dtype(self):
        return self.token_embeddings.weight.dtype

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def get_input_embeddings(self):
        return self.token_embeddings

    def diffusion_forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
        **_: dict,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x_t.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds configured max_seq_len {self.config.max_seq_len}")

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=x_t.device, dtype=torch.long)

        positions = torch.arange(seq_len, device=x_t.device).unsqueeze(0)
        hidden = self.latent_in(x_t)
        
        if self.position_embeddings is not None:
            hidden = hidden + self.position_embeddings(positions)
            rope_cos, rope_sin = None, None
        else:
            cos, sin = self.rope(seq_len, x_t.device)
            # Reshape for broadcasting in attention: [1, L, 1, D]
            rope_cos = cos.view(1, seq_len, 1, -1)
            rope_sin = sin.view(1, seq_len, 1, -1)
            
        hidden = hidden + self.timestep_embedder(t, self.diffusion_steps).unsqueeze(1)

        if self_cond is not None and self.config.self_conditioning:
            hidden = hidden + self.self_cond_proj(self_cond)

        hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)

        for block in self.blocks:
            hidden = block(hidden, attention_mask, rope_cos=rope_cos, rope_sin=rope_sin)

        hidden = self.final_norm(hidden)
        x0_pred = self.x0_head(hidden)
        return x0_pred * attention_mask.unsqueeze(-1).to(x0_pred.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        return self.diffusion_forward(inputs_embeds, timesteps, attention_mask=attention_mask, self_cond=self_cond)
