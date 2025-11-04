import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from einops import einsum, rearrange
from cs336_basics.attention import Attention


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention module.
    
    Args:
        d_model: Dimensionality of the Transformer block inputs
        num_heads: Number of heads to use in multi-head self-attention
    """
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # d_k = d_v = d_model / num_heads
        self.d_v = d_model // num_heads
        
        # Learnable projection matrices
        # W_Q, W_K, W_V: (h * d_k) x d_model
        self.W_Q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, num_heads * self.d_v, bias=False)
        
        # W_O: d_model x (h * d_v)
        self.W_O = nn.Linear(num_heads * self.d_v, d_model, bias=False)
        
        # Initialize the Attention module
        self.attention = Attention(self.d_k, self.d_v)
    
    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"]
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Apply causal multi-head self-attention.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
        
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # Get shape information
        *batch_dims, seq_len, d_model = x.shape
        
        # Apply projections: (..., seq_len, d_model) -> (..., seq_len, h * d_k)
        Q = self.W_Q(x)  # (..., seq_len, num_heads * d_k)
        K = self.W_K(x)  # (..., seq_len, num_heads * d_k)
        V = self.W_V(x)  # (..., seq_len, num_heads * d_v)
        
        # Reshape to separate heads: (..., seq_len, num_heads, d_k)
        # Then rearrange to: (..., num_heads, seq_len, d_k)
        Q = rearrange(Q, '... s (h d) -> ... h s d', h=self.num_heads)
        K = rearrange(K, '... s (h d) -> ... h s d', h=self.num_heads)
        V = rearrange(V, '... s (h d) -> ... h s d', h=self.num_heads)
        
        # Create causal mask: lower triangular matrix
        # mask[i, j] = True if i >= j (token i can attend to token j)
        # Shape: (seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        
        # Apply scaled dot-product attention with causal mask
        # Q, K, V: (..., num_heads, seq_len, d_k/d_v)
        # attn_output: (..., num_heads, seq_len, d_v)
        attn_output = self.attention(Q, K, V, mask=causal_mask)
        
        # Concatenate heads: (..., num_heads, seq_len, d_v) -> (..., seq_len, num_heads * d_v)
        attn_output = rearrange(attn_output, '... h s d -> ... s (h d)')
        
        # Apply output projection: (..., seq_len, num_heads * d_v) -> (..., seq_len, d_model)
        output = self.W_O(attn_output)
        
        return output    