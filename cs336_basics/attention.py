import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from cs336_basics.soft_max import calc_softmax
from einops import einsum, rearrange

class Attention(nn.Module):
    """
    Scaled Dot-Product Attention module.
    
    Args:
        d_k: Dimension of key/query vectors
        d_v: Dimension of value vectors
    """
    
    def __init__(self, d_k: int, d_v: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.scale = d_k ** 0.5
    
    def forward(
        self,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_v"],
        mask: Float[Tensor, "... queries keys"] | None = None,
    ) -> Float[Tensor, "... queries d_v"]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor of shape (..., queries, d_k)
            K: Key tensor of shape (..., keys, d_k)
            V: Value tensor of shape (..., values, d_v)
            mask: Optional mask tensor of shape (..., queries, keys)
                  True values indicate positions to mask (set to -inf before softmax)
        
        Returns:
            Attention output of shape (..., queries, d_v)
        """
        # import pdb;pdb.set_trace()
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # Q: (..., queries, d_k), K: (..., keys, d_k)
        # scores: (..., queries, keys)
        # scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        x1 = einsum(Q, K, "... q_k d_k, ... d_ff d_k -> ... q_k d_ff")
        x1 /= self.scale
        
        # Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            x1 = x1.masked_fill(~mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = calc_softmax(x1, dim=-1)
        
        # Compute weighted sum of values
        # attn_weights: (..., queries, keys), V: (..., keys, d_v)
        # output: (..., queries, d_v)
        output = einsum(attn_weights, V, "... d_a d_k, ... d_k d_v -> ... d_a d_v")
        
        return output


# def run_scaled_dot_product_attention(
#     Q: Float[Tensor, "... queries d_k"],
#     K: Float[Tensor, "... keys d_k"],
#     V: Float[Tensor, "... values d_v"],
#     mask: Float[Tensor, "... queries keys"] | None = None,
# ) -> Float[Tensor, "... queries d_v"]:
#     """
#     Given key (K), query (Q), and value (V) tensors, return
#     the output of your scaled dot product attention implementation.

#     Args:
#         Q: Query tensor
#         K: Key tensor
#         V: Values tensor
#         mask: Mask tensor (True values will be masked)
#     Returns:
#         Output of SDPA
#     """
#     d_k = Q.shape[-1]
#     d_v = V.shape[-1]
#     attention = Attention(d_k, d_v)
#     return attention(Q=Q, K=K, V=V, mask=mask)