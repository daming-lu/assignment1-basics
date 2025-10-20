import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool

class Attention(nn.Module):
    def __init__(self, d_k: int, d_v: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        
        # Linear projections for Q, K, V for all heads
        self.W_Q = nn.Linear(d_v, d_k, bias=False)
        self.W_K = nn.Linear(d_v, d_k, bias=False)
        self.W_V = nn.Linear(d_v, d_v, bias=False)
        
        # Output linear projection
        self.W_O = nn.Linear(d_v, d_v, bias=False)
        
    def forward(
        self, 
        Q: Float[Tensor, "batch seq_len d_model"], 
        K: Float[Tensor, "batch seq_len d_model"], 
        V: Float[Tensor, "batch seq_len d_model"], 
        mask: Bool[Tensor, "batch seq_len seq_len"] | None = None
    ) -> Float[Tensor, "batch seq_len d_model"]:
        batch_size, seq_len, _ = Q.shape
        
        # Linear projections and split into heads
        Q = self.W_Q(Q).view(batch_size, seq_len, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, seq_len, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, seq_len, self.d_v).transpose(1, 2)
        
        # Apply scaled dot product attention
        attn_output = run_scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply output linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_v)
        output = self.W_O(attn_output)
        
        return output