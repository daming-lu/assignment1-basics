import torch
from torch import nn
from einops import einsum, rearrange
from cs336_basics.linear import Linear
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SwiGLU(nn.Module):
    def __init__(self, w1_weight, w2_weight, w3_weight, d_model, d_ff):
        """

        
        """
        # import pdb;pdb.set_trace()
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        factory_kwargs = {"device": device}

        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)

        
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        """
        
        """
        # Project to hidden (d_ff)
        x1 = einsum(in_features, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        x3 = einsum(in_features, self.w3, "... d_model, d_ff d_model -> ... d_ff")

        # Apply SwiGLU activation
        x = torch.nn.functional.silu(x3) * x1

        # Project back to model dimension
        out = einsum(in_features, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        return out
    
