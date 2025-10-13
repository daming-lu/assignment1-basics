import torch
from torch import nn
from einops import einsum, rearrange


class RMSLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        
        """
        # import pdb;pdb.set_trace()
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype  
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weights = nn.Parameter(torch.empty((d_model), **factory_kwargs))
        # nn.init.trunc_normal_(self.embedding, std=1, a=-3, b=3)        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        
        """         
        # import pdb;pdb.set_trace()     
        in_dtype = x.dtype
        x = x.to(torch.float32)        
        # RMS_a = (1/self.d_model  * torch.sum(x **2) + self.eps) ** 0.5
        RMS_x = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        norm_x = x/RMS_x
                
        result =  norm_x * self.weights
        return result.to(in_dtype)