import torch
from torch import nn
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Constructs a linear transformation module without bias.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            device (torch.device | None): Device to store the parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation to the input: y = x @ W.T

        Args:
            x (torch.Tensor): Input tensor of shape (... , in_features)

        Returns:
            torch.Tensor: Output tensor of shape (... , out_features)
        """

        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
        
if __name__ == '__main__':
    linear = Linear(32, 64)
    x = torch.randn(1, 32)
    print(linear(x).shape)