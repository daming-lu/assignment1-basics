import torch
from torch import nn
from einops import einsum, rearrange


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module. This function should accept the following parameters:
        
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        
        """
        # import pdb;pdb.set_trace()
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        nn.init.trunc_normal_(self.embedding, std=1, a=-3, b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.        
        
        """
        # import pdb;pdb.set_trace()
        # print('self.embedding:', self.embedding.shape)
        # print('token_ids:', token_ids.shape)
        return self.embedding[token_ids]
    
