import torch

# 1. Create three tensors of the same size (2 rows, 3 columns)
t1 = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])

t2 = torch.tensor([[7, 8, 9],
                   [10, 11, 12]])
                   
t3 = torch.tensor([[13, 14, 15],
                   [16, 17, 18]])

# The list of tensors to stack
tensors = [t1, t2, t3]

print(tensors)
print([t.shape for t in tensors])
print('\n---\n')
# 2. Stack the tensors along a new dimension (dim=0)
stacked_tensor = torch.stack(tensors, dim=1)

print(stacked_tensor)
print(stacked_tensor.shape)
print('stacked_tensor[0][0]', stacked_tensor[0][0])


