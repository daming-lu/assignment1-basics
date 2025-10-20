import torch

# Create two tensors with different dimensions (2x3 and 2x3)
tensor_a = torch.tensor([[1, 2, 3], 
                         [4, 5, 6]])
tensor_b = torch.tensor([[7, 8, 9], 
                         [10, 11, 12]])

print("Original tensors:")
print("tensor_a:\n", tensor_a)
print("tensor_b:\n", tensor_b)
print("Shapes:", tensor_a.shape, tensor_b.shape)

concat0 = torch.concat([tensor_a, tensor_b], dim=0)
print("concat0:\n", concat0)
print("Shape:", concat0.shape)

concat1 = torch.concat([tensor_a, tensor_b], dim=1)
print("concat1:\n", concat1)
print("Shape:", concat1.shape)

stack0 = torch.stack([tensor_a, tensor_b], dim=0)
print("stack0:\n", stack0)
print("Shape:", stack0.shape)

stack1 = torch.stack([tensor_a, tensor_b], dim=1)
print("stack1:\n", stack1)
print("Shape:", stack1.shape)
