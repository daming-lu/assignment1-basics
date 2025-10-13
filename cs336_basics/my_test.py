import torch

channels_last = torch.randn(1, 32, 32, 3) # (batch, height, width, channel)
print('channels_last: ')
# print(channels_last)
print(channels_last[0][0][0])

B = torch.randn(32*32, 32*32)
print('B: ')
# print(B)
print(B[0][0])


## Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(
-1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)

print('channels_last_transformed: ')
print(channels_last.shape)
