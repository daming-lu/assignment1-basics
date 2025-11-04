import torch
from einops import einsum

# Example tensors
Q = torch.arange(1, 13).reshape(3, 4).float()  # shape: (3, 4) - 3 queries, d_k=4
K = torch.arange(1, 21).reshape(5, 4).float()  # shape: (5, 4) - 5 keys, d_k=4

print("Q shape:", Q.shape, "- (queries=3, d_k=4)")
print("K shape:", K.shape, "- (keys=5, d_k=4)")
print("\nQ:\n", Q)
print("\nK:\n", K)

# ===== Q.T @ K (transpose Q first) =====
# Q.T has shape (4, 3), K has shape (5, 4)
# Result: (4, 3) @ (5, 4).T = (4, 3) @ (4, 5) = (4, 5)? NO!
# Actually: (4, 3) @ (4, 5) doesn't work directly
# We need: (3, 4).T @ (5, 4).T = (4, 3) @ (4, 5) = (4, 5)
import pdb;pdb.set_trace()
result_QT_K = einsum(Q, K, '... q qk, ... k qk -> ... q k')
result_QT_K_standard = Q.T @ K

print("\n" + "="*50)
print("Q.T @ K (transpose Q, then multiply by K transpose)")
print("="*50)
print("einops notation: 'd q, k d -> q k'")
print("Result shape:", result_QT_K.shape, "- (queries=3, keys=5)")
print("\nResult:\n", result_QT_K)
print("\nStandard way (Q.T @ K.T):\n", result_QT_K_standard)
print("Are they equal?", torch.allclose(result_QT_K, result_QT_K_standard))


# ===== Q @ K.T (what we use in attention) =====
# Q has shape (3, 4), K.T has shape (4, 5)
# Result: (3, 4) @ (4, 5) = (3, 5)

result_Q_KT = einsum(Q, K, 'q d, k d -> q k')
result_Q_KT_standard = Q @ K.T

print("\n" + "="*50)
print("Q @ K.T (multiply Q by K transpose) - ATTENTION SCORES")
print("="*50)
print("einops notation: 'q d, k d -> q k'")
print("Result shape:", result_Q_KT.shape, "- (queries=3, keys=5)")
print("\nResult:\n", result_Q_KT)
print("\nStandard way (Q @ K.T):\n", result_Q_KT_standard)
print("Are they equal?", torch.allclose(result_Q_KT, result_Q_KT_standard))


# ===== Show the difference =====
print("\n" + "="*50)
print("COMPARISON: Q.T @ K vs Q @ K.T")
print("="*50)
print("\nQ.T @ K result:\n", result_QT_K)
print("\nQ @ K.T result:\n", result_Q_KT)
print("\nThey are different matrices!")
print("Both have same shape (3, 5) but different values")