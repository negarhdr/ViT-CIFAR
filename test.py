import torch
import torch.nn.functional as F

# Assuming you have a tensor of size (B * N * D)
tensor_size = (4, 10, 8)  # Example sizes
tensor = torch.ones(tensor_size)  # Example tensor
# print(tensor)
# Dropout probability

print("Original Tensor:")
print(tensor)

dropout_prob = 0.5
tensor_with_dropout = tensor
# Applying dropout independently to each token vector
# tensor_with_dropout = F.dropout(tensor, p=dropout_prob, training=True, dim=1)
# for i in range(tensor.size(1)):  # Iterate over tokens
    # tensor_with_dropout[:, i, :] = F.dropout(tensor[:, i, :], p=dropout_prob, training=True)
tensor_with_dropout = F.dropout(tensor, p=dropout_prob, training=True)
# Print the original and dropout-applied tensors
# print(tensor[:, i, :].shape)
print("\nTensor with Dropout:")
print(tensor_with_dropout)
