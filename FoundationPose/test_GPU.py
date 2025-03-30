import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
else:
    print("CUDA is not available. Running on CPU.")


# Create a tensor directly on the GPU
tensor = torch.randn(1, 3, 256, 256, device='cuda')

# Check the device
print(tensor.device)

