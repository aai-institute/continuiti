import os
import torch

# Device
device = torch.device("cpu")

# If we are not running on GitHub Actions, we choose MPS or CUDA
if os.getenv("GITHUB_ACTIONS") != "true":
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")
