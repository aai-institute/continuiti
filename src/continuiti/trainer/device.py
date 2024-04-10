"""
`continuiti.trainer.device`

Default torch device.
"""

import torch
import os
import torch.distributed as dist


def get_device() -> torch.device:
    """Get torch device.

    Defaults to `cuda` or `mps` if available, otherwise to `cpu`.

    Use the environment variable `USE_MPS_BACKEND` to disable the `mps` backend.

    Returns:
        Device.
    """
    device = torch.device("cpu")
    use_mps_backend = os.environ.get("USE_MPS_BACKEND", "True").lower() in ("true", "1")

    if use_mps_backend and torch.backends.mps.is_available():
        device = torch.device("mps")

    if torch.cuda.is_available():
        if "RANK" in os.environ:
            if not dist.is_initialized():
                dist.init_process_group("nccl")
            rank = dist.get_rank()
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cuda")

    return device
