import os
import torch


def configure_determinism(enabled):
    """Set the process-wide deterministic policy before CUDA initialization."""
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    if enabled:
        value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if value not in (":4096:8", ":16:8"):
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization, then restart the process"
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled
