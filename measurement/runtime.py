"""Shared runtime choices for measurement tools."""
import os
from pathlib import Path


def resolve_project_root(script_file, env=None):
    """Resolve the active checkout without a retired-host path."""
    values = os.environ if env is None else env
    return values.get(
        "ANIMA_LAB_ROOT",
        str(Path(script_file).resolve().parent.parent),
    )


def resolve_torch_device(torch):
    """Prefer CUDA, then Apple's native Metal backend, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
