"""Shared runtime choices for measurement tools."""


def resolve_torch_device(torch):
    """Prefer CUDA, then Apple's native Metal backend, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
