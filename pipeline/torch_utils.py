from __future__ import annotations

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except Exception:  # pragma: no cover - handled by ensure_deps()
    torch = None
    DataLoader = object
    Dataset = object
    transforms = None

__all__ = ["torch", "DataLoader", "Dataset", "transforms"]
