from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np

from .config import PipelineConfig
from .data import ImageDataset, build_transform
from .model_repo import _local_repo_hint, auto_model_repo
from .torch_utils import DataLoader, torch, transforms


class DINOv2Embedder:
    def __init__(self, cfg: PipelineConfig, device: str) -> None:
        self.cfg = cfg
        self.device = device
        self.model = self._load_model(cfg.model_name, device)
        self.model.eval()
        self.transform = build_transform(cfg)
        self.embed_dim = self._infer_dim()

    def _load_model(self, model_name: str, device: str):
        auto_repo = auto_model_repo(self.cfg)
        if auto_repo is not None:
            self.cfg.model_repo = auto_repo
        try:
            model = self._load_from_repo(model_name, self.cfg.model_repo)
        except Exception as exc:  # pragma: no cover - runtime-only
            if self._is_ssl_cert_error(exc):
                local_repo = _local_repo_hint()
                if local_repo is not None:
                    try:
                        model = self._load_from_repo(model_name, local_repo, source="local")
                        return model.to(device)
                    except Exception:
                        pass
            raise RuntimeError(
                "Failed to load DINOv2 from torch.hub. "
                "Ensure internet access on first run or cache the model locally. "
                "If you are behind a custom SSL proxy or your Python install lacks certificates, "
                "pass --ssl-ca-bundle /path/to/ca-bundle.pem (or set SSL_CERT_FILE) so HTTPS can verify. "
                "You can also pass --model-repo /path/to/dinov2 (local clone) "
                "or set DINOv2_REPO to a local repo path to avoid HTTPS."
            ) from exc
        return model.to(device)

    @staticmethod
    def _load_from_repo(
        model_name: str, repo: Optional[str], source: Optional[str] = None
    ):
        repo_ref, inferred_source = DINOv2Embedder._normalize_repo(repo)
        return torch.hub.load(
            repo_ref,
            model_name,
            pretrained=True,
            source=source or inferred_source,
        )

    @staticmethod
    def _normalize_repo(repo: Optional[str]) -> tuple[str, str]:
        if not repo:
            return "facebookresearch/dinov2", "github"

        repo = repo.strip()
        path_candidate = Path(repo).expanduser()
        if path_candidate.is_absolute() or repo.startswith((".", "~")):
            if path_candidate.exists():
                return str(path_candidate), "local"
            raise ValueError(f"Local repo path does not exist: {path_candidate}")

        if path_candidate.exists():
            return str(path_candidate), "local"

        if repo.startswith("git@github.com:"):
            path = repo.split("git@github.com:", 1)[1]
        elif "github.com" in repo:
            if "://" in repo:
                parsed = urlparse(repo)
                path = parsed.path
            else:
                path = repo.split("github.com", 1)[1]
        else:
            return repo, "github"

        path = path.lstrip(":/")
        parts = [part for part in path.split("/") if part]
        if len(parts) < 2:
            return repo, "github"

        owner = parts[0]
        name = parts[1].removesuffix(".git")
        ref = None
        if len(parts) >= 4 and parts[2] == "tree":
            ref = parts[3]

        repo_ref = f"{owner}/{name}"
        if ref:
            repo_ref = f"{repo_ref}:{ref}"
        return repo_ref, "github"

    @staticmethod
    def _is_ssl_cert_error(exc: Exception) -> bool:
        seen = set()
        cur: Optional[BaseException] = exc
        while cur is not None and cur not in seen:
            seen.add(cur)
            if cur.__class__.__name__ == "SSLCertVerificationError":
                return True
            if "CERTIFICATE_VERIFY_FAILED" in str(cur):
                return True
            cur = cur.__cause__ or cur.__context__
        return False

    def _infer_dim(self) -> int:
        dim = getattr(self.model, "embed_dim", None)
        if isinstance(dim, int):
            return dim
        dummy = torch.zeros(1, 3, self.cfg.img_size, self.cfg.img_size, device=self.device)
        with torch.inference_mode():
            feats = self._forward_features(dummy)
        return int(feats.shape[-1])

    def _forward_features(self, batch: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(batch)
        else:
            out = self.model(batch)
        if isinstance(out, dict):
            for key in ("x_norm_clstoken", "x_clstoken", "cls_token", "x_norm"):
                if key in out:
                    out = out[key]
                    break
        if out.ndim == 3:
            out = out[:, 0]
        return out

    def embed(self, dataset: ImageDataset, out_path: Path, meta_path: Path) -> None:
        n_samples = len(dataset)
        dtype = np.float16 if self.cfg.dtype == "float16" else np.float32
        out_path.parent.mkdir(parents=True, exist_ok=True)
        emb = np.memmap(out_path, mode="w+", dtype=dtype, shape=(n_samples, self.embed_dim))
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=self.cfg.num_workers > 0,
        )
        start = time.time()
        written = 0
        for batch_idx, images in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            with torch.inference_mode():
                feats = self._forward_features(images).detach().cpu().numpy()
            if dtype == np.float16:
                feats = feats.astype(np.float16)
            bsz = feats.shape[0]
            emb[written : written + bsz] = feats
            written += bsz
            if written % max(1000, self.cfg.batch_size * 10) == 0:
                elapsed = time.time() - start
                rate = written / max(elapsed, 1e-6)
                print(f"Embedded {written}/{n_samples} images ({rate:.1f} img/s)")
        emb.flush()
        meta = {
            "num_images": n_samples,
            "embed_dim": self.embed_dim,
            "dtype": self.cfg.dtype,
            "model_name": self.cfg.model_name,
            "img_size": self.cfg.img_size,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def resolve_device() -> str:
    if torch is None:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_deps() -> None:
    if torch is None or transforms is None:
        raise RuntimeError(
            "Missing required dependencies. Install: torch, torchvision, umap-learn, hdbscan, pillow, numpy."
        )
