from __future__ import annotations

import os
import ssl
import sys
from pathlib import Path
from typing import Optional

from .config import PipelineConfig


def _discover_ca_bundle() -> Optional[Path]:
    env_candidates = [
        os.environ.get("SSL_CERT_FILE"),
        os.environ.get("REQUESTS_CA_BUNDLE"),
    ]
    for value in env_candidates:
        if value:
            path = Path(value).expanduser()
            if path.exists():
                return path

    try:  # Optional dependency; use if available.
        import certifi  # type: ignore

        cert_path = Path(certifi.where())
        if cert_path.exists():
            return cert_path
    except Exception:
        pass

    candidates = [
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/usr/local/etc/openssl@3/cert.pem",
        "/opt/homebrew/etc/openssl@3/cert.pem",
        "/usr/local/etc/openssl/cert.pem",
    ]
    for value in candidates:
        path = Path(value)
        if path.exists():
            return path
    return None


def configure_ssl(cfg: PipelineConfig) -> None:
    if cfg.ssl_ca_bundle is not None:
        ca_path = Path(cfg.ssl_ca_bundle).expanduser()
        if not ca_path.exists():
            raise ValueError(f"ssl_ca_bundle does not exist: {ca_path}")
    else:
        ca_path = _discover_ca_bundle()
        if ca_path is None:
            if not cfg.dino_model:
                print(
                    "Warning: No CA bundle found for HTTPS verification. "
                    "If model download fails with SSL errors, pass "
                    "--ssl-ca-bundle /path/to/ca-bundle.pem or set SSL_CERT_FILE. "
                    "Alternatively, set --dino-model to a local DINOv2 clone.",
                    file=sys.stderr,
                )
            return

    os.environ.setdefault("SSL_CERT_FILE", str(ca_path))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(ca_path))
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=str(ca_path))
