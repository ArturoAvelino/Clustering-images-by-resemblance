"""Extract a small, row-aligned cache without loading the full dataset into RAM."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .config import StagePaths


def prepare_subset_cache(source: StagePaths, output: StagePaths, selection: Path) -> None:
    """Match exact index entries, retaining selection order and rejecting ambiguity.

    The source index is streamed once; only requested names and row numbers are
    retained. Memory-mapped arrays are accessed only at selected rows.
    """
    if source.index_path.parent.resolve() == output.index_path.parent.resolve():
        raise ValueError("Subset output_dir must differ from dino_files; preserve the full cache.")
    source_files = [source.index_path, source.emb_path, source.meta_path, source.size_path]
    for path in source_files + [selection]:
        if not path.is_file():
            raise ValueError(f"Missing subset input: {path}")
    # Also protect caches reached through symlinks or hard links.
    for destination in vars(output).values():
        if destination.exists() and any(
            destination.samefile(path) for path in source_files + [selection]
        ):
            raise ValueError(f"Subset output would overwrite an input: {destination}")

    with selection.open(encoding="utf-8") as stream:
        names = [line.strip() for line in stream if line.strip()]
    if not names:
        raise ValueError("subset_images is empty.")
    wanted = set(names)
    if len(wanted) != len(names):
        raise ValueError("subset_images contains duplicate entries.")
    meta = json.loads(source.meta_path.read_text(encoding="utf-8"))
    n, dim = int(meta["num_images"]), int(meta["embed_dim"])
    if n <= 0 or dim <= 0 or meta["dtype"] not in {"float16", "float32"}:
        raise ValueError("Invalid embedding dimensions or dtype in embeddings.json.")
    dtype = np.dtype(meta["dtype"])
    if source.emb_path.stat().st_size != n * dim * dtype.itemsize:
        raise ValueError("embeddings.dat byte size does not match embeddings.json.")
    sizes = np.load(source.size_path, mmap_mode="r", allow_pickle=False)
    if sizes.shape not in {(n,), (n, 1)}:
        raise ValueError(f"sizes.npy must contain one size per embedding; got {sizes.shape}.")

    print(f"[subset] Scanning {source.index_path} for {len(names)} requested images...", flush=True)
    found: dict[str, int] = {}
    count = 0
    with source.index_path.open(encoding="utf-8") as stream:
        for line in stream:
            name = line.strip()
            if not name:
                continue
            if name in wanted:
                if name in found:
                    raise ValueError(f"Requested image occurs more than once in source images.txt: {name}")
                found[name] = count
            count += 1
    if count != n:
        raise ValueError(
            f"Source images.txt has {count} entries, but embeddings.json reports {n}. "
            "Restore the original full images.txt in its original order."
        )
    missing = [name for name in names if name not in found]
    if missing:
        raise ValueError(
            f"{len(missing)} subset images not found in source images.txt: {missing[:5]}. "
            "Use exact entries, including any directory prefixes."
        )
    indices = np.array([found[name] for name in names], dtype=np.int64)
    embeddings = np.memmap(source.emb_path, mode="r", dtype=dtype, shape=(n, dim))
    output.index_path.parent.mkdir(parents=True, exist_ok=True)
    # Validate everything before writing; stage complete files before replacing outputs.
    with TemporaryDirectory(prefix=".subset-", dir=output.index_path.parent) as temporary:
        temp = Path(temporary)
        with (temp / "embeddings.dat").open("wb") as stream:
            for start in range(0, len(indices), 5000):
                embeddings[indices[start:start + 5000]].tofile(stream)
        np.save(temp / "sizes.npy", np.asarray(sizes[indices]))
        subset_meta = dict(meta, num_images=len(names))
        (temp / "embeddings.json").write_text(json.dumps(subset_meta, indent=2), encoding="utf-8")
        (temp / "images.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        for destination in [output.emb_path, output.size_path, output.meta_path, output.index_path]:
            (temp / destination.name).replace(destination)
    print(f"[subset] Copied {len(names)} of {n} cached feature rows to {output.index_path.parent}.")
