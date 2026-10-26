from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Iterator


def _yield_vectors_from_object(obj: object) -> Iterator[np.ndarray]:
    import numpy as np

    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            yield obj
        else:
            for row in obj:
                yield row
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, np.ndarray):
                if item.ndim == 1:
                    yield item
                else:
                    for row in item:
                        yield row
            else:
                yield np.asarray(item)
        return
    yield np.asarray(obj)


def iter_pkl_vectors(pkl_path: Path) -> Iterator[np.ndarray]:
    """Yield vectors from a .pkl written as multiple pickle dumps or one object."""
    with pkl_path.open("rb") as f:
        unpickler = pickle.Unpickler(f)
        while True:
            try:
                obj = unpickler.load()
            except EOFError:
                break
            for vec in _yield_vectors_from_object(obj):
                yield vec


def l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    import numpy as np

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return arr / norms


def build_indexflat_from_pkl(
    pkl_path: Path,
    out_dir: Path,
    *,
    batch_size: int = 4096,
    metric: str = "ip",
    log_every_batches: int = 10,
    overwrite: bool = False,
) -> Path:
    """
    Build a FAISS IndexFlat over vectors streamed from a .pkl file.

    - metric="ip": uses IndexFlatIP and L2-normalizes vectors (cosine-like)
    - metric="l2": uses IndexFlatL2
    """
    metric_l = metric.lower()
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("faiss is required. Install faiss-cpu (or faiss-gpu).") from exc

    if metric_l == "l2":
        out_path = out_dir / f"{pkl_path.stem}.indexflat_l2.faiss"
    elif metric_l == "ip":
        out_path = out_dir / f"{pkl_path.stem}.indexflat.faiss"
    else:
        raise ValueError(f"metric must be 'l2' or 'ip', got {metric!r}")

    if out_path.exists() and not overwrite:
        return out_path

    vectors_iter = iter_pkl_vectors(pkl_path)
    first = next(vectors_iter, None)
    if first is None:
        raise ValueError(f"No vectors found in {pkl_path}")
    import numpy as np

    first = np.asarray(first, dtype="float32")
    if first.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape {first.shape}")
    d = int(first.shape[0])

    if metric_l == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        index = faiss.IndexFlatIP(d)

    def add_batch(batch: Iterable[np.ndarray]) -> int:
        arr = np.vstack([np.asarray(v, dtype="float32") for v in batch])
        if metric_l == "ip":
            arr = l2_normalize_rows(arr)
        arr = np.ascontiguousarray(arr, dtype="float32")
        index.add(arr)
        return int(arr.shape[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    batch = [first]
    batch_id = 0

    for vec in vectors_iter:
        batch.append(vec)
        if len(batch) >= batch_size:
            batch_id += 1
            count += add_batch(batch)
            if log_every_batches and batch_id % log_every_batches == 0:
                print(f"[IndexFlat {metric_l}] batches={batch_id} vectors={count}")
            batch.clear()
    if batch:
        batch_id += 1
        count += add_batch(batch)
        print(f"[IndexFlat {metric_l}] batches={batch_id} vectors={count}")

    faiss.write_index(index, str(out_path))
    return out_path

