from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BGE_M3_PATH = PROJECT_ROOT / "bge-m3"


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is required for expert team validation"
        ) from exc

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"bge-m3 model directory not found: {path}")
    return SentenceTransformer(str(path))


def embed_texts(
    texts: Sequence[str],
    *,
    model_path: Path | str = DEFAULT_BGE_M3_PATH,
    batch_size: int = 8,
):
    import numpy as np  # local import

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = _load_model(str(model_path))
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def cosine_similarity_matrix(embeddings):
    if embeddings.size == 0:
        import numpy as np

        return np.empty((0, 0), dtype=np.float32)
    return embeddings @ embeddings.T


def max_pairwise_cosine(texts: Sequence[str], **kwargs) -> tuple[float, int, int]:
    if len(texts) < 2:
        return 0.0, -1, -1

    embeddings = embed_texts(texts, **kwargs)
    sim = cosine_similarity_matrix(embeddings)
    n = sim.shape[0]
    best = -1.0
    best_i, best_j = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            value = float(sim[i, j])
            if value > best:
                best = value
                best_i, best_j = i, j
    return best, best_i, best_j
