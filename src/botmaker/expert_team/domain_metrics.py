from __future__ import annotations

from pathlib import Path

from .embeddings import DEFAULT_BGE_M3_PATH, cosine_similarity_matrix, embed_texts
from .models import PerspectiveSlot


def _domain_cosine(a: str, b: str, *, model_path: Path | str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    vecs = embed_texts([a.strip(), b.strip()], model_path=model_path)
    return float(cosine_similarity_matrix(vecs)[0, 1])


def compute_domain_similarity(
    slots: list[PerspectiveSlot],
    native_domain: str,
    *,
    model_path: Path | str = DEFAULT_BGE_M3_PATH,
) -> dict:
    if not slots:
        return {
            "native_domain": native_domain,
            "slots": [],
            "max_pairwise_domain_cosine": 0.0,
            "max_pair_domain": None,
        }

    keys = [s.slot_id or f"slot_{i + 1}" for i, s in enumerate(slots)]
    domains = [s.source_domain for s in slots]

    native_sims = [
        round(_domain_cosine(d, native_domain, model_path=model_path), 6) for d in domains
    ]

    n = len(slots)
    pairs: list[dict] = []
    max_dom = -1.0
    max_dom_pair = None
    if n >= 2:
        embeddings = embed_texts(domains, model_path=model_path)
        sim = cosine_similarity_matrix(embeddings)
        for i in range(n):
            for j in range(i + 1, n):
                value = round(float(sim[i, j]), 6)
                pair = {
                    "slot_a": keys[i],
                    "slot_b": keys[j],
                    "source_domain_a": domains[i],
                    "source_domain_b": domains[j],
                    "cosine_similarity": value,
                }
                pairs.append(pair)
                if value > max_dom:
                    max_dom = value
                    max_dom_pair = dict(pair)

    slot_rows = [
        {
            "slot_id": keys[i],
            "source_domain": domains[i],
            "native_domain_cosine": native_sims[i],
        }
        for i in range(n)
    ]

    return {
        "native_domain": native_domain,
        "slots": slot_rows,
        "max_pairwise_domain_cosine": round(max_dom, 6) if max_dom >= 0 else 0.0,
        "max_pair_domain": max_dom_pair,
        "pairwise_domain_pairs": pairs,
    }
