"""
Task 5 repackaging:
Given extracted passages under <agent_dir>/passages/*.jsonl, read corresponding paragraph
embeddings from a shard pkl directory, write a compact embeddings pkl + meta jsonl, then
build a paper-level FAISS IndexFlat.

Embedding store assumption (same as notebook):
- There exists one embeddings pkl per shard: <embeddings_dir>/<chunk_name>*.pkl
- Each record is a pickled numpy array, fixed serialized size (record_size_bytes)
- Passage record includes {"id": <paragraph_idx>, "raw_id": "...", "text": "..."}
"""

from __future__ import annotations

import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .faiss_build import build_indexflat_from_pkl


DEFAULT_RECORD_SIZE_BYTES = 1664  # PICKLED_VECTOR_BYTES in notebook

def find_chunk_embedding_pkl(embeddings_dir: Path, chunk_name: str) -> Path:
    candidates = [
        embeddings_dir / f"{chunk_name}.pkl",
        embeddings_dir / f"{chunk_name}.embeddings.pkl",
        embeddings_dir / f"{chunk_name}.embeddings.f16.pkl",
        embeddings_dir / f"{chunk_name}.paragraph_embeddings.pkl",
        embeddings_dir / f"{chunk_name}.paragraphs.embeddings.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(embeddings_dir.glob(f"{chunk_name}*.pkl"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No embedding pkl found for chunk {chunk_name!r} in {embeddings_dir}")


def load_vector_at_index(fin, paragraph_idx: int, record_size_bytes: int):
    offset = paragraph_idx * record_size_bytes
    fin.seek(offset)
    payload = fin.read(record_size_bytes)
    if len(payload) != record_size_bytes:
        raise EOFError(
            f"Cannot read vector at index {paragraph_idx}: wanted {record_size_bytes} bytes, got {len(payload)} bytes"
        )
    import numpy as np  # local import (keeps module import light)

    return np.asarray(pickle.loads(payload))


def iter_jsonl_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def extract_chunk_embeddings(
    passages_path: Path,
    embeddings_dir: Path,
    record_size_bytes: int,
    paper_by_raw_id: Optional[Dict[str, dict]] = None,
) -> Tuple[str, List[object], List[dict], dict, List[str]]:
    import numpy as np  # local import

    chunk_name = passages_path.stem
    vectors: List[np.ndarray] = []
    metas: List[dict] = []
    messages: List[str] = []
    missing = 0
    skipped = 0

    try:
        source_pkl = find_chunk_embedding_pkl(embeddings_dir, chunk_name)
    except FileNotFoundError as exc:
        missing = sum(1 for _ in iter_jsonl_records(passages_path))
        return chunk_name, vectors, metas, {"saved": 0, "missing": missing, "skipped": 0}, [f"  {exc}"]

    with source_pkl.open("rb") as fin:
        for rec in iter_jsonl_records(passages_path):
            paragraph_id = rec.get("id")
            if paragraph_id is None:
                skipped += 1
                continue

            try:
                paragraph_idx = int(paragraph_id)
                vec = load_vector_at_index(fin, paragraph_idx, record_size_bytes)
            except Exception as exc:
                skipped += 1
                messages.append(f"  Skip {chunk_name} id={paragraph_id}: {exc}")
                continue

            vectors.append(vec)
            rid = str(rec.get("raw_id") or "")
            meta = {
                "chunk": chunk_name,
                "text": rec.get("text"),
                "id": paragraph_idx,
                "raw_id": rec.get("raw_id"),
            }
            if paper_by_raw_id and rid in paper_by_raw_id:
                meta.update(paper_by_raw_id[rid])
            metas.append(meta)

    messages.append(f"  {chunk_name}: {len(vectors)} embeddings extracted")
    return chunk_name, vectors, metas, {"saved": len(vectors), "missing": missing, "skipped": skipped}, messages


def write_extracted_embeddings(
    chunk_results: List[Tuple[str, List[object], List[dict], dict, List[str]]],
    out_pkl: Path,
    out_meta: Path,
) -> None:
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as fout, out_meta.open("w", encoding="utf-8") as meta_out:
        for _, vectors, metas, _, _ in chunk_results:
            for vec, meta in zip(vectors, metas):
                pickle.dump(vec, fout, protocol=pickle.HIGHEST_PROTOCOL)
                meta_out.write(json.dumps(meta, ensure_ascii=False) + "\n")


def extract_embeddings_for_passages_dir(
    passages_dir: Path,
    embeddings_dir: Path,
    out_pkl: Path,
    out_meta: Path,
    *,
    agent_dir: Optional[Path] = None,
    record_size_bytes: int = DEFAULT_RECORD_SIZE_BYTES,
    overwrite: bool = False,
    workers: int = 1,
) -> dict:
    if out_pkl.exists() and out_meta.exists() and not overwrite:
        meta_count = sum(1 for _ in iter_jsonl_records(out_meta))
        return {"saved": meta_count, "missing": 0, "skipped": 0}

    paper_by_raw_id: Dict[str, dict] = {}
    if agent_dir is not None:
        from .title_string_match import load_paper_metadata_by_raw_id

        paper_by_raw_id = load_paper_metadata_by_raw_id(Path(agent_dir))

    chunk_files = sorted(passages_dir.glob("*.jsonl"))
    workers_n = max(1, int(workers))
    chunk_results: List[Tuple[str, List[object], List[dict], dict, List[str]] | None] = [None] * len(chunk_files)

    if workers_n > 1:
        with ProcessPoolExecutor(max_workers=min(workers_n, len(chunk_files) or 1)) as executor:
            futures = {
                executor.submit(
                    extract_chunk_embeddings,
                    path,
                    embeddings_dir,
                    record_size_bytes,
                    paper_by_raw_id,
                ): idx
                for idx, path in enumerate(chunk_files)
            }
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                chunk_results[idx] = result
                for msg in result[4]:
                    print(msg)
    else:
        for idx, path in enumerate(chunk_files):
            result = extract_chunk_embeddings(
                path, embeddings_dir, record_size_bytes, paper_by_raw_id
            )
            chunk_results[idx] = result
            for msg in result[4]:
                print(msg)

    ordered_results = [r for r in chunk_results if r is not None]
    saved = sum(r[3]["saved"] for r in ordered_results)
    missing = sum(r[3]["missing"] for r in ordered_results)
    skipped = sum(r[3]["skipped"] for r in ordered_results)

    write_extracted_embeddings(ordered_results, out_pkl, out_meta)
    return {"saved": saved, "missing": missing, "skipped": skipped}


def extract_passage_embeddings_and_build_index(
    agent_dir: Path | str,
    *,
    embeddings_dir: Path | str,
    record_size_bytes: int = DEFAULT_RECORD_SIZE_BYTES,
    batch_size: int = 4096,
    metric: str = "ip",
    overwrite_embeddings: bool = False,
    workers: int = 1,
) -> dict:
    """
    Creates:
      <agent_dir>/passage_embeddings/passage_embeddings.pkl
      <agent_dir>/passage_embeddings/passage_embeddings.meta.jsonl
      <agent_dir>/passage_embeddings/passage_index/passage_embeddings.indexflat*.faiss
    """
    agent_dir = Path(agent_dir)
    embeddings_dir = Path(embeddings_dir)
    passages_dir = agent_dir / "passages"
    if not passages_dir.is_dir():
        raise FileNotFoundError(f"No passages directory found: {passages_dir} (run extraction first)")
    if not embeddings_dir.is_dir():
        raise FileNotFoundError(f"Embedding pkl directory not found: {embeddings_dir}")

    embedding_dir = agent_dir / "passage_embeddings"
    index_dir = embedding_dir / "passage_index"
    out_pkl = embedding_dir / "passage_embeddings.pkl"
    out_meta = embedding_dir / "passage_embeddings.meta.jsonl"

    extract_stat = extract_embeddings_for_passages_dir(
        passages_dir=passages_dir,
        embeddings_dir=embeddings_dir,
        out_pkl=out_pkl,
        out_meta=out_meta,
        agent_dir=agent_dir,
        record_size_bytes=record_size_bytes,
        overwrite=overwrite_embeddings,
        workers=workers,
    )

    if extract_stat["saved"] == 0:
        return {**extract_stat, "index": None}

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = build_indexflat_from_pkl(
        out_pkl, index_dir, batch_size=batch_size, metric=metric, overwrite=False
    )
    return {**extract_stat, "index": str(index_path)}

