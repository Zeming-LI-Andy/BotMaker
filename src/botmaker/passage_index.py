"""
Offline raw_id -> byte-offset indexes for OpenScholar passage shards.

Repackaged from repository root `passage_index.py`.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

RAW_ID_BYTES_RE = re.compile(rb"\"raw_id\": \"([0-9]+)\"")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PASSAGES_DIR = PROJECT_ROOT / "OpenScholar-DataStore-V3" / "passages"
DEFAULT_PASSAGE_INDEX_DIR = PROJECT_ROOT / "OpenScholar-DataStore-V3" / "passage_indexes"


def index_path_for_chunk(index_dir: Path, chunk_name: str) -> Path:
    return index_dir / f"{chunk_name}.offsets.jsonl"


def list_passage_shards(passages_dir: Path) -> List[Path]:
    return sorted(passages_dir.glob("raw_passages-*-of-*.jsonl"))


def shard_name_from_path(path: Path) -> str:
    return path.stem


def build_passage_raw_id_index(
    passages_dir: Path,
    index_dir: Path,
    chunk_name: str,
    *,
    force: bool = False,
) -> dict:
    src_file = passages_dir / f"{chunk_name}.jsonl"
    out_file = index_path_for_chunk(index_dir, chunk_name)

    if not src_file.exists():
        raise FileNotFoundError(f"Passage shard not found: {src_file}")

    if out_file.exists() and not force:
        if out_file.stat().st_mtime >= src_file.stat().st_mtime:
            return {"chunk": chunk_name, "skipped": True, "index_path": str(out_file)}

    index_dir.mkdir(parents=True, exist_ok=True)
    offsets_by_id: Dict[str, List[int]] = defaultdict(list)
    total_lines = 0

    t0 = time.perf_counter()
    with src_file.open("rb") as fin:
        while True:
            offset = fin.tell()
            line = fin.readline()
            if not line:
                break
            if not line.strip():
                continue
            total_lines += 1
            match = RAW_ID_BYTES_RE.search(line)
            if match:
                offsets_by_id[match.group(1).decode("ascii")].append(offset)

    tmp_path = out_file.with_suffix(out_file.suffix + ".tmp")
    unique_ids = 0
    with tmp_path.open("w", encoding="utf-8") as fout:
        for raw_id in sorted(offsets_by_id.keys()):
            offsets = offsets_by_id[raw_id]
            if not offsets:
                continue
            unique_ids += 1
            fout.write(
                json.dumps({"raw_id": raw_id, "offsets": offsets}, ensure_ascii=False)
                + "\n"
            )
    tmp_path.replace(out_file)
    elapsed = time.perf_counter() - t0

    return {
        "chunk": chunk_name,
        "skipped": False,
        "index_path": str(out_file),
        "total_lines": total_lines,
        "unique_raw_ids": unique_ids,
        "elapsed_s": round(elapsed, 2),
    }


def _build_index_worker(args: Tuple[str, str, str, bool]) -> dict:
    chunk_name, passages_dir, index_dir, force = args
    try:
        return build_passage_raw_id_index(
            Path(passages_dir), Path(index_dir), chunk_name, force=force
        )
    except Exception as exc:
        return {"chunk": chunk_name, "error": str(exc)}


def build_all_passage_raw_id_indexes(
    passages_dir: Path | str = DEFAULT_PASSAGES_DIR,
    index_dir: Path | str = DEFAULT_PASSAGE_INDEX_DIR,
    *,
    workers: int = 1,
    force: bool = False,
    chunk_names: Optional[Sequence[str]] = None,
) -> dict:
    passages_dir = Path(passages_dir)
    index_dir = Path(index_dir)

    if chunk_names is None:
        shards = list_passage_shards(passages_dir)
        chunk_names = [shard_name_from_path(p) for p in shards]
    else:
        chunk_names = list(chunk_names)

    if not chunk_names:
        print(f"No passage shards found under {passages_dir}")
        return {}

    workers = max(1, int(workers))
    results: List[dict] = []

    if workers == 1 or len(chunk_names) <= 1:
        for name in chunk_names:
            stat = build_passage_raw_id_index(passages_dir, index_dir, name, force=force)
            results.append(stat)
            _print_build_stat(stat)
    else:
        tasks = [
            (name, str(passages_dir.resolve()), str(index_dir.resolve()), force)
            for name in chunk_names
        ]
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = [executor.submit(_build_index_worker, t) for t in tasks]
            for future in as_completed(futures):
                stat = future.result()
                results.append(stat)
                _print_build_stat(stat)

    built = sum(1 for r in results if not r.get("skipped") and not r.get("error"))
    skipped = sum(1 for r in results if r.get("skipped"))
    errors = [r for r in results if r.get("error")]
    return {"shards": results, "built": built, "skipped": skipped, "errors": errors}


def _print_build_stat(stat: dict) -> None:
    chunk = stat.get("chunk", "?")
    if stat.get("error"):
        print(f"  {chunk}: ERROR — {stat['error']}")
    elif stat.get("skipped"):
        print(f"  {chunk}: skip (up-to-date) -> {stat.get('index_path')}")
    else:
        print(
            f"  {chunk}: {stat.get('total_lines', 0):,} lines, "
            f"{stat.get('unique_raw_ids', 0):,} raw_ids, "
            f"{stat.get('elapsed_s', 0)}s -> {stat.get('index_path')}"
        )


def load_target_offsets(index_file: Path, target_ids: Set[str]) -> Dict[str, List[int]]:
    hits: Dict[str, List[int]] = {}
    if not index_file.exists() or not target_ids:
        return hits

    with index_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(rec.get("raw_id") or "")
            if rid in target_ids:
                offsets = rec.get("offsets") or []
                hits[rid] = [int(o) for o in offsets]
    return hits


def _scan_passages_for_ids(src_file: Path, target_ids: Set[str]) -> List[bytes]:
    lines: List[bytes] = []
    with src_file.open("rb") as fin:
        for line in fin:
            if not line.strip():
                continue
            match = RAW_ID_BYTES_RE.search(line)
            if match and match.group(1).decode("ascii") in target_ids:
                lines.append(line)
    return lines


def _extract_chunk_passages(
    chunk_name: str,
    raw_ids: List[str],
    passages_dir: Path,
    out_dir: Path,
    *,
    index_dir: Optional[Path] = DEFAULT_PASSAGE_INDEX_DIR,
    use_index: bool = True,
    overwrite: bool = False,
) -> Tuple[str, int, int, Optional[str]]:
    target_ids: Set[str] = {rid for rid in raw_ids if rid}
    if not target_ids:
        return chunk_name, 0, 0, None

    src_file = passages_dir / f"{chunk_name}.jsonl"
    out_file = out_dir / f"{chunk_name}.jsonl"

    if out_file.exists() and not overwrite:
        return chunk_name, 0, len(target_ids), f"  Skip existing output -> {out_file}"

    if not src_file.exists():
        return (
            chunk_name,
            0,
            len(target_ids),
            f"  Source chunk not found: {src_file} — skipping {len(target_ids)} raw_id(s)",
        )

    index_file = index_path_for_chunk(index_dir, chunk_name) if index_dir else None
    can_use_index = bool(use_index and index_file and index_file.exists())
    mode = "index" if can_use_index else "scan"

    saved_lines = 0
    if can_use_index:
        offset_map = load_target_offsets(index_file, target_ids)
        missing_ids = target_ids - set(offset_map.keys())
        with src_file.open("rb") as fin, out_file.open("wb") as fout:
            for rid in sorted(offset_map.keys()):
                for offset in offset_map[rid]:
                    fin.seek(offset)
                    line = fin.readline()
                    if line:
                        fout.write(line)
                        saved_lines += 1
        if missing_ids:
            extra = _scan_passages_for_ids(src_file, missing_ids)
            if extra:
                with out_file.open("ab") as fout:
                    for line in extra:
                        fout.write(line)
                        saved_lines += 1
    else:
        with src_file.open("r", encoding="utf-8") as fin, out_file.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("raw_id") in target_ids:
                    fout.write(line)
                    saved_lines += 1

    return chunk_name, saved_lines, len(target_ids), f"  {chunk_name}: {saved_lines} lines written ({mode})"


def _extract_worker(args: Tuple) -> Tuple[str, int, int, Optional[str]]:
    (
        chunk_name,
        raw_ids,
        passages_dir,
        out_dir,
        index_dir,
        use_index,
        overwrite,
    ) = args
    return _extract_chunk_passages(
        chunk_name,
        raw_ids,
        Path(passages_dir),
        Path(out_dir),
        index_dir=Path(index_dir) if index_dir else None,
        use_index=use_index,
        overwrite=overwrite,
    )


def extract_passages_from_search_results(
    agent_dir: Path | str,
    *,
    passages_dir: Path | str = DEFAULT_PASSAGES_DIR,
    index_dir: Path | str | None = DEFAULT_PASSAGE_INDEX_DIR,
    use_index: bool = True,
    workers: int = 1,
    overwrite: bool = False,
    search_results_name: str = "search_results.json",
) -> dict:
    """
    Extract all passage lines for raw_ids listed in <agent_dir>/<search_results_name>.

    Output:
      <agent_dir>/passages/<shard>.jsonl
    """
    agent_dir = Path(agent_dir)
    passages_dir = Path(passages_dir)
    index_dir_p = Path(index_dir) if index_dir is not None else None

    sr_path = agent_dir / search_results_name
    if not sr_path.exists():
        raise FileNotFoundError(f"search_results.json not found: {sr_path}")
    if not passages_dir.is_dir():
        raise FileNotFoundError(f"Passages directory not found: {passages_dir}")

    out_dir = agent_dir / "passages"
    out_dir.mkdir(parents=True, exist_ok=True)

    with sr_path.open("r", encoding="utf-8") as f:
        chunk_map: Dict[str, List[str]] = json.load(f)
    if not chunk_map:
        return {"total": 0, "saved": 0}

    workers_n = max(1, int(workers))
    dir_stat = {"total": 0, "saved": 0}

    if workers_n > 1 and len(chunk_map) > 1:
        tasks = [
            (
                chunk_name,
                raw_ids,
                str(passages_dir.resolve()),
                str(out_dir.resolve()),
                str(index_dir_p.resolve()) if index_dir_p else "",
                use_index,
                overwrite,
            )
            for chunk_name, raw_ids in chunk_map.items()
        ]
        with ProcessPoolExecutor(max_workers=min(workers_n, len(tasks))) as executor:
            futures = [executor.submit(_extract_worker, t) for t in tasks]
            for future in as_completed(futures):
                chunk_name, saved_lines, total_ids, msg = future.result()
                dir_stat["total"] += total_ids
                dir_stat["saved"] += saved_lines
                if msg:
                    print(msg)
    else:
        for chunk_name, raw_ids in chunk_map.items():
            chunk_name, saved_lines, total_ids, msg = _extract_chunk_passages(
                chunk_name,
                raw_ids,
                passages_dir,
                out_dir,
                index_dir=index_dir_p,
                use_index=use_index,
                overwrite=overwrite,
            )
            dir_stat["total"] += total_ids
            dir_stat["saved"] += saved_lines
            if msg:
                print(msg)

    return dir_stat

