"""
Build BM25 coarse-ranking indexes over first_sentence, and search by paper title.

Data: OpenScholar-DataStore-V3/first_sentences/*.first_sentences.jsonl
  {"raw_id": "...", "first_sentence": "..."}   # first 100 chars of first paragraph

Requires conda env `huggingface-download` (bm25s installed).

Examples:
  conda run -n huggingface-download python bm25_db_construct.py build
  conda run -n huggingface-download python bm25_db_construct.py build --shard raw_passages-0-of-16.first_sentences.jsonl
  conda run -n huggingface-download python bm25_db_construct.py search --dir microbiome_ecology --top-k 1
  conda run -n huggingface-download python bm25_db_construct.py search --title "Energy security and competition"
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import bm25s

# ---------------------------------------------------------------------------
# Defaults (OpenScholar-DataStore-V3 layout)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "OpenScholar-DataStore-V3" / "first_sentences"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "OpenScholar-DataStore-V3" / "sentence_bm25_indexes"
DEFAULT_META_DIR = PROJECT_ROOT / "OpenScholar-DataStore-V3" / "sentence_meta"

INDEX_MARKER = "params.index.json"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def shard_stem(jsonl_name: str) -> str:
    """raw_passages-0-of-16.first_sentences.jsonl -> raw_passages-0-of-16"""
    return Path(jsonl_name).stem.split(".")[0]


def list_sentence_jsonls(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*.first_sentences.jsonl"))


def load_raw_id_map(meta_path: Path) -> Dict[int, str]:
    """Load full raw_id sidecar: line index (doc_id) -> raw_id."""
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    raw_id_by_idx: Dict[int, str] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                raw_id_by_idx[i] = str(json.loads(line).get("raw_id", ""))
            except json.JSONDecodeError:
                raw_id_by_idx[i] = ""
    return raw_id_by_idx


def write_raw_id_meta(meta_path: Path, raw_ids: Sequence[str]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        for rid in raw_ids:
            f.write(json.dumps({"raw_id": rid}, ensure_ascii=False) + "\n")


def load_corpus_from_jsonl(
    jsonl_path: Path,
    *,
    max_docs: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    corpus: List[str] = []
    raw_ids: List[str] = []
    for rec in iter_jsonl(jsonl_path):
        text = (rec.get("first_sentence") or "").strip()
        rid = str(rec.get("raw_id") or "").strip()
        if not text or not rid:
            continue
        corpus.append(text)
        raw_ids.append(rid)
        if max_docs is not None and len(corpus) >= max_docs:
            break
    return corpus, raw_ids


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------
def build_shard_index(
    jsonl_path: Path,
    index_dir: Path,
    meta_dir: Path,
    *,
    max_docs: Optional[int] = None,
    force: bool = False,
    show_progress: bool = True,
) -> Path:
    """
    Build one BM25 index for a single first_sentences jsonl shard.
    Saves bm25s artifacts under index_dir/<shard_stem>/ and raw_id meta alongside FAISS layout.
    """
    stem = shard_stem(jsonl_path.name)
    out_path = index_dir / stem
    marker = out_path / INDEX_MARKER
    meta_path = meta_dir / f"{stem}.first_sentences.raw_id.jsonl"

    if marker.exists() and meta_path.exists() and not force:
        print(f"Skip existing BM25 index -> {out_path}")
        return out_path

    print(f"Loading corpus from {jsonl_path.name} ...")
    corpus, raw_ids = load_corpus_from_jsonl(jsonl_path, max_docs=max_docs)
    if not corpus:
        raise ValueError(f"No documents loaded from {jsonl_path}")

    if max_docs is not None and meta_path.exists():
        raise ValueError(
            f"--max-docs is for testing only; existing meta would desync: {meta_path}"
        )

    print(f"  docs={len(corpus):,}  tokenizing ...")
    tokens = bm25s.tokenize(
        corpus,
        stopwords=None,
        stemmer=None,
        show_progress=show_progress,
    )
    retriever = bm25s.BM25()
    retriever.index(tokens)

    out_path.mkdir(parents=True, exist_ok=True)
    # corpus.jsonl stores first_sentence for retrieval display
    retriever.save(str(out_path), corpus=corpus)

    if meta_path.exists():
        print(f"Reuse existing raw_id meta -> {meta_path}")
    else:
        write_raw_id_meta(meta_path, raw_ids)
        print(f"Saved raw_id meta -> {meta_path}")
    print(f"Saved BM25 index -> {out_path}")
    return out_path


def rebuild_meta_shard(jsonl_path: Path, meta_dir: Path) -> Path:
    """Regenerate raw_id sidecar from first_sentences jsonl (line order = doc id)."""
    stem = shard_stem(jsonl_path.name)
    meta_path = meta_dir / f"{stem}.first_sentences.raw_id.jsonl"
    print(f"Writing meta for {jsonl_path.name} -> {meta_path}")
    raw_ids: List[str] = []
    for rec in iter_jsonl(jsonl_path):
        rid = str(rec.get("raw_id") or "").strip()
        if rid:
            raw_ids.append(rid)
    write_raw_id_meta(meta_path, raw_ids)
    print(f"  lines={len(raw_ids):,}")
    return meta_path


def rebuild_all_meta(input_dir: Path, meta_dir: Path, *, shards: Optional[Sequence[str]] = None) -> None:
    jsonl_files = list_sentence_jsonls(input_dir)
    if shards:
        allow = {s if s.endswith(".jsonl") else f"{s}" for s in shards}
        jsonl_files = [p for p in jsonl_files if p.name in allow]
    for p in jsonl_files:
        rebuild_meta_shard(p, meta_dir)


def build_all_indexes(
    input_dir: Path,
    index_dir: Path,
    meta_dir: Path,
    *,
    shards: Optional[Sequence[str]] = None,
    max_docs: Optional[int] = None,
    force: bool = False,
) -> List[Path]:
    jsonl_files = list_sentence_jsonls(input_dir)
    if shards:
        allow = {s if s.endswith(".jsonl") else f"{s}" for s in shards}
        jsonl_files = [p for p in jsonl_files if p.name in allow]
    if not jsonl_files:
        raise FileNotFoundError(f"No *.first_sentences.jsonl under {input_dir}")

    built: List[Path] = []
    for i, p in enumerate(jsonl_files, 1):
        print(f"\n[{i}/{len(jsonl_files)}] {p.name}")
        built.append(
            build_shard_index(
                p,
                index_dir,
                meta_dir,
                max_docs=max_docs,
                force=force,
            )
        )
    return built


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
@dataclass
class BM25Hit:
    score: float
    doc_id: int
    raw_id: str
    first_sentence: str
    shard: str
    index_dir: Path


class BM25ShardSearcher:
    def __init__(
        self,
        index_path: Path,
        meta_path: Path,
        *,
        mmap: bool = True,
    ) -> None:
        self.shard = index_path.name
        self.index_path = index_path
        self.meta_path = meta_path
        print(f"Loading BM25 index {self.shard} ...")
        self.retriever = bm25s.BM25.load(
            str(index_path),
            load_corpus=True,
            mmap=mmap,
        )

        print(f"Loading meta {meta_path.name} ...")
        self.raw_id_by_idx = load_raw_id_map(meta_path)

    def search(
        self,
        queries: Sequence[str],
        k: int,
        *,
        n_threads: int = 0,
    ) -> List[List[BM25Hit]]:
        if not queries:
            return []
        q_tokens = bm25s.tokenize(
            list(queries),
            stopwords=None,
            stemmer=None,
            show_progress=False,
        )
        t0 = time.perf_counter()
        results, scores = self.retriever.retrieve(
            q_tokens,
            k=k,
            n_threads=n_threads,
            show_progress=False,
        )
        t1 = time.perf_counter()
        print(f"Time consumed: {t1 - t0:.2f} seconds")

        all_hits: List[List[BM25Hit]] = []
        for qi in range(len(queries)):
            hits: List[BM25Hit] = []
            row = results[qi]
            row_scores = scores[qi]
            doc_ids: List[int] = []
            for item in row:
                if isinstance(item, dict):
                    doc_ids.append(int(item["id"]))
                    # text available in item["text"]
                else:
                    doc_ids.append(int(item))

            for rank, doc_id in enumerate(doc_ids):
                sc = float(row_scores[rank])
                rid = self.raw_id_by_idx.get(doc_id, "")
                fs = ""
                if self.retriever.corpus is not None:
                    c = self.retriever.corpus[doc_id]
                    fs = c["text"] if isinstance(c, dict) else str(c)
                hits.append(
                    BM25Hit(
                        score=sc,
                        doc_id=doc_id,
                        raw_id=rid,
                        first_sentence=fs,
                        shard=self.shard,
                        index_dir=self.index_path,
                    )
                )
            all_hits.append(hits)
        return all_hits


def _resolve_shard_workers(shard_workers: int, num_shards: int) -> int:
    """<=1 sequential; >1 explicit cap; 0 auto (min(shards, cpu_count, 16))."""
    if num_shards <= 1:
        return 1
    if shard_workers <= 1:
        return 1
    if shard_workers > 1:
        return min(shard_workers, num_shards)
    cpu = os.cpu_count() or 4
    return min(num_shards, cpu, 16)


def _search_shard(
    shard: BM25ShardSearcher,
    titles: Sequence[str],
    per_shard_k: int,
    n_threads: int,
) -> List[List[BM25Hit]]:
    return shard.search(titles, k=per_shard_k, n_threads=n_threads)


class MultiShardBM25Searcher:
    """Search titles across all shard indexes and merge top-k globally per query."""

    def __init__(
        self,
        index_dir: Path,
        meta_dir: Path,
        *,
        mmap: bool = True,
    ) -> None:
        self.index_dir = index_dir
        self.meta_dir = meta_dir
        self.mmap = mmap
        self._shards: List[BM25ShardSearcher] = []

    def _discover_shards(self) -> List[Tuple[Path, Path]]:
        pairs: List[Tuple[Path, Path]] = []
        for idx_path in sorted(self.index_dir.iterdir()):
            if not (idx_path / INDEX_MARKER).exists():
                continue
            meta_path = self.meta_dir / f"{idx_path.name}.first_sentences.raw_id.jsonl"
            if not meta_path.exists():
                print(f"Warning: missing meta for {idx_path.name}, skip")
                continue
            pairs.append((idx_path, meta_path))
        return pairs

    def load(self) -> None:
        pairs = self._discover_shards()
        if not pairs:
            raise FileNotFoundError(
                f"No BM25 indexes under {self.index_dir}. Run `build` first."
            )
        self._shards = [
            BM25ShardSearcher(ip, mp, mmap=self.mmap) for ip, mp in pairs
        ]
        print(f"Loaded {len(self._shards)} BM25 shard(s)")

    def search(
        self,
        titles: Sequence[str],
        *,
        final_k: int = 5,
        per_shard_k: int = 20,
        n_threads: int = 0,
        shard_workers: int = 0,
    ) -> List[List[BM25Hit]]:
        if not self._shards:
            self.load()

        merged: List[List[BM25Hit]] = [[] for _ in titles]
        workers = _resolve_shard_workers(shard_workers, len(self._shards))

        if workers <= 1:
            for shard in self._shards:
                shard_hits = shard.search(titles, k=per_shard_k, n_threads=n_threads)
                for qi, hits in enumerate(shard_hits):
                    merged[qi].extend(hits)
        else:
            print(f"Searching {len(self._shards)} shard(s) with {workers} parallel worker(s)")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(_search_shard, shard, titles, per_shard_k, n_threads)
                    for shard in self._shards
                ]
                for fut in as_completed(futures):
                    shard_hits = fut.result()
                    for qi, hits in enumerate(shard_hits):
                        merged[qi].extend(hits)

        out: List[List[BM25Hit]] = []
        for qi in range(len(titles)):
            # higher score = better
            best = heapq.nlargest(
                final_k,
                merged[qi],
                key=lambda h: h.score,
            )
            out.append(best)
        return out


def load_titles_from_search_paper(path: Path) -> List[str]:
    titles: List[str] = []
    for rec in iter_jsonl(path):
        t = (rec.get("title") or "").strip()
        if t:
            titles.append(t)
    return titles


def hits_to_search_results(
    all_hits: List[List[BM25Hit]],
    *,
    dedupe: bool = True,
) -> Dict[str, List[str]]:
    """
    Same layout as FAISS search_results.json:
      {"raw_passages-0-of-16": ["raw_id", ...], ...}
    Aggregates hits from all queries into per-shard lists.
    """
    output: Dict[str, List[str]] = {}
    for hits in all_hits:
        for h in hits:
            key = h.shard
            output.setdefault(key, [])
            if h.raw_id and (not dedupe or h.raw_id not in output[key]):
                output[key].append(h.raw_id)
    return output


def search_titles(
    titles: Sequence[str],
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
    meta_dir: Path = DEFAULT_META_DIR,
    final_k: int = 5,
    per_shard_k: int = 20,
    n_threads: int = 0,
    shard_workers: int = 0,
) -> List[List[BM25Hit]]:
    searcher = MultiShardBM25Searcher(index_dir, meta_dir)
    return searcher.search(
        titles,
        final_k=final_k,
        per_shard_k=per_shard_k,
        n_threads=n_threads,
        shard_workers=shard_workers,
    )


def search_search_paper_dir(
    dir_path: Path,
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
    meta_dir: Path = DEFAULT_META_DIR,
    final_k: int = 1,
    per_shard_k: int = 20,
    n_threads: int = 0,
    shard_workers: int = 0,
    output_name: str = "search_results_bm25.json",
) -> Dict[str, List[str]]:
    jsonl_path = dir_path / "search_paper.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"search_paper.jsonl not found in {dir_path}")

    titles = load_titles_from_search_paper(jsonl_path)
    print(f"Loaded {len(titles)} titles from {jsonl_path}")

    all_hits = search_titles(
        titles,
        index_dir=index_dir,
        meta_dir=meta_dir,
        final_k=final_k,
        per_shard_k=per_shard_k,
        n_threads=n_threads,
        shard_workers=shard_workers,
    )
    output = hits_to_search_results(all_hits)

    out_path = dir_path / output_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {out_path}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_meta(args: argparse.Namespace) -> None:
    shards = [args.shard] if args.shard else None
    rebuild_all_meta(Path(args.input_dir), Path(args.meta_dir), shards=shards)


def _cmd_build(args: argparse.Namespace) -> None:
    shards = [args.shard] if args.shard else None
    build_all_indexes(
        Path(args.input_dir),
        Path(args.index_dir),
        Path(args.meta_dir),
        shards=shards,
        max_docs=args.max_docs,
        force=args.force,
    )


def _cmd_search(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    meta_dir = Path(args.meta_dir)

    if args.dir:
        search_search_paper_dir(
            Path(args.dir),
            index_dir=index_dir,
            meta_dir=meta_dir,
            final_k=args.top_k,
            per_shard_k=args.per_shard_k,
            n_threads=args.n_threads,
            shard_workers=args.shard_workers,
            output_name=args.output,
        )
        return

    titles: List[str] = list(args.title or [])
    if args.titles_file:
        with Path(args.titles_file).open("r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    titles.append(t)
    if not titles:
        raise SystemExit("Provide --dir, --title, or --titles-file")

    all_hits = search_titles(
        titles,
        index_dir=index_dir,
        meta_dir=meta_dir,
        final_k=args.top_k,
        per_shard_k=args.per_shard_k,
        n_threads=args.n_threads,
        shard_workers=args.shard_workers,
    )
    for i, hits in enumerate(all_hits):
        print(f"\nQuery {i + 1}: {titles[i][:120]}{'...' if len(titles[i]) > 120 else ''}")
        for rank, h in enumerate(hits, 1):
            preview = h.first_sentence[:80] + ("..." if len(h.first_sentence) > 80 else "")
            print(
                f"  {rank}. score={h.score:.4f} raw_id={h.raw_id} shard={h.shard}\n"
                f"      {preview}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BM25 indexes on first_sentence and search by paper title.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build per-shard BM25 indexes")
    p_build.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory with *.first_sentences.jsonl",
    )
    p_build.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
        help="Output directory for bm25s indexes",
    )
    p_build.add_argument(
        "--meta-dir",
        default=str(DEFAULT_META_DIR),
        help="Directory for raw_id sidecar jsonl (reused if already built)",
    )
    p_build.add_argument(
        "--shard",
        default=None,
        help="Build only one shard file name, e.g. raw_passages-0-of-16.first_sentences.jsonl",
    )
    p_build.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit docs per shard (for testing)",
    )
    p_build.add_argument("--force", action="store_true", help="Rebuild even if index exists")
    p_build.set_defaults(func=_cmd_build)

    p_meta = sub.add_parser("meta", help="Regenerate raw_id sidecar jsonl from source shards")
    p_meta.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    p_meta.add_argument("--meta-dir", default=str(DEFAULT_META_DIR))
    p_meta.add_argument("--shard", default=None)
    p_meta.set_defaults(func=_cmd_meta)

    p_search = sub.add_parser("search", help="Search by title(s)")
    p_search.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p_search.add_argument("--meta-dir", default=str(DEFAULT_META_DIR))
    p_search.add_argument(
        "--dir",
        default=None,
        help="Agent dir with search_paper.jsonl; writes search_results_bm25.json",
    )
    p_search.add_argument("--title", action="append", default=None, help="Title query (repeatable)")
    p_search.add_argument("--titles-file", default=None, help="One title per line")
    p_search.add_argument("--top-k", type=int, default=1, help="Final merged top-k per query")
    p_search.add_argument(
        "--per-shard-k",
        type=int,
        default=5,
        help="Top-k retrieved from each shard before merge",
    )
    p_search.add_argument(
        "--n-threads",
        type=int,
        default=0,
        help="Threads for bm25s.retrieve per shard (0 = bm25s auto)",
    )
    p_search.add_argument(
        "--shard-workers",
        type=int,
        default=0,
        help="Parallel shards to search (0=auto, 1=sequential)",
    )
    p_search.add_argument(
        "--output",
        default="search_results_bm25.json",
        help="Output filename when using --dir",
    )
    p_search.set_defaults(func=_cmd_search)

    t0 = time.perf_counter()
    args = parser.parse_args()
    args.func(args)
    t1 = time.perf_counter()
    print(f"Time consumed: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
