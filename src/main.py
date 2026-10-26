from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python -m src.main` without installing the package.
_SRC_ROOT = Path(__file__).resolve().parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from botmaker.pipeline import (
    PipelineConfig,
    run_build_passage_index,
    run_embeddings,
    run_extract,
    run_match,
    run_pipeline,
    run_search,
)


def _add_common_agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--expert-name", required=True, help="Agent / expert directory name")
    p.add_argument(
        "--output-root",
        default=".",
        help="Parent directory containing <expert-name>/ outputs",
    )


def _add_datastore_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--first-sent-dir", default=None)
    p.add_argument("--passages-dir", default=None)
    p.add_argument("--passage-index-dir", default=None)


def _apply_datastore_paths(cfg: PipelineConfig, args: argparse.Namespace) -> None:
    if getattr(args, "first_sent_dir", None):
        cfg.first_sent_dir = Path(args.first_sent_dir)
    if getattr(args, "passages_dir", None):
        cfg.passages_dir = Path(args.passages_dir)
    if getattr(args, "passage_index_dir", None):
        cfg.passage_index_dir = Path(args.passage_index_dir)


def _cfg_from_agent_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        expert_name=args.expert_name,
        output_root=Path(args.output_root),
    )


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="BotMaker pipeline — run each step independently or all at once",
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ search
    p_search = sub.add_parser(
        "search",
        help="Step 1: Semantic Scholar bulk search -> search_paper.jsonl",
    )
    _add_common_agent_args(p_search)
    p_search.add_argument("--expert-description", required=True)
    p_search.add_argument("--start-query", default=None)
    p_search.add_argument("--openai-model", default="gpt-4.1")
    p_search.add_argument("--max-queries", type=int, default=10)
    p_search.add_argument("--min-papers", type=int, default=100_000)
    p_search.add_argument("--max-papers", type=int, default=1_000_000)

    # ------------------------------------------------------------------ match
    p_match = sub.add_parser(
        "match",
        help="Step 2: string match titles -> search_results.json",
    )
    _add_common_agent_args(p_match)
    _add_datastore_args(p_match)

    # ------------------------------------------------------------------ build-index
    p_idx = sub.add_parser(
        "build-index",
        help="Step 3a (global): build raw_id byte-offset indexes for passage shards",
    )
    p_idx.add_argument("--passages-dir", default=None)
    p_idx.add_argument("--passage-index-dir", default=None)
    p_idx.add_argument("--workers", type=int, default=4)
    p_idx.add_argument("--force", action="store_true")

    # ------------------------------------------------------------------ extract
    p_extract = sub.add_parser(
        "extract",
        help="Step 3b: extract full passages -> passages/*.jsonl",
    )
    _add_common_agent_args(p_extract)
    _add_datastore_args(p_extract)
    p_extract.add_argument("--workers", type=int, default=4)
    p_extract.add_argument("--overwrite", action="store_true")
    p_extract.add_argument(
        "--no-index",
        action="store_true",
        help="Full-file scan instead of offset index",
    )

    # ------------------------------------------------------------------ embeddings
    p_emb = sub.add_parser(
        "embeddings",
        help="Step 4: extract paragraph embeddings + build FAISS IndexFlat",
    )
    _add_common_agent_args(p_emb)
    p_emb.add_argument(
        "--paragraph-embeddings-dir",
        required=True,
        help="Shard embedding PKL directory",
    )
    p_emb.add_argument("--record-size-bytes", type=int, default=1664)
    p_emb.add_argument("--workers", type=int, default=1)
    p_emb.add_argument("--faiss-batch-size", type=int, default=4096)
    p_emb.add_argument("--faiss-metric", default="ip", choices=["ip", "l2"])
    p_emb.add_argument("--overwrite-embeddings", action="store_true")

    # ------------------------------------------------------------------ all
    p_all = sub.add_parser("all", help="Run steps 1-4 in sequence")
    _add_common_agent_args(p_all)
    _add_datastore_args(p_all)
    p_all.add_argument("--expert-description", required=True)
    p_all.add_argument("--start-query", default=None)
    p_all.add_argument("--openai-model", default="gpt-4.1")
    p_all.add_argument("--max-queries", type=int, default=10)
    p_all.add_argument("--min-papers", type=int, default=100_000)
    p_all.add_argument("--max-papers", type=int, default=1_000_000)
    p_all.add_argument(
        "--build-passage-index",
        action="store_true",
        help="Build offset indexes before extract (one-time per datastore)",
    )
    p_all.add_argument("--passage-index-force", action="store_true")
    p_all.add_argument("--extract-workers", type=int, default=4)
    p_all.add_argument("--extract-overwrite", action="store_true")
    p_all.add_argument(
        "--paragraph-embeddings-dir",
        required=True,
    )
    p_all.add_argument("--record-size-bytes", type=int, default=1664)
    p_all.add_argument("--embedding-workers", type=int, default=1)
    p_all.add_argument("--faiss-batch-size", type=int, default=4096)
    p_all.add_argument("--faiss-metric", default="ip", choices=["ip", "l2"])
    p_all.add_argument("--overwrite-embeddings", action="store_true")

    return root


def dispatch(args: argparse.Namespace) -> dict:
    cmd = args.command

    if cmd == "search":
        cfg = _cfg_from_agent_args(args)
        cfg.expert_description = args.expert_description
        cfg.start_query = args.start_query
        cfg.openai_model = args.openai_model
        cfg.max_queries = args.max_queries
        cfg.min_papers = args.min_papers
        cfg.max_papers = args.max_papers
        return run_search(cfg)

    if cmd == "match":
        cfg = _cfg_from_agent_args(args)
        _apply_datastore_paths(cfg, args)
        return run_match(cfg)

    if cmd == "build-index":
        cfg = PipelineConfig(expert_name="_", output_root=".")
        _apply_datastore_paths(cfg, args)
        cfg.extract_workers = args.workers
        cfg.passage_index_force = args.force
        return run_build_passage_index(cfg)

    if cmd == "extract":
        cfg = _cfg_from_agent_args(args)
        _apply_datastore_paths(cfg, args)
        cfg.extract_workers = args.workers
        cfg.extract_overwrite = args.overwrite
        cfg.use_passage_index = not args.no_index
        return run_extract(cfg)

    if cmd == "embeddings":
        cfg = _cfg_from_agent_args(args)
        cfg.paragraph_embeddings_dir = Path(args.paragraph_embeddings_dir)
        cfg.record_size_bytes = args.record_size_bytes
        cfg.embedding_workers = args.workers
        cfg.faiss_batch_size = args.faiss_batch_size
        cfg.faiss_metric = args.faiss_metric
        cfg.overwrite_embeddings = args.overwrite_embeddings
        return run_embeddings(cfg)

    if cmd == "all":
        cfg = _cfg_from_agent_args(args)
        cfg.expert_description = args.expert_description
        cfg.start_query = args.start_query
        cfg.openai_model = args.openai_model
        cfg.max_queries = args.max_queries
        cfg.min_papers = args.min_papers
        cfg.max_papers = args.max_papers
        cfg.build_passage_index = args.build_passage_index
        cfg.passage_index_force = args.passage_index_force
        cfg.extract_workers = args.extract_workers
        cfg.extract_overwrite = args.extract_overwrite
        cfg.paragraph_embeddings_dir = Path(args.paragraph_embeddings_dir)
        cfg.record_size_bytes = args.record_size_bytes
        cfg.embedding_workers = args.embedding_workers
        cfg.faiss_batch_size = args.faiss_batch_size
        cfg.faiss_metric = args.faiss_metric
        cfg.overwrite_embeddings = args.overwrite_embeddings
        _apply_datastore_paths(cfg, args)
        return run_pipeline(cfg)

    raise ValueError(f"Unknown command: {cmd}")


def main() -> None:
    args = build_parser().parse_args()
    result = dispatch(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
