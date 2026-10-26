from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .passage_index import DEFAULT_PASSAGE_INDEX_DIR, DEFAULT_PASSAGES_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIRST_SENT_DIR = PROJECT_ROOT / "OpenScholar-DataStore-V3" / "first_sentences"
DEFAULT_PARAGRAPH_EMBEDDINGS_DIR = PROJECT_ROOT / "OpenScholar-DataStore-V3" / "passage_embeddings"


@dataclass
class PipelineConfig:
    expert_name: str
    expert_description: str = ""
    output_root: Path = Path(".")

    # Step 1 — Semantic Scholar
    start_query: Optional[str] = None
    openai_model: str = "gpt-4.1"
    max_queries: int = 10
    min_papers: int = 100_000
    max_papers: int = 1_000_000

    # Step 2 — string match
    first_sent_dir: Path = DEFAULT_FIRST_SENT_DIR

    # Step 3 — passage extract
    passages_dir: Path = DEFAULT_PASSAGES_DIR
    passage_index_dir: Path = DEFAULT_PASSAGE_INDEX_DIR
    extract_workers: int = 4
    extract_overwrite: bool = False
    use_passage_index: bool = True

    # Step 3b — optional index build (only used by `all`)
    build_passage_index: bool = False
    passage_index_force: bool = False

    # Step 4 — embeddings + FAISS
    paragraph_embeddings_dir: Path = DEFAULT_PARAGRAPH_EMBEDDINGS_DIR
    record_size_bytes: int = 1664
    embedding_workers: int = 1
    faiss_batch_size: int = 4096
    faiss_metric: str = "ip"
    overwrite_embeddings: bool = False


def agent_dir(cfg: PipelineConfig) -> Path:
    return Path(cfg.output_root).resolve() / cfg.expert_name


def run_search(cfg: PipelineConfig) -> dict:
    """Step 1: Semantic Scholar bulk search -> search_paper.jsonl"""
    from .semantic_scholar_bulk import search_papers

    if not cfg.expert_description:
        raise ValueError("expert_description is required for search")

    path = search_papers(
        expert_name=cfg.expert_name,
        expert_description=cfg.expert_description,
        output_path=cfg.output_root,
        start_query=cfg.start_query,
        model=cfg.openai_model,
        max_queries=cfg.max_queries,
        min_papers_per_query=cfg.min_papers,
        max_papers_per_query=cfg.max_papers,
    )
    return {
        "step": "search",
        "agent_dir": str(agent_dir(cfg)),
        "search_paper_jsonl": str(path),
    }


def run_match(cfg: PipelineConfig) -> dict:
    """Step 2: title string match -> search_results.json"""
    from .title_string_match import match_titles_in_dir

    path = match_titles_in_dir(
        agent_dir(cfg),
        first_sent_dir=cfg.first_sent_dir,
        output_name="search_results.json",
    )
    return {
        "step": "match",
        "agent_dir": str(agent_dir(cfg)),
        "search_results_json": str(path),
    }


def run_build_passage_index(cfg: PipelineConfig) -> dict:
    """Step 3a (optional, global): build raw_id -> byte-offset indexes for all shards."""
    from .passage_index import build_all_passage_raw_id_indexes

    result = build_all_passage_raw_id_indexes(
        cfg.passages_dir,
        cfg.passage_index_dir,
        workers=max(1, cfg.extract_workers),
        force=cfg.passage_index_force,
    )
    return {"step": "build-index", **result}


def run_extract(cfg: PipelineConfig) -> dict:
    """Step 3b: extract full passages -> passages/*.jsonl"""
    from .passage_index import extract_passages_from_search_results

    stats = extract_passages_from_search_results(
        agent_dir(cfg),
        passages_dir=cfg.passages_dir,
        index_dir=cfg.passage_index_dir,
        use_index=cfg.use_passage_index,
        workers=max(1, cfg.extract_workers),
        overwrite=cfg.extract_overwrite,
        search_results_name="search_results.json",
    )
    return {
        "step": "extract",
        "agent_dir": str(agent_dir(cfg)),
        "extract": stats,
    }


def run_embeddings(cfg: PipelineConfig) -> dict:
    """Step 4: extract paragraph embeddings + build FAISS IndexFlat."""
    from .passage_embeddings import extract_passage_embeddings_and_build_index

    stats = extract_passage_embeddings_and_build_index(
        agent_dir(cfg),
        embeddings_dir=cfg.paragraph_embeddings_dir,
        record_size_bytes=cfg.record_size_bytes,
        batch_size=cfg.faiss_batch_size,
        metric=cfg.faiss_metric,
        overwrite_embeddings=cfg.overwrite_embeddings,
        workers=max(1, cfg.embedding_workers),
    )
    return {
        "step": "embeddings",
        "agent_dir": str(agent_dir(cfg)),
        "passage_embeddings": stats,
    }


def run_pipeline(cfg: PipelineConfig) -> dict:
    """Run all steps in order (optional passage index build before extract)."""
    cfg.output_root = Path(cfg.output_root)
    agent_dir(cfg).mkdir(parents=True, exist_ok=True)

    out: dict = {"agent_dir": str(agent_dir(cfg)), "steps": {}}

    # out["steps"]["search"] = run_search(cfg)
    out["steps"]["search"] = {}
    out["steps"]["match"] = run_match(cfg)

    if cfg.build_passage_index:
        out["steps"]["build-index"] = run_build_passage_index(cfg)

    out["steps"]["extract"] = run_extract(cfg)
    out["steps"]["embeddings"] = run_embeddings(cfg)
    return out
