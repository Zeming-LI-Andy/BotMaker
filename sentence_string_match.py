"""
Strict string matching: map paper titles to raw_id via first_sentence (100-char prefix).

Builds Aho-Corasick index over query titles, scans OpenScholar first_sentences shards,
outputs search_results.json (same layout as FAISS / BM25).

Examples:
  python sentence_string_match.py search --dir microbiome_ecology_copy
  python sentence_string_match.py search --dir clinical_disease --output search_results.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import ahocorasick
except ImportError as e:
    raise ImportError(
        "pyahocorasick is required. Install: pip install pyahocorasick"
    ) from e

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FIRST_SENT_DIR = PROJECT_ROOT / "OpenScholar-DataStore-V3" / "first_sentences"

FIRST_SENT_LEN = 100
MIN_TITLE_LEN = 20
MIN_TITLE_TOKENS = 3
REQUIRE_TITLE_PREFIX = True

GENERIC_TITLE_NORMS = {
    "introduction", "editorial", "preface", "reply", "commentary", "comment",
    "index", "abstract", "summary", "conclusion", "acknowledgments",
    "acknowledgement", "foreword", "appendix", "erratum", "corrigendum",
    "book review", "letter", "news", "report", "case report", "a case report",
    "in this issue", "the role of", "for evaluating", "answer", "intake",
}

_ws_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize(s: str) -> str:
    s = (s or "").strip().casefold()
    s = _punct_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip()
    return s


def title_raw_prefix(title: str) -> str:
    return (title or "")[:FIRST_SENT_LEN].strip()


def title_fs_key(title: str) -> str:
    return normalize(title_raw_prefix(title))


def title_tokens(tn: str) -> int:
    return len([t for t in tn.split() if t])


def is_indexable_title(tn: str) -> bool:
    if len(tn) < MIN_TITLE_LEN:
        return False
    if title_tokens(tn) < MIN_TITLE_TOKENS:
        return False
    if tn in GENERIC_TITLE_NORMS:
        return False
    return True


def shard_from_jsonl_name(filename: str) -> str:
    """raw_passages-0-of-16.first_sentences.jsonl -> raw_passages-0-of-16"""
    return Path(filename).stem.split(".")[0]


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@dataclass(frozen=True)
class StringMatchHit:
    title: str
    title_norm: str
    raw_id: str
    source_file: str
    source_line: int
    match_pos: int
    match_len: int
    first_sentence: str
    match_mode: str

    @property
    def shard(self) -> str:
        return shard_from_jsonl_name(self.source_file)


@dataclass
class TitleIndex:
    short_auto: ahocorasick.Automaton
    long_by_fs_key: Dict[str, List[Tuple[str, str]]]
    short_count: int
    long_count: int


def split_titles(titles: List[str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], int]:
    short, long_ = [], []
    seen: set[str] = set()
    skipped = 0
    for t in titles:
        tn = normalize(t)
        if not tn or tn in seen:
            continue
        seen.add(tn)
        if not is_indexable_title(tn):
            skipped += 1
            continue
        if len(t.strip()) > FIRST_SENT_LEN:
            long_.append((t, tn))
        else:
            short.append((t, tn))
    return short, long_, skipped


def build_title_index(titles: Sequence[str]) -> TitleIndex:
    short, long_, skipped = split_titles(list(titles))

    short_auto = ahocorasick.Automaton()
    for t, tn in short:
        short_auto.add_word(tn, (t, tn, len(tn), "short"))
    short_auto.make_automaton()

    long_by_fs_key: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for t, tn in long_:
        fs_key = title_fs_key(t)
        if fs_key:
            long_by_fs_key[fs_key].append((t, tn))

    print(
        f"Title index: short={len(short):,}, long={len(long_):,}, "
        f"skipped={skipped:,} (too short/generic)"
    )
    return TitleIndex(short_auto, dict(long_by_fs_key), len(short), len(long_))


def prefix_token_match(fsn: str, title_norm: str) -> bool:
    ft = fsn.split()
    tt = title_norm.split()
    if not ft or not tt or len(ft) > len(tt):
        return False
    if ft == tt[: len(ft)]:
        return True
    if len(ft) >= 2 and ft[:-1] == tt[: len(ft) - 1]:
        frag, full = ft[-1], tt[len(ft) - 1]
        return len(frag) >= 3 and full.startswith(frag)
    return False


def strict_match_score(
    fsn: str,
    title: str,
    title_norm: str,
    mode: str,
) -> Optional[Tuple[int, int, int]]:
    if mode == "long":
        fs_key = title_fs_key(title)
        if fsn == fs_key:
            return (3, len(title_norm), len(fsn))
        if prefix_token_match(fsn, title_norm):
            return (2, len(title_norm), len(fsn))
        if len(fsn) >= 80 and title_norm.startswith(fsn):
            return (1, len(fsn), 0)
        return None

    if REQUIRE_TITLE_PREFIX:
        if not fsn.startswith(title_norm):
            return None
        if len(fsn) > len(title_norm) and fsn[len(title_norm)] not in " \t":
            return None
    else:
        if title_norm not in fsn:
            return None
    return (len(title_norm), 0, 0)


def iter_strict_hits(
    fsn: str,
    index: TitleIndex,
    wanted: Optional[set[str]],
):
    if not fsn:
        return

    for title, title_norm in index.long_by_fs_key.get(fsn, ()):
        if wanted is not None and title_norm not in wanted:
            continue
        sc = strict_match_score(fsn, title, title_norm, "long")
        if sc is not None:
            yield title, title_norm, 0, len(fsn), "long", sc

    for end_idx, (title, title_norm, mlen, mode) in index.short_auto.iter(fsn):
        if wanted is not None and title_norm not in wanted:
            continue
        sc = strict_match_score(fsn, title, title_norm, "short")
        if sc is None:
            continue
        start = 0 if fsn.startswith(title_norm) else end_idx - mlen + 1
        yield title, title_norm, start, len(title_norm), "short", sc


def scan_first_sentences(
    first_sent_dir: Path,
    index: TitleIndex,
    *,
    wanted_titles_norm: Optional[set[str]] = None,
    log_every_file: bool = True,
) -> Tuple[Dict[str, StringMatchHit], dict]:
    files = sorted(first_sent_dir.glob("*.first_sentences.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *.first_sentences.jsonl under {first_sent_dir}")

    best: Dict[str, Tuple[Tuple[int, ...], StringMatchHit]] = {}
    total_lines = 0
    total_hits = 0

    for fp in files:
        if log_every_file:
            print(f"Scanning {fp.name} ...")
        with fp.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                total_lines += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                raw_id = str(rec.get("raw_id") or "").strip()
                fs = (rec.get("first_sentence") or "").strip()
                if not raw_id or not fs:
                    continue

                fsn = normalize(fs)
                for title, title_norm, pos, mlen, mode, score in iter_strict_hits(
                    fsn, index, wanted_titles_norm
                ):
                    total_hits += 1
                    m = StringMatchHit(
                        title=title,
                        title_norm=title_norm,
                        raw_id=raw_id,
                        source_file=fp.name,
                        source_line=line_idx,
                        match_pos=pos,
                        match_len=mlen,
                        first_sentence=fs,
                        match_mode=mode,
                    )
                    prev = best.get(title_norm)
                    if prev is None or score > prev[0]:
                        best[title_norm] = (score, m)

    best_matches = {tn: m for tn, (_, m) in best.items()}
    stats = {
        "files": len(files),
        "total_lines": total_lines,
        "total_hits": total_hits,
        "matched_titles": len(best_matches),
    }
    return best_matches, stats


def matches_to_search_results(
    matches: Iterable[StringMatchHit],
    *,
    dedupe: bool = True,
) -> Dict[str, List[str]]:
    """
    Same layout as FAISS / BM25 search_results.json:
      {"raw_passages-0-of-16": ["raw_id", ...], ...}
    """
    output: Dict[str, List[str]] = {}
    for m in matches:
        key = m.shard
        output.setdefault(key, [])
        if m.raw_id and (not dedupe or m.raw_id not in output[key]):
            output[key].append(m.raw_id)
    return output


def load_titles_from_search_paper(path: Path) -> List[str]:
    titles: List[str] = []
    for rec in iter_jsonl(path):
        t = (rec.get("title") or "").strip()
        if t:
            titles.append(t)
    return titles


def write_unmatched_jsonl(
    search_paper_jsonl: Path,
    matched_norms: set[str],
    out_path: Path,
) -> dict:
    unmatched_count = skipped_count = no_match_count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for rec in iter_jsonl(search_paper_jsonl):
            title = (rec.get("title") or "").strip()
            if not title:
                continue
            tn = normalize(title)
            if not tn or tn in matched_norms:
                continue
            if not is_indexable_title(tn):
                reason = "skipped_not_indexable"
                skipped_count += 1
            else:
                reason = "no_match_in_corpus"
                no_match_count += 1
            unmatched_count += 1
            out.write(
                json.dumps(
                    {
                        "paperId": rec.get("paperId"),
                        "title": title,
                        "title_norm": tn,
                        "year": rec.get("year"),
                        "reason": reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "unmatched": unmatched_count,
        "skipped": skipped_count,
        "no_match": no_match_count,
    }


def search_search_paper_dir(
    dir_path: Path,
    *,
    first_sent_dir: Path = DEFAULT_FIRST_SENT_DIR,
    output_name: str = "search_results.json",
    unmatched_name: str = "title_unmatched.jsonl",
    detail_jsonl_name: Optional[str] = None,
    write_unmatched: bool = True,
) -> Dict[str, List[str]]:
    """
    Match all titles in dir/search_paper.jsonl against first_sentences corpus.
    Writes search_results.json (shard -> raw_id list).
    """
    base = Path(dir_path)
    jsonl_path = base / "search_paper.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"search_paper.jsonl not found in {base}")

    titles = load_titles_from_search_paper(jsonl_path)
    print(f"Loaded {len(titles)} titles from {jsonl_path}")

    index = build_title_index(titles)
    wanted = {normalize(t) for t in titles if normalize(t) and is_indexable_title(normalize(t))}

    best, stats = scan_first_sentences(
        first_sent_dir,
        index,
        wanted_titles_norm=wanted,
    )
    print("Stats:", stats)
    print(f"Matched: {len(best)} / {len(wanted)} (indexable titles)")

    output = matches_to_search_results(best.values())
    out_path = base / output_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved search results -> {out_path}")

    if detail_jsonl_name:
        detail_path = base / detail_jsonl_name
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with detail_path.open("w", encoding="utf-8") as f:
            for m in best.values():
                f.write(
                    json.dumps(
                        {
                            "title": m.title,
                            "raw_id": m.raw_id,
                            "shard": m.shard,
                            "source_file": m.source_file,
                            "source_line": m.source_line,
                            "match_pos": m.match_pos,
                            "match_len": m.match_len,
                            "match_mode": m.match_mode,
                            "first_sentence": m.first_sentence,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"Saved detail matches -> {detail_path}")

    if write_unmatched:
        um_stats = write_unmatched_jsonl(
            jsonl_path,
            set(best.keys()),
            base / unmatched_name,
        )
        print(f"Saved unmatched -> {base / unmatched_name}")
        print(f"Unmatched: {um_stats['unmatched']} (skipped={um_stats['skipped']}, no_match={um_stats['no_match']})")

    return output


def _cmd_search(args: argparse.Namespace) -> None:
    search_search_paper_dir(
        Path(args.dir),
        first_sent_dir=Path(args.first_sent_dir),
        output_name=args.output,
        unmatched_name=args.unmatched,
        detail_jsonl_name=args.detail_jsonl,
        write_unmatched=not args.no_unmatched,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict string match titles to first_sentence raw_id (Aho-Corasick).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="Match search_paper.jsonl titles")
    p.add_argument("--dir", required=True, help="Dir with search_paper.jsonl")
    p.add_argument(
        "--first-sent-dir",
        default=str(DEFAULT_FIRST_SENT_DIR),
        help="Directory with *.first_sentences.jsonl",
    )
    p.add_argument(
        "--output",
        default="search_results.json",
        help="Output JSON (shard -> raw_id list)",
    )
    p.add_argument(
        "--unmatched",
        default="title_unmatched.jsonl",
        help="Unmatched titles jsonl",
    )
    p.add_argument(
        "--detail-jsonl",
        default=None,
        help="Optional per-title match detail jsonl",
    )
    p.add_argument(
        "--no-unmatched",
        action="store_true",
        help="Do not write title_unmatched.jsonl",
    )
    p.set_defaults(func=_cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
