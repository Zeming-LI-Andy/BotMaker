#!/usr/bin/env python3
"""Run BotMaker pipeline for every expert in an expert_team_output JSON file.

Example (from repo root):

    python -m src.run_team_pipeline expert_team_output/metabolites_bgc_d8.json

Example (from src/):

    python run_team_pipeline.py expert_team_output/metabolites_bgc_d8.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_ROOT.parent
_DEFAULT_TEAM_DIR = _SRC_ROOT / "expert_team_output"
_DEFAULT_EMBEDDINGS_DIR = Path(r"C:\Projects\Biomaker_tmpdata\passage_embeddings")


def _resolve_team_path(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    candidate = _DEFAULT_TEAM_DIR / path
    if candidate.is_file():
        return candidate.resolve()
    candidate = _SRC_ROOT / path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Team JSON not found: {path}")


def _load_team(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    team = data.get("team")
    if not isinstance(team, dict) or not team:
        raise ValueError(f"No non-empty 'team' object in {path}")
    for key, expert in team.items():
        if not isinstance(expert, dict):
            raise ValueError(f"Invalid expert entry for {key!r}")
        if not expert.get("expert_description", "").strip():
            raise ValueError(f"Missing expert_description for {key!r}")
    return team


def _build_cmd(
    *,
    expert_name: str,
    expert_description: str,
    openai_model: str,
    paragraph_embeddings_dir: Path,
    output_root: Path,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "all",
        "--expert-name",
        expert_name,
        "--expert-description",
        expert_description,
        "--openai-model",
        openai_model,
        "--paragraph-embeddings-dir",
        str(paragraph_embeddings_dir),
        "--output-root",
        str(output_root),
        "--max-queries",
        "6",
    ]
    cmd.extend(extra_args)
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run `python -m src.main all` for each expert in a team JSON file.",
    )
    parser.add_argument(
        "team_json",
        help="Path to expert team JSON (under src/expert_team_output/ or absolute)",
    )
    parser.add_argument(
        "--openai-model",
        default="deepseek-v4-pro",
        help='OpenAI-compatible model (default: "deepseek-v4-pro")',
    )
    parser.add_argument(
        "--paragraph-embeddings-dir",
        type=Path,
        default=_DEFAULT_EMBEDDINGS_DIR,
        help="Shard embedding PKL directory for pipeline step 4",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT,
        help="Parent directory for <expert-name>/ outputs (default: repo root)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="EXPERT_KEY",
        help="Run only these team keys (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args, main_extra = parser.parse_known_args(argv)

    team_path = _resolve_team_path(args.team_json)
    team = _load_team(team_path)

    keys = list(team.keys())
    if args.only:
        missing = [k for k in args.only if k not in team]
        if missing:
            raise SystemExit(f"Unknown expert key(s): {', '.join(missing)}")
        keys = [k for k in keys if k in args.only]

    print(f"Team file: {team_path}", file=sys.stderr)
    print(f"Experts to run ({len(keys)}): {', '.join(keys)}", file=sys.stderr)

    failed: list[str] = []
    for i, name in enumerate(keys, 1):
        description = team[name]["expert_description"].strip()
        cmd = _build_cmd(
            expert_name=name,
            expert_description=description,
            openai_model=args.openai_model,
            paragraph_embeddings_dir=args.paragraph_embeddings_dir,
            output_root=args.output_root.resolve(),
            extra_args=main_extra,
        )
        print(f"\n[{i}/{len(keys)}] {name}", file=sys.stderr)
        if args.dry_run:
            print(subprocess.list2cmdline(cmd))
            continue

        result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        if result.returncode != 0:
            failed.append(name)
            print(f"FAILED: {name} (exit {result.returncode})", file=sys.stderr)

    if args.dry_run:
        return 0
    if failed:
        print(f"\nFinished with failures: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nAll experts completed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
