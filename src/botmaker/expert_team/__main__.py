from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from botmaker.expert_team.generator import generate_expert_team  # noqa: E402
from botmaker.expert_team.llm import get_default_model  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a diverse multi-expert team for a research topic (PRISM pipeline).",
    )
    parser.add_argument(
        "research_context",
        help="Research topic context description",
    )
    parser.add_argument(
        "-n",
        "--num-experts",
        type=int,
        default=3,
        help="Number of experts to generate (default: 3)",
    )
    parser.add_argument(
        "-d",
        "--diversity",
        type=int,
        default=5,
        help="Expert diversity score 1-10 (default: 5)",
    )
    parser.add_argument(
        "--model",
        default=get_default_model(),
        help="OpenAI model for LLM stages (default: OPENAI_MODEL env or gpt-4.1)",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="auto",
        metavar="PATH",
        help="Output JSON path (default: expert_team_output/<timestamp>_team.json)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress per-stage progress messages on stderr",
    )
    args = parser.parse_args(argv)

    output_path = None
    output_dir = None
    if args.save and args.save != "auto":
        output_path = Path(args.save)
    elif args.save == "auto" or args.save is None:
        output_dir = PROJECT_ROOT / "expert_team_output"

    try:
        payload = generate_expert_team(
            args.research_context,
            args.num_experts,
            args.diversity,
            model=args.model,
            verbose=not args.quiet,
            output_path=output_path,
            output_dir=output_dir,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
