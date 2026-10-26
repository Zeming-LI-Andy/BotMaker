from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from .allocate import allocate_slots
from .diversity import diversity_profile
from .instantiate import instantiate_expert, instantiate_experts, members_to_team
from .llm import get_default_model, get_openai_client
from .models import DiversityProfile, ExpertTeamResult
from .parse import parse_research
from .validate import (
    DEFAULT_MAX_REFINE_ROUNDS,
    compute_expert_similarities,
    validate_and_refine,
)


def _stage_done(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr)


def default_output_path(output_dir: Path | str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_team.json"


def save_expert_team_result(payload: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def generate_expert_team(
    research_context: str,
    num_experts: int,
    diversity: int,
    *,
    model: str | None = None,
    client=None,
    max_refine_rounds: int = DEFAULT_MAX_REFINE_ROUNDS,
    bge_model_path: Path | str | None = None,
    verbose: bool = True,
    output_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict:
    result, profile = generate_expert_team_detailed(
        research_context,
        num_experts,
        diversity,
        model=model,
        client=client,
        max_refine_rounds=max_refine_rounds,
        bge_model_path=bge_model_path,
        verbose=verbose,
        return_profile=True,
    )
    payload = result.to_dict(
        research_context=research_context,
        num_experts=num_experts,
        diversity=diversity,
        profile=profile,
    )

    if output_path is not None:
        save_path = save_expert_team_result(payload, output_path)
    else:
        save_dir = Path(output_dir) if output_dir is not None else _default_output_dir()
        save_path = save_expert_team_result(payload, default_output_path(save_dir))

    if verbose:
        print(f"Saved full results to {save_path}", file=sys.stderr)

    return payload


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "expert_team_output"


def generate_expert_team_detailed(
    research_context: str,
    num_experts: int,
    diversity: int,
    *,
    model: str | None = None,
    client=None,
    max_refine_rounds: int = DEFAULT_MAX_REFINE_ROUNDS,
    bge_model_path: Path | str | None = None,
    verbose: bool = True,
    return_profile: bool = False,
) -> ExpertTeamResult | tuple[ExpertTeamResult, DiversityProfile]:
    if model is None:
        model = get_default_model()
    if not research_context or not research_context.strip():
        raise ValueError("research_context must be non-empty")
    if num_experts < 1:
        raise ValueError("num_experts must be at least 1")
    if not 1 <= diversity <= 10:
        raise ValueError("diversity must be between 1 and 10")

    if client is None:
        client = get_openai_client()

    profile = diversity_profile(diversity, num_experts)
    from .embeddings import DEFAULT_BGE_M3_PATH

    embed_path = bge_model_path or DEFAULT_BGE_M3_PATH

    structure = parse_research(
        research_context, profile, model=model, client=client
    )
    _stage_done(
        f"[1/4] Parse complete — tier={profile.tier}, native_domain={structure.native_domain!r}, "
        f"{len(structure.core_relations)} relation(s)",
        verbose=verbose,
    )

    slots, domain_report, slot_logs = allocate_slots(
        structure,
        num_experts,
        profile,
        model=model,
        client=client,
        model_path=embed_path,
        verbose=verbose,
    )
    domains = ", ".join(s.source_domain for s in slots)
    _stage_done(
        f"[2/4] Allocate complete — {len(slots)} slot(s): {domains} "
        f"(max domain↔native cosine={_max_native_cos(domain_report):.3f})",
        verbose=verbose,
    )
    if slot_logs and verbose:
        print(f"  Slot refine notes: {len(slot_logs)}", file=sys.stderr)

    members = instantiate_experts(slots, model=model, client=client)
    _stage_done(
        f"[3/4] Instantiate complete — {len(members)} expert(s): "
        + ", ".join(m.key for m in members),
        verbose=verbose,
    )

    other_domains_by_slot: dict[str, list[str]] = {}
    for member in members:
        other_domains_by_slot[member.key] = [
            m.slot.source_domain for m in members if m.key != member.key
        ]

    def refine_fn(member, avoid_description: str):
        return instantiate_expert(
            member.slot,
            other_domains_by_slot.get(member.key, []),
            model=model,
            client=client,
            avoid_description=avoid_description,
        )

    members, issues = validate_and_refine(
        members,
        profile,
        model_path=embed_path,
        max_rounds=max_refine_rounds,
        refine_fn=refine_fn,
        verbose=verbose,
    )

    expert_similarity = compute_expert_similarities(
        members,
        threshold=profile.max_expert_cosine,
        model_path=embed_path,
    )
    if slot_logs:
        domain_report = {**domain_report, "slot_refine_log": slot_logs}
    max_cos = expert_similarity["max_pairwise"]
    _stage_done(
        f"[4/4] Validate complete — tier={profile.tier}, "
        f"expert max cosine={max_cos:.3f} (max allowed {profile.max_expert_cosine:.3f}"
        + (
            f", min wanted {profile.min_expert_cosine:.3f}"
            if profile.min_expert_cosine is not None
            else ""
        )
        + f"), {len(issues)} refine issue(s)",
        verbose=verbose,
    )

    team = members_to_team(members)
    result = ExpertTeamResult(
        team=team,
        structure=structure,
        slots=slots,
        members=members,
        validation_issues=issues,
        expert_similarity=expert_similarity,
        domain_similarity=domain_report,
    )
    if return_profile:
        return result, profile
    return result


def _max_native_cos(domain_report: dict) -> float:
    rows = domain_report.get("slots") or []
    if not rows:
        return 0.0
    return max(r.get("native_domain_cosine", 0.0) for r in rows)
