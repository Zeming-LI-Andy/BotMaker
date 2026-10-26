from __future__ import annotations

import json
import sys
from pathlib import Path

from .diversity import DiversityProfile
from .domain_metrics import compute_domain_similarity
from .embeddings import (
    DEFAULT_BGE_M3_PATH,
    cosine_similarity_matrix,
    embed_texts,
    max_pairwise_cosine,
)
from .llm import call_llm_json, get_default_model, load_prompt
from .models import ExpertMember, PerspectiveSlot, ResearchStructure, ValidationIssue

DEFAULT_MAX_REFINE_ROUNDS = 10
_SLOT_REFINE_ROUNDS = 10


def compute_expert_similarities(
    members: list[ExpertMember],
    *,
    threshold: float | None = None,
    model_path: Path | str = DEFAULT_BGE_M3_PATH,
) -> dict:
    if len(members) < 2:
        out: dict = {"max_pairwise": 0.0, "max_pair": None, "pairs": []}
        if threshold is not None:
            out["threshold"] = threshold
        return out

    keys = [m.key for m in members]
    texts = [m.expert_description for m in members]
    embeddings = embed_texts(texts, model_path=model_path)
    sim = cosine_similarity_matrix(embeddings)

    pairs: list[dict] = []
    max_sim = -1.0
    max_pair: dict | None = None
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            value = round(float(sim[i, j]), 6)
            pair = {
                "member_key_a": keys[i],
                "member_key_b": keys[j],
                "cosine_similarity": value,
            }
            pairs.append(pair)
            if value > max_sim:
                max_sim = value
                max_pair = dict(pair)

    out = {
        "max_pairwise": round(max_sim, 6),
        "max_pair": max_pair,
        "pairs": pairs,
    }
    if threshold is not None:
        out["threshold"] = threshold
    return out


def _pick_expert_refinement_target(issue: ValidationIssue) -> str:
    return issue.member_key_b


def find_expert_violations(
    members: list[ExpertMember],
    profile: DiversityProfile,
    *,
    model_path: Path | str = DEFAULT_BGE_M3_PATH,
) -> list[ValidationIssue]:
    if len(members) < 2:
        return []

    texts = [m.expert_description for m in members]
    max_sim, idx_a, idx_b = max_pairwise_cosine(texts, model_path=model_path)
    issues: list[ValidationIssue] = []

    if max_sim > profile.max_expert_cosine:
        issues.append(
            ValidationIssue(
                member_key_a=members[idx_a].key,
                member_key_b=members[idx_b].key,
                cosine_similarity=max_sim,
                reason=(
                    f"Expert descriptions too similar: cosine {max_sim:.3f} > "
                    f"max {profile.max_expert_cosine:.3f}"
                ),
            )
        )

    if profile.min_expert_cosine is not None and max_sim < profile.min_expert_cosine:
        issues.append(
            ValidationIssue(
                member_key_a=members[idx_a].key,
                member_key_b=members[idx_b].key,
                cosine_similarity=max_sim,
                reason=(
                    f"Expert descriptions too dissimilar for focused tier: "
                    f"max cosine {max_sim:.3f} < min {profile.min_expert_cosine:.3f}"
                ),
            )
        )

    return issues[:1]


def _is_anchor_slot(slot: PerspectiveSlot) -> bool:
    return slot.role.strip().lower() == "anchor"


def _slot_id_to_index(slots: list[PerspectiveSlot]) -> dict[str, int]:
    return {slot.slot_id: i for i, slot in enumerate(slots)}


def _row_for_slot(domain_report: dict, idx: int) -> dict:
    rows = domain_report.get("slots", [])
    if 0 <= idx < len(rows):
        return rows[idx]
    return {}


def _annotate_domain_report(domain_report: dict, slots: list[PerspectiveSlot]) -> dict:
    rows = domain_report.get("slots", [])
    for i, row in enumerate(rows):
        if i < len(slots):
            row["role"] = slots[i].role
    return domain_report


def _best_native_slot_index(domain_report: dict) -> int:
    rows = domain_report.get("slots", [])
    if not rows:
        return 0
    best_idx = 0
    best_cos = -1.0
    for i, row in enumerate(rows):
        native_cos = float(row.get("native_domain_cosine", 0.0))
        if native_cos > best_cos:
            best_cos = native_cos
            best_idx = i
    return best_idx


def find_slot_violations(
    slots: list[PerspectiveSlot],
    profile: DiversityProfile,
    native_domain: str,
    domain_report: dict,
) -> list[tuple[int, str]]:
    """Return (slot_index, revision_requirement) for each violation."""
    violations: list[tuple[int, str]] = []
    if not slots:
        return violations

    anchor_indices = [i for i, slot in enumerate(slots) if _is_anchor_slot(slot)]
    if not anchor_indices:
        idx = _best_native_slot_index(domain_report)
        sid = slots[idx].slot_id
        violations.append(
            (
                idx,
                f"Slot {sid}: team is missing the required anchor expert. Revise this slot "
                f"to role='anchor' with source_domain rooted in native domain '{native_domain}' "
                "or a very close applied subfield. This expert should be the conventional "
                "domain expert most directly matched to the research topic.",
            )
        )
        return violations[:1]

    if len(anchor_indices) > 1:
        idx = anchor_indices[1]
        violations.append(
            (
                idx,
                f"Slot {slots[idx].slot_id}: only one anchor expert is allowed. Change this "
                "slot to role='perspective' and use a distinct non-anchor source_domain.",
            )
        )
        return violations[:1]

    anchor_idx = anchor_indices[0]
    anchor_cos = float(_row_for_slot(domain_report, anchor_idx).get("native_domain_cosine", 0.0))
    if anchor_cos < profile.anchor_min_source_native_cosine:
        violations.append(
            (
                anchor_idx,
                f"Slot {slots[anchor_idx].slot_id}: anchor source_domain "
                f"'{slots[anchor_idx].source_domain}' is not close enough to native domain "
                f"'{native_domain}' (cosine {anchor_cos:.3f} < "
                f"{profile.anchor_min_source_native_cosine:.3f}). Keep role='anchor' and "
                "move source_domain to the native domain or a close applied subfield.",
            )
        )
        return violations[:1]

    for i, row in enumerate(domain_report.get("slots", [])):
        native_cos = row.get("native_domain_cosine", 0.0)
        sid = slots[i].slot_id if i < len(slots) else row.get("slot_id", f"slot_{i + 1}")
        if i < len(slots) and _is_anchor_slot(slots[i]):
            continue

        if profile.min_source_native_cosine is not None and native_cos < profile.min_source_native_cosine:
            violations.append(
                (
                    i,
                    f"Slot {sid}: source_domain '{slots[i].source_domain}' is too far from "
                    f"native domain '{native_domain}' (cosine {native_cos:.3f} < "
                    f"{profile.min_source_native_cosine:.3f}). Move to native domain or an "
                    "immediate subfield only.",
                )
            )

        if profile.max_source_native_cosine is not None and native_cos > profile.max_source_native_cosine:
            violations.append(
                (
                    i,
                    f"Slot {sid}: source_domain '{slots[i].source_domain}' is too close to "
                    f"native domain '{native_domain}' (cosine {native_cos:.3f} > "
                    f"{profile.max_source_native_cosine:.3f}). Use a more distant field.",
                )
            )

    slot_indices = _slot_id_to_index(slots)
    for pair in domain_report.get("pairwise_domain_pairs", []):
        idx_a = slot_indices.get(str(pair.get("slot_a")), -1)
        idx_b = slot_indices.get(str(pair.get("slot_b")), -1)
        if idx_a < 0 or idx_b < 0:
            continue
        if _is_anchor_slot(slots[idx_a]) or _is_anchor_slot(slots[idx_b]):
            continue
        pair_cos = float(pair.get("cosine_similarity", 0.0))
        if pair_cos <= profile.max_pairwise_domain_cosine:
            continue
        violations.append(
            (
                idx_b,
                f"Slots {pair.get('slot_a')} and {pair.get('slot_b')} have non-anchor "
                f"source_domains too similar to each other (cosine {pair_cos:.3f} > "
                f"{profile.max_pairwise_domain_cosine:.3f}). Spread domains apart per tier.",
            )
        )
        break

    return violations[:1]


def refine_slot(
    slot: PerspectiveSlot,
    structure: ResearchStructure,
    other_domains: list[str],
    revision_requirement: str,
    *,
    model: str | None = None,
    client=None,
) -> PerspectiveSlot:
    if model is None:
        model = get_default_model()

    template = load_prompt("refine_slot.txt")
    prompt = template.format(
        problem_summary=structure.problem_summary,
        native_domain=structure.native_domain,
        slot_json=json.dumps(
            {
                "slot_id": slot.slot_id,
                "role": slot.role,
                "lens": slot.lens,
                "source_domain": slot.source_domain,
                "anchor_relation": slot.anchor_relation,
                "object_mapping_focus": slot.object_mapping_focus,
                "slot_focus": slot.slot_focus,
                "contrast_hint": slot.contrast_hint,
            },
            indent=2,
            ensure_ascii=False,
        ),
        revision_requirement=revision_requirement,
        other_domains=", ".join(other_domains) if other_domains else "(none)",
    )
    data = call_llm_json(
        prompt,
        model=model,
        system="You revise panel perspective slots to satisfy domain constraints.",
        client=client,
    )
    if not isinstance(data, dict):
        raise ValueError("Slot refine must return a JSON object")
    refined = PerspectiveSlot.from_dict(data)
    if not refined.slot_id:
        refined = PerspectiveSlot(
            slot_id=slot.slot_id,
            role=refined.role or slot.role,
            lens=refined.lens or slot.lens,
            source_domain=refined.source_domain,
            anchor_relation=refined.anchor_relation or slot.anchor_relation,
            object_mapping_focus=refined.object_mapping_focus or slot.object_mapping_focus,
            slot_focus=refined.slot_focus or slot.slot_focus,
            contrast_hint=refined.contrast_hint or slot.contrast_hint,
        )
    return refined


def validate_slots_and_refine(
    slots: list[PerspectiveSlot],
    structure: ResearchStructure,
    profile: DiversityProfile,
    *,
    model: str | None = None,
    client=None,
    model_path: Path | str = DEFAULT_BGE_M3_PATH,
    max_rounds: int = _SLOT_REFINE_ROUNDS,
    verbose: bool = True,
) -> tuple[list[PerspectiveSlot], dict, list[str]]:
    """Programmatic domain checks + optional slot LLM refine."""
    log: list[str] = []
    native = structure.native_domain

    for round_i in range(max_rounds + 1):
        domain_report = _annotate_domain_report(
            compute_domain_similarity(slots, native, model_path=model_path),
            slots,
        )
        violations = find_slot_violations(slots, profile, native, domain_report)
        if not violations:
            return slots, domain_report, log

        idx, requirement = violations[0]
        log.append(requirement)
        if round_i >= max_rounds:
            if verbose:
                print(f"  Slot validate: unresolved — {requirement}", file=sys.stderr)
            return slots, domain_report, log

        if idx < 0:
            idx = 0

        if verbose:
            print(f"  Refining slot {slots[idx].slot_id} (domain check)", file=sys.stderr)

        other = [s.source_domain for j, s in enumerate(slots) if j != idx]
        slots[idx] = refine_slot(
            slots[idx],
            structure,
            other,
            requirement,
            model=model,
            client=client,
        )

    domain_report = _annotate_domain_report(
        compute_domain_similarity(slots, native, model_path=model_path),
        slots,
    )
    return slots, domain_report, log


def validate_and_refine(
    members: list[ExpertMember],
    profile: DiversityProfile,
    *,
    model_path: Path | str = DEFAULT_BGE_M3_PATH,
    max_rounds: int = DEFAULT_MAX_REFINE_ROUNDS,
    refine_fn=None,
    verbose: bool = True,
) -> tuple[list[ExpertMember], list[ValidationIssue]]:
    all_issues: list[ValidationIssue] = []

    for _ in range(max_rounds + 1):
        issues = find_expert_violations(members, profile, model_path=model_path)
        if not issues:
            return members, all_issues

        issue = issues[0]
        all_issues.append(issue)

        if refine_fn is None or _ >= max_rounds:
            break

        if verbose:
            print(f"  Refining expert {issue.member_key_b} ({issue.reason})", file=sys.stderr)

        target_idx = next(i for i, m in enumerate(members) if m.key == _pick_expert_refinement_target(issue))
        avoid_member = next(m for m in members if m.key == issue.member_key_a)

        new_type, new_desc = refine_fn(members[target_idx], avoid_description=avoid_member.expert_description)
        members[target_idx] = ExpertMember(
            key=members[target_idx].key,
            type=new_type,
            expert_description=new_desc,
            slot=members[target_idx].slot,
        )

    return members, all_issues
