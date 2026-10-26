from __future__ import annotations

import json
import re

from .llm import call_llm_json, get_default_model, load_prompt
from .models import ExpertMember, PerspectiveSlot

_SYSTEM = "You create scientific expert personas grounded in their academic domain."

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    slug = _SLUG_RE.sub("_", text.lower()).strip("_")
    return slug or "expert"


def _unique_key(base: str, used: set[str]) -> str:
    key = slugify(base)
    if key not in used:
        used.add(key)
        return key
    i = 2
    while f"{key}_{i}" in used:
        i += 1
    final = f"{key}_{i}"
    used.add(final)
    return final


def _slot_domain_blueprint(slot: PerspectiveSlot) -> dict:
    return {
        "slot_id": slot.slot_id,
        "role": slot.role,
        "source_domain": slot.source_domain,
        "lens": slot.lens,
        "anchor_relation": slot.anchor_relation,
    }


def _slot_panel_context(slot: PerspectiveSlot, other_domains: list[str]) -> dict:
    return {
        "contrast_hint": slot.contrast_hint,
        "object_mapping_focus": slot.object_mapping_focus,
        "slot_focus": slot.slot_focus,
        "other_domains_already_filled": other_domains or ["(none yet)"],
    }


def instantiate_expert(
    slot: PerspectiveSlot,
    other_domains: list[str],
    *,
    model: str | None = None,
    client=None,
    avoid_description: str | None = None,
) -> tuple[str, str]:
    if model is None:
        model = get_default_model()
    avoidance_clause = ""
    if avoid_description:
        avoidance_clause = (
            f"\n- This expert must be substantially different from the following description "
            f"(do not reuse the same domain framing or responsibilities):\n  \"{avoid_description}\""
        )

    template = load_prompt("instantiate_expert.txt")
    prompt = template.format(
        domain_blueprint_json=json.dumps(
            _slot_domain_blueprint(slot),
            indent=2,
            ensure_ascii=False,
        ),
        panel_context_json=json.dumps(
            _slot_panel_context(slot, other_domains),
            indent=2,
            ensure_ascii=False,
        ),
        avoidance_clause=avoidance_clause,
    )
    data = call_llm_json(prompt, model=model, system=_SYSTEM, client=client)
    if not isinstance(data, dict):
        raise ValueError("Instantiate stage must return a JSON object")

    expert_type = str(data.get("type", "")).strip()
    description = str(data.get("expert_description", "")).strip()
    if len(expert_type) < 1:
        raise ValueError("Expert type must be non-empty")
    if len(description) < 5:
        raise ValueError("Expert description must be at least 5 characters")
    return expert_type, description


def instantiate_experts(
    slots: list[PerspectiveSlot],
    *,
    model: str | None = None,
    client=None,
) -> list[ExpertMember]:
    if model is None:
        model = get_default_model()
    members: list[ExpertMember] = []
    used_keys: set[str] = set()
    filled_domains: list[str] = []

    for slot in slots:
        expert_type, description = instantiate_expert(
            slot,
            filled_domains,
            model=model,
            client=client,
        )
        key = _unique_key(expert_type, used_keys)
        members.append(
            ExpertMember(
                key=key,
                type=expert_type,
                expert_description=description,
                slot=slot,
            )
        )
        filled_domains.append(slot.source_domain)
    return members


def members_to_team(members: list[ExpertMember]) -> dict[str, dict[str, str]]:
    return {member.key: member.to_schema_entry() for member in members}
