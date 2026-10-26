from __future__ import annotations

import json

from .diversity import DiversityProfile, profile_summary
from .llm import call_llm_json, get_default_model, load_prompt
from .models import PerspectiveSlot, ResearchStructure
from .validate import validate_slots_and_refine

_SYSTEM = "You allocate expert perspective slots for scientific panels."


def allocate_slots(
    structure: ResearchStructure,
    num_experts: int,
    profile: DiversityProfile,
    *,
    model: str | None = None,
    client=None,
    model_path=None,
    verbose: bool = True,
) -> tuple[list[PerspectiveSlot], dict, list[str]]:
    if model is None:
        model = get_default_model()
    if num_experts < 1:
        raise ValueError("num_experts must be at least 1")

    template = load_prompt("allocate_slots.txt")
    prompt = template.format(
        research_structure_json=json.dumps(
            {
                "problem_summary": structure.problem_summary,
                "core_objects": [{"name": o.name, "role": o.role} for o in structure.core_objects],
                "core_relations": structure.core_relations,
                "constraints": structure.constraints,
                "native_domain": structure.native_domain,
                "key_concepts": structure.key_concepts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        diversity_profile_summary=profile_summary(
            profile, num_experts, structure.native_domain
        ),
        num_experts=num_experts,
    )
    data = call_llm_json(prompt, model=model, system=_SYSTEM, client=client)
    if not isinstance(data, dict):
        raise ValueError("Allocate stage must return a JSON object")

    raw_slots = data.get("slots", [])
    if not isinstance(raw_slots, list) or len(raw_slots) != num_experts:
        raise ValueError(
            f"Expected exactly {num_experts} slots, got "
            f"{len(raw_slots) if isinstance(raw_slots, list) else 'non-list'}"
        )

    slots = [PerspectiveSlot.from_dict(item) for item in raw_slots]

    kwargs = {}
    if model_path is not None:
        kwargs["model_path"] = model_path

    return validate_slots_and_refine(
        slots,
        structure,
        profile,
        model=model,
        client=client,
        verbose=verbose,
        **kwargs,
    )
