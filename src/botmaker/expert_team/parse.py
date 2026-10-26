from __future__ import annotations

from .diversity import DiversityProfile, parse_guidance
from .llm import call_llm_json, get_default_model, load_prompt
from .models import ResearchStructure


def parse_research(
    research_context: str,
    profile: DiversityProfile,
    *,
    model: str | None = None,
    client=None,
) -> ResearchStructure:
    if model is None:
        model = get_default_model()

    template = load_prompt("parse.txt")
    prompt = template.format(
        research_context=research_context.strip(),
        parse_guidance=parse_guidance(profile),
    )
    data = call_llm_json(
        prompt,
        model=model,
        system="You extract structured research problem representations for expert panel design.",
        client=client,
    )
    if not isinstance(data, dict):
        raise ValueError("Parse stage must return a JSON object")

    structure = ResearchStructure.from_dict(data)
    return structure
