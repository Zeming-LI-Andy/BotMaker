from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExpertTeamRequest:
    research_context: str
    num_experts: int
    diversity: int  # 1–10


@dataclass
class DiversityProfile:
    diversity: int
    tier: str  # focused | balanced | diverse
    max_expert_cosine: float
    min_expert_cosine: float | None
    anchor_min_source_native_cosine: float
    min_source_native_cosine: float | None
    max_source_native_cosine: float | None
    max_pairwise_domain_cosine: float


@dataclass
class CoreObject:
    name: str
    role: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoreObject:
        return cls(name=str(data["name"]), role=str(data["role"]))


@dataclass
class ResearchStructure:
    problem_summary: str
    core_objects: list[CoreObject]
    core_relations: list[str]
    constraints: list[str]
    native_domain: str
    key_concepts: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchStructure:
        relations = data.get("core_relations", [])
        normalized_relations: list[str] = []
        for r in relations:
            if isinstance(r, dict):
                normalized_relations.append(json.dumps(r, ensure_ascii=False))
            else:
                normalized_relations.append(str(r))
        return cls(
            problem_summary=str(data.get("problem_summary", "")),
            core_objects=[CoreObject.from_dict(o) for o in data.get("core_objects", [])],
            core_relations=normalized_relations,
            constraints=[str(c) for c in data.get("constraints", [])],
            native_domain=str(data.get("native_domain", "")),
            key_concepts=[str(k) for k in data.get("key_concepts", [])],
        )


def _normalize_anchor_relation(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


@dataclass
class PerspectiveSlot:
    slot_id: str
    role: str
    lens: str
    source_domain: str
    anchor_relation: str
    object_mapping_focus: str
    slot_focus: str
    contrast_hint: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerspectiveSlot:
        slot_id = str(data.get("slot_id", ""))
        return cls(
            slot_id=slot_id,
            role=str(data.get("role") or ("anchor" if slot_id == "slot_1" else "perspective")),
            lens=str(data.get("lens", "")),
            source_domain=str(data.get("source_domain", "")),
            anchor_relation=_normalize_anchor_relation(data.get("anchor_relation", "")),
            object_mapping_focus=str(data.get("object_mapping_focus", "")),
            slot_focus=str(data.get("slot_focus") or data.get("assigned_subquestion", "")),
            contrast_hint=str(data.get("contrast_hint", "")),
        )


@dataclass
class ExpertMember:
    key: str
    type: str
    expert_description: str
    slot: PerspectiveSlot

    def to_schema_entry(self) -> dict[str, str]:
        return {
            "type": self.type,
            "expert_description": self.expert_description,
        }


@dataclass
class ValidationIssue:
    member_key_a: str
    member_key_b: str
    cosine_similarity: float
    reason: str


@dataclass
class ExpertTeamResult:
    team: dict[str, dict[str, str]]
    structure: ResearchStructure
    slots: list[PerspectiveSlot]
    members: list[ExpertMember] = field(default_factory=list)
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    expert_similarity: dict[str, Any] = field(default_factory=dict)
    domain_similarity: dict[str, Any] = field(default_factory=dict)

    def to_dict(
        self,
        *,
        research_context: str,
        num_experts: int,
        diversity: int,
        profile: DiversityProfile | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "research_context": research_context,
            "num_experts": num_experts,
            "diversity": diversity,
            "team": self.team,
            "structure": {
                "problem_summary": self.structure.problem_summary,
                "core_objects": [
                    {"name": o.name, "role": o.role} for o in self.structure.core_objects
                ],
                "core_relations": self.structure.core_relations,
                "constraints": self.structure.constraints,
                "native_domain": self.structure.native_domain,
                "key_concepts": self.structure.key_concepts,
            },
            "slots": [
                {
                    "slot_id": s.slot_id,
                    "role": s.role,
                    "lens": s.lens,
                    "source_domain": s.source_domain,
                    "anchor_relation": s.anchor_relation,
                    "object_mapping_focus": s.object_mapping_focus,
                    "slot_focus": s.slot_focus,
                    "contrast_hint": s.contrast_hint,
                }
                for s in self.slots
            ],
            "members": [
                {
                    "key": m.key,
                    "type": m.type,
                    "expert_description": m.expert_description,
                    "slot": {
                        "slot_id": m.slot.slot_id,
                        "role": m.slot.role,
                        "lens": m.slot.lens,
                        "source_domain": m.slot.source_domain,
                        "anchor_relation": m.slot.anchor_relation,
                        "object_mapping_focus": m.slot.object_mapping_focus,
                        "slot_focus": m.slot.slot_focus,
                        "contrast_hint": m.slot.contrast_hint,
                    },
                }
                for m in self.members
            ],
            "validation_issues": [
                {
                    "member_key_a": i.member_key_a,
                    "member_key_b": i.member_key_b,
                    "cosine_similarity": i.cosine_similarity,
                    "reason": i.reason,
                }
                for i in self.validation_issues
            ],
            "expert_similarity": self.expert_similarity,
            "domain_similarity": self.domain_similarity,
        }
        if profile is not None:
            payload["diversity_profile"] = {
                "diversity": profile.diversity,
                "tier": profile.tier,
                "max_expert_cosine": profile.max_expert_cosine,
                "min_expert_cosine": profile.min_expert_cosine,
                "anchor_min_source_native_cosine": profile.anchor_min_source_native_cosine,
                "min_source_native_cosine": profile.min_source_native_cosine,
                "max_source_native_cosine": profile.max_source_native_cosine,
                "max_pairwise_domain_cosine": profile.max_pairwise_domain_cosine,
            }
        return payload
