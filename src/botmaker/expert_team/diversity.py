from __future__ import annotations

from .models import DiversityProfile


def _clamp_diversity(diversity: int) -> int:
    return max(1, min(10, diversity))


def _tier(diversity: int) -> str:
    """Three tiers only: focused (homogeneous) -> balanced -> diverse."""
    d = _clamp_diversity(diversity)
    if d <= 3:
        return "focused"
    if d <= 6:
        return "balanced"
    return "diverse"


def diversity_profile(diversity: int, num_experts: int) -> DiversityProfile:
    d = _clamp_diversity(diversity)
    tier = _tier(d)

    # Expert description cosine (bge-m3): high d -> lower allowed max similarity.
    max_expert_cosine = round(0.92 - (d - 1) * 0.032, 2)

    if tier == "focused":
        return DiversityProfile(
            diversity=d,
            tier=tier,
            max_expert_cosine=max_expert_cosine,
            min_expert_cosine=round(0.28 + (d - 1) * 0.04, 2),
            anchor_min_source_native_cosine=0.58,
            min_source_native_cosine=0.42,
            max_source_native_cosine=None,
            max_pairwise_domain_cosine=0.72,
        )
    if tier == "balanced":
        return DiversityProfile(
            diversity=d,
            tier=tier,
            max_expert_cosine=max_expert_cosine,
            min_expert_cosine=None,
            anchor_min_source_native_cosine=0.55,
            min_source_native_cosine=None,
            max_source_native_cosine=0.58,
            max_pairwise_domain_cosine=0.65,
        )
    return DiversityProfile(
        diversity=d,
        tier=tier,
        max_expert_cosine=max_expert_cosine,
        min_expert_cosine=None,
        anchor_min_source_native_cosine=0.52,
        min_source_native_cosine=None,
        max_source_native_cosine=0.45,
        max_pairwise_domain_cosine=0.55,
    )


def parse_guidance(profile: DiversityProfile) -> str:
    if profile.tier == "focused":
        return (
            f"Panel tier: FOCUSED (low diversity={profile.diversity}). "
            "Extract problem structure within the native domain or immediate subfields. "
            "Do NOT span distant cross-disciplinary objects or relations."
        )
    if profile.tier == "balanced":
        return (
            f"Panel tier: BALANCED (diversity={profile.diversity}). "
            "Extract problem structure with moderate breadth -- related fields are fine, "
            "avoid forcing exotic distant domains unless clearly relevant."
        )
    return (
        f"Panel tier: DIVERSE (high diversity={profile.diversity}). "
        "Extract problem structure at a functional level so multiple fields can map to it, "
        "while preserving the native domain as the required anchor perspective."
    )


def allocation_guidance(profile: DiversityProfile, native_domain: str) -> str:
    anchor_rule = (
        f"- slot_1 MUST have role='anchor' and source_domain rooted in '{native_domain}' "
        "or a very close applied subfield. This is the conventional expert who would "
        "normally review or lead the research.\n"
    )
    if profile.tier == "focused":
        return (
            f"Allocation tier: FOCUSED -- build a cohesive panel around native domain "
            f"'{native_domain}' and its immediate subfields.\n"
            f"{anchor_rule}"
            "- All remaining slots should stay in the native domain or close subfields.\n"
            "- Do NOT use distant analogies (ecology, physics, theater, etc.).\n"
            "- object_mapping_focus: brief methodological link only -- no cross-domain metaphor.\n"
            "- contrast_hint: contrast sub-approaches within the same field."
        )
    if profile.tier == "balanced":
        return (
            f"Allocation tier: BALANCED -- moderate spread around '{native_domain}'.\n"
            f"{anchor_rule}"
            "- Majority of remaining slots should connect clearly to the native domain or adjacent areas.\n"
            "- At most one remaining slot may come from a clearly related neighboring field.\n"
            "- Avoid gratuitous cross-domain analogies; functional relevance is required."
        )
    return (
        f"Allocation tier: DIVERSE -- maximize substantive perspective spread.\n"
        f"{anchor_rule}"
        "- Remaining slots should draw from other fields, but must stay scientifically useful for the topic.\n"
        "- Do NOT use purely theatrical, artistic, or metaphor-only domains unless the research topic is about them.\n"
        "- Map problem objects by functional role when borrowing another field.\n"
        "- contrast_hint: explain what the external field contributes vs. default native-domain practice."
    )


def profile_summary(profile: DiversityProfile, num_experts: int, native_domain: str) -> str:
    lines = [
        f"Diversity score: {profile.diversity} (tier={profile.tier})",
        f"Number of experts: {num_experts}",
        f"Native domain: {native_domain}",
        allocation_guidance(profile, native_domain),
        "Programmatic checks after allocation:",
        "  - exactly one anchor slot is required (slot_1, role='anchor')",
        f"  - anchor source_domain ↔ native_domain cosine: >= {profile.anchor_min_source_native_cosine}",
        f"  - non-anchor source_domain ↔ native_domain cosine: "
        f"min>={profile.min_source_native_cosine} (focused), "
        f"max<={profile.max_source_native_cosine} (diverse/balanced per tier)",
        f"  - max pairwise source_domain cosine among non-anchor slots: <= {profile.max_pairwise_domain_cosine}",
        "Programmatic checks after instantiate:",
        f"  - max expert-description cosine: <= {profile.max_expert_cosine}",
    ]
    if profile.min_expert_cosine is not None:
        lines.append(
            f"  - max expert-description cosine should be >= {profile.min_expert_cosine} (focused)"
        )
    return "\n".join(lines)

