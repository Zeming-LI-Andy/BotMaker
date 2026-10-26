"""Smoke test for expert_team validation (bge-m3 cosine). Run with huggingface-download env."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from botmaker.expert_team.diversity import diversity_profile
from botmaker.expert_team.instantiate import members_to_team
from botmaker.expert_team.models import ExpertMember, PerspectiveSlot
from botmaker.expert_team.validate import validate_and_refine

slot = PerspectiveSlot("s1", "computational", "ml", "r", "m", "q")
members = [
    ExpertMember(
        "a",
        "A",
        "Focuses on deep learning for medical imaging with CNN architectures.",
        slot,
    ),
    ExpertMember(
        "b",
        "B",
        "Focuses on deep learning for medical imaging with CNN architectures and transfer learning.",
        slot,
    ),
]
profile = diversity_profile(8, 2)
refined, issues = validate_and_refine(members, profile, max_rounds=0)
print("issues", len(issues))
if issues:
    print("max_cosine", round(issues[0].cosine_similarity, 3))
print(members_to_team(refined))
