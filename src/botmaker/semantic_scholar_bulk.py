"""
Task 2 repackaging: Semantic Scholar bulk search with iterative query broadening.

Outputs:
  <output_path>/<expert_name>/search_paper.jsonl
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests

_QUERY_SYNTAX = """\
Semantic Scholar Boolean query syntax:
- `fish ladder`        -> implicit AND
- `+`                  -> explicit AND
- `|`                  -> OR (use the pipe character; do NOT write the word `OR`)
- `-`                  -> NOT (negate a term)
- `"..."`              -> exact phrase
- `*`                  -> prefix wildcard (e.g. `bio*`)
- `( )`                -> grouping / precedence
- `~N`                 -> edit distance (e.g. `fish~`, `fish~1`)
- `"..."~N`            -> phrase slop (e.g. `"fish ladder"~3`)
Terms are stemmed (running <-> run). Searches are case-insensitive."""

_JSON_OUTPUT = (
    'Output a JSON object with a single key "query" wrapped in ```json fences.\n'
    "Example:\n```json\n{\"query\": \"...\"}\n```"
)

_LOW_YIELD_THRESHOLD_FLOOR = 50
_MAX_BROADEN_PER_FACET = 2
_DEFAULT_MAX_QUERIES = 10
_SATURATION_NEW_RATIO = 0.10
_MIN_AND_GROUPS = 3
_MAX_OR_TERMS_PER_GROUP = 12
_MAX_QUERY_CHARS = 2200
_TRANSIENT_STATUS_CODES = {500, 502, 503, 504}
_MAX_TRANSIENT_RETRIES = 3
_QUERY_HISTORY_NAME = "search_query_history.jsonl"
_ANCHOR_PIVOT_HISTORY_THRESHOLD = 8

_QUERY_SHAPE = (
    "Query shape — balanced recall with a stable topic anchor:\n"
    "- Use exactly 3 AND (+) groups on seed and facet-pivot queries:\n"
    "    G1 = broad expert field / evidence context entry point (methods/tools are optional; include them only when they define the searchable domain)\n"
    "    G2 = core application/topic anchor from the expert description (preserve it normally; broaden to its parent topic family only when recall is too low)\n"
    "    G3 = bounded facet: outcome, object subtype, mechanism, population, or analysis angle to rotate\n"
    "- Each alternative group must use `|` between terms and needs ≥2 terms; at least one term per group must be a multi-word phrase "
    "or domain-specific proper name.\n"
    f"- Each `|` group may contain at most {_MAX_OR_TERMS_PER_GROUP} terms. Choose the most connected terms; "
    "do not enumerate a whole category atlas or long list of loosely related terms.\n"
    "- Related expansion is encouraged only within the expert's connected topical neighborhood "
    "(such as the same application family, mechanism family, method family, object family, or outcome family).\n"
    "- Keep at least two strong domain anchors across the three groups; never let the query become only generic methods, "
    "generic populations, or generic outcomes.\n"
    "- Terms must be title/abstract-observable: use phrases that authors commonly write in paper titles or abstracts, "
    "not internal reasoning labels, evaluation goals, or overly abstract descriptions.\n"
    "- Use prefix wildcards (`*`) sparingly and only on domain-specific stems that preserve the expert's topic.\n"
    "- Do NOT use ultra-broad wildcards (bio*, data*, model*, health*, disease*, study*, method*, network*, *omics* alone).\n"
    "- Forbidden as the sole core of any group: machine learning, deep learning, artificial intelligence, "
    "analysis, model, study, research, data, algorithm, biomarker, network, approach, system.\n"
    "- Use NOT (-) sparingly (0–2 terms), only for clearly irrelevant major fields.\n"
    "- Exactly 3 AND groups are required in every mode, including narrowed queries."
)

_TITLE_ABSTRACT_TERM_GUIDANCE = (
    "Title/abstract matching guidance:\n"
    "- Semantic Scholar queries work best when terms are likely to appear literally in titles or abstracts.\n"
    "- Prefer established field names, object names, method family names, study-design phrases, outcome/application "
    "phrases, and common abbreviations.\n"
    "- Avoid terms that are mainly interpretive or internal to the expert profile, such as 'biological plausibility', "
    "'clinically actionable', 'routine care', 'research vocabulary', 'analytical concern', or 'methodological contribution', "
    "unless those phrases are established literature labels.\n"
    "- Convert profile-specific descriptions into searchable title/abstract phrases."
)

_WILDCARD_GUIDANCE = (
    "Wildcard guidance:\n"
    "- Prefix wildcard `*` can improve recall by covering title/abstract variants, but it must be used sparingly.\n"
    "- Use at most 1-2 wildcard terms per AND group, and only when the stem is domain-specific enough to preserve meaning.\n"
    "- Good wildcard candidates are field, assay, object, method-family, or application-family stems that remain specific after truncation.\n"
    "- Do NOT wildcard generic stems such as bio*, data*, model*, method*, study*, disease*, health*, clinical*, analysis*, "
    "network*, system*, risk*, outcome*, population*, or *omics* alone.\n"
    "- Do NOT use leading wildcards. Only prefix wildcards are allowed, e.g. `specificstem*`, never `*stem` or `*omics*`.\n"
    "- If a wildcard could plausibly pull in unrelated disciplines, use quoted phrases or explicit alternatives instead."
)

_CORE_ANCHOR_GUIDANCE = (
    "Core-anchor guidance:\n"
    "- Main anchors (G1/G2) should come from the expert's core domain, core object/material/system, "
    "core method/process family, data/source context, or application family.\n"
    "- External outcomes, diseases, populations, settings, or downstream use cases should usually be G3 constraints, "
    "not G1/G2 main anchors, unless they are explicitly the expert's own core domain.\n"
    "- Anchor-pivot means reorganizing core-domain anchors, not promoting a loosely related outcome category into the main search topic.\n"
    "- Preserve the expert identity: a query should still look like it belongs to this expert if G3 is removed."
)

_EVIDENCE_CONTEXT_GUIDANCE = (
    "Evidence-context guidance:\n"
    "- Study-design, population, setting, or evidence-context terms are useful, but they should not be mandatory in every query.\n"
    "- Do not require terms such as cohort, prospective, longitudinal, epidemiologic, population-based, patient, "
    "or clinical in every pivot unless they are the expert's own core domain.\n"
    "- Use evidence-context terms as one possible rotating facet or as a light context group, not as a fixed anchor across all searches.\n"
    "- If recall is low or recent pivots are saturated, relax the evidence-context group first: replace narrow study-design terms "
    "with broader title/abstract-observable context terms, or move the evidence context to G3.\n"
    "- A good recall-oriented query can use core field + core object/method/application + sample/source/application context, "
    "without always forcing study-design words."
)

_METHOD_DETAIL_GUIDANCE = (
    "Method-detail guidance:\n"
    "- Method keywords are allowed, but prefer method families over named tools, databases, metrics, preprocessing steps, "
    "or specific algorithms, especially in early or low-recall queries.\n"
    "- Named tools/databases/algorithms/metrics should appear only as optional alternatives inside a broader method-family group, "
    "not as the majority of that group and not as a separate AND group.\n"
    "- If prior queries under-collected, relax method/tool details first: replace named tools, databases, metrics, "
    "preprocessing steps, and narrow algorithms with their parent method family.\n"
    "- A method group should remain title/abstract-observable and broad enough to retrieve a sizable literature area.\n"
    "- Avoid building a query around a checklist of technical procedures from the expert description."
)

def _low_yield_threshold(min_papers: int) -> int:
    return max(_LOW_YIELD_THRESHOLD_FLOOR, min_papers // 500)


def _normalize_boolean_words_outside_quotes(text: str) -> str:
    """Normalize common Boolean words to Semantic Scholar syntax outside quoted phrases."""
    parts = re.split(r'(".*?")', text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\bOR\b", "|", parts[i], flags=re.I)
        parts[i] = re.sub(r"\bAND\b", "+", parts[i], flags=re.I)
    return "".join(parts)


def _normalize_query_syntax(query: str) -> str:
    query = _normalize_boolean_words_outside_quotes(query.strip())
    query = re.sub(r"\s*\|\s*", "|", query)
    query = re.sub(r"\s*\+\s*", " + ", query)
    query = re.sub(r"\s+", " ", query)
    return query.strip()


def _format_yield_history(query_yields: list[tuple[str, int]], *, min_papers: int) -> str:
    if not query_yields:
        return "(none yet)"
    threshold = _low_yield_threshold(min_papers)
    lines = []
    omitted = max(0, len(query_yields) - 20)
    if omitted:
        lines.append(f"  (... {omitted} older query record(s) omitted; avoid repeating recent queries below ...)")
    for i, (q, n) in enumerate(query_yields[-20:], omitted + 1):
        status = "exhausted" if n < threshold else "productive"
        lines.append(f"  [{i}] +{n:,} new — {status}\n      {q}")
    return "\n".join(lines)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_query_history(path: Path) -> list[dict]:
    history: list[dict] = []
    if not path.exists():
        return history
    for rec in _iter_jsonl(path):
        query = str(rec.get("query") or "").strip()
        if query:
            history.append(rec)
    return history


def _append_query_history(
    path: Path,
    *,
    attempt: int,
    mode: str,
    query: str,
    new_count: int,
    total_returned: int,
    cumulative_total: int,
) -> None:
    rec = {
        "attempt": attempt,
        "mode": mode,
        "query": query,
        "new_count": new_count,
        "total_returned": total_returned,
        "cumulative_total": cumulative_total,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


_FACET_PIVOT_STREAK = 2


def _is_and_separator(query: str, index: int) -> bool:
    """Return True when '+' looks like a Boolean AND separator, not a term character."""
    prev_char = query[index - 1] if index > 0 else ""
    next_char = query[index + 1] if index + 1 < len(query) else ""
    return prev_char != "+" and next_char != "+"


def _count_and_groups(query: str) -> int:
    return len(_split_and_groups(query))


def _split_and_groups(query: str) -> list[str]:
    groups: list[str] = []
    start = 0
    in_quote = False

    for i, ch in enumerate(query):
        if ch == '"':
            in_quote = not in_quote
        elif ch == "+" and not in_quote and _is_and_separator(query, i):
            part = query[start:i].strip()
            if part:
                groups.append(part)
            start = i + 1

    tail = query[start:].strip()
    if tail:
        groups.append(tail)
    return groups


def _count_or_terms(group: str) -> int:
    count = 1
    in_quote = False
    for ch in group:
        if ch == '"':
            in_quote = not in_quote
        elif ch == "|" and not in_quote:
            count += 1
    return count


def _query_passes_shape_check(query: str, *, require_three_groups: bool) -> bool:
    return _query_shape_failure_reason(query, require_three_groups=require_three_groups) == ""


def _query_shape_failure_reason(query: str, *, require_three_groups: bool) -> str:
    group_count = _count_and_groups(query)
    if require_three_groups and group_count != _MIN_AND_GROUPS:
        return f"detected {group_count} AND groups; need exactly {_MIN_AND_GROUPS}"
    if len(query) > _MAX_QUERY_CHARS:
        return f"query is too long ({len(query)} chars); maximum is {_MAX_QUERY_CHARS}"
    for i, group in enumerate(_split_and_groups(query), 1):
        term_count = _count_or_terms(group)
        if term_count > _MAX_OR_TERMS_PER_GROUP:
            return (
                f"AND group {i} has {term_count} OR terms; "
                f"maximum is {_MAX_OR_TERMS_PER_GROUP}"
            )
    return ""


def _choose_query_mode(
    *,
    query_yields: list[tuple[str, int]],
    low_yield_streak: int,
    last_mode: str,
    broaden_round: int,
    last_query_productive: bool,
    last_query_saturated: bool,
) -> str:
    """Return query-generation mode. At most _MAX_BROADEN_PER_FACET broadens per facet."""
    if not query_yields:
        return "seed"

    if (
        len(query_yields) >= _ANCHOR_PIVOT_HISTORY_THRESHOLD
        and low_yield_streak >= 1
    ):
        return "anchor-pivot"

    if last_query_saturated:
        if len(query_yields) >= _ANCHOR_PIVOT_HISTORY_THRESHOLD:
            return "anchor-pivot"
        return "facet-pivot"

    if last_mode in ("facet-pivot", "anchor-pivot", "narrowed"):
        if low_yield_streak >= _FACET_PIVOT_STREAK:
            return "anchor-pivot"
        return "facet-pivot"

    if broaden_round >= _MAX_BROADEN_PER_FACET or last_query_saturated:
        return "facet-pivot"
    if low_yield_streak >= _FACET_PIVOT_STREAK:
        return "anchor-pivot"
    if last_mode in ("seed", "start") and last_query_productive:
        return "facet-pivot"
    if last_mode in ("seed", "start", "broaden") and broaden_round < _MAX_BROADEN_PER_FACET:
        if low_yield_streak >= 1 or last_mode in ("seed", "start"):
            return "broaden"
    return "facet-pivot"


def _format_target_band(cumulative: int, min_papers: int, max_papers: int) -> str:
    gap = max(0, min_papers - cumulative)
    if cumulative >= min_papers:
        return (
            f"Target cumulative: {min_papers:,}–{max_papers:,} unique papers. "
            f"Current cumulative: {cumulative:,} (minimum already met)."
        )
    if gap > 500_000:
        next_low, next_high = 10_000, 80_000
    elif gap > 50_000:
        next_low = 2_000
        next_high = min(50_000, max(10_000, gap // 2))
    else:
        next_low = max(1_000, gap // 3)
        next_high = min(50_000, max(next_low * 2, gap))
    return (
        f"Target cumulative: {min_papers:,}–{max_papers:,} unique papers.\n"
        f"Current cumulative: {cumulative:,}. Gap to minimum: {gap:,}.\n"
        f"For the NEXT query, aim for roughly {next_low:,}–{next_high:,} NEW unique papers. "
        "Prefer a related, higher-recall query that keeps at least two strong domain anchors over a very broad query."
    )


def _broaden_ladder_instruction(broaden_round: int) -> str:
    if broaden_round == 0:
        return (
            "Single broaden step (keep exactly 3 AND + groups):\n"
            "1. If G2 is a method-only group, fix the structure: merge methods into G1 and make G2 the core topic/application anchor.\n"
            "2. Expand `|` alternatives inside G1 and G3: abbreviations, full names, "
            "close domain phrases from the expert description.\n"
            "3. If any group is too method/tool-heavy, replace named tools, databases, metrics, or algorithms with parent method-family terms.\n"
            "4. If a study-design/population/evidence-context group is limiting recall, move it to G3 or broaden it to a source/context phrase.\n"
            "5. Add prefix wildcards ONLY on specific domain terms already in the query "
            "(use a domain stem, not an ultra-broad stem).\n"
            f"6. Keep each `|` group at ≤{_MAX_OR_TERMS_PER_GROUP} terms; replace weaker terms instead of appending indefinitely.\n"
            "7. Do NOT drop AND groups. Do NOT remove or replace the core topic anchor."
        )
    return (
        "Second and final broaden step for this facet:\n"
        "1. Keep G2 as the same core topic/application anchor.\n"
        "2. Make G3 a moderately broader parent facet within the same topic family.\n"
        "3. Keep G1 concise; prefer the parent field/source context or method family over an expanding list of methods/tools or study-design terms."
    )


def _coverage_stage_instruction(cumulative_total: int, min_papers: int, facet_index: int) -> str:
    if cumulative_total < max(10_000, min_papers // 5):
        return (
            "Coverage stage: early high-recall expansion within a 10-query budget.\n"
            "- Choose a broad, high-yield facet family that is clearly connected to the expert field.\n"
            "- Build the main relevant paper pool first; do not overfit to technical details from the expert description.\n"
            "- Prefer parent outcome/object/application/evidence-context facets over narrow mechanisms, "
            "single indicators, demographic strata, analysis techniques, validation metrics, or named tools.\n"
            "- Do not spend early pivots on a micro-topic unless prior queries already covered the major facet families."
        )
    if cumulative_total < max(30_000, min_papers // 2):
        return (
            "Coverage stage: mid-search diversification.\n"
            "- Rotate to a different major facet family with limited overlap to prior queries.\n"
            "- Specific methods, mechanisms, or subtypes are allowed only when they represent a sizable literature area.\n"
            "- Prefer a broad neighboring facet with partial overlap over a highly novel but tiny niche."
        )
    return (
        "Coverage stage: late-search gap filling.\n"
        f"- Use facet index {facet_index} to cover remaining relevant subareas while avoiding tiny micro-topics."
    )


def _anchor_rotation_instruction(
    facet_index: int,
    query_yields: list[tuple[str, int]],
    *,
    min_papers: int,
    force_anchor_shift: bool = False,
) -> str:
    threshold = _low_yield_threshold(min_papers)
    recent_low_yield = len(query_yields) >= 3 and all(n < threshold for _, n in query_yields[-3:])
    patterns = [
        (
            "Pattern A: core field/domain anchor + core object/material/system anchor + broad application or evidence context. "
            "Use when building the main paper pool; keep outcomes and study-design terms as G3/context rather than fixed G1/G2 unless they define the expert domain."
        ),
        (
            "Pattern B: core field/domain anchor + core method/process family + concrete object/material/application context. "
            "Use when the expert's method family is central, but keep it title/abstract-observable."
        ),
        (
            "Pattern C: core object/material/system anchor + data/source/sample context + method or process family. "
            "Use to escape repeated outcome-only or study-design-only pivots while preserving the expert identity."
        ),
        (
            "Pattern D: core object/material/system anchor + analysis/integration family + light evidence/source context. "
            "Use when prior queries over-fixed the same study-design or population group; keep external outcomes in G3."
        ),
    ]
    pattern = patterns[facet_index % len(patterns)]
    if force_anchor_shift:
        extra = (
            "\nAnchor-pivot mode is active. You MUST change one main anchor role (G1 or G2), "
            "not only replace G3. The new G1/G2 pair must not be the same semantic pair as the last query. "
            "G1/G2 must be core-domain anchors, not external outcome/category anchors. "
            "Preserve exactly 3 AND groups and at least two strong domain anchors."
        )
    elif recent_low_yield:
        extra = (
            "\nRecent pivots were low-yield. You MUST change one main anchor role (G1 or G2), "
            "not only replace G3, while preserving exactly 3 AND groups and at least two strong core-domain anchors."
        )
    else:
        extra = ""
    return f"Anchor-rotation target for this pivot:\n- {pattern}{extra}"


def _build_seed_prompt(
    expert_name: str,
    expert_description: str,
    *,
    min_papers: int,
    max_papers: int,
    cumulative_total: int = 0,
) -> str:
    resume_instruction = (
        "Resume/gap-filling context:\n"
        f"- {cumulative_total:,} unique papers are already loaded from previous searches.\n"
        "- Do NOT regenerate the most obvious central query. Generate a complementary, high-yield query "
        "that is likely to add new papers while staying in the expert's domain.\n"
        "- If the central domain and major outcome/application facets were probably covered, broaden one anchor "
        "to its parent concept family and use the other two groups to preserve precision.\n\n"
        if cumulative_total > 0
        else ""
    )
    return (
        "You are a literature retrieval specialist. "
        "Generate a SINGLE Semantic Scholar bulk-search query for the expert below.\n\n"
        f"Expert Name: {expert_name}\n"
        f"Expert Description: {expert_description}\n\n"
        f"{_format_target_band(cumulative_total, min_papers, max_papers)}\n\n"
        f"{resume_instruction}"
        f"{_QUERY_SYNTAX}\n\n"
        f"{_QUERY_SHAPE}\n\n"
        f"{_TITLE_ABSTRACT_TERM_GUIDANCE}\n\n"
        f"{_WILDCARD_GUIDANCE}\n\n"
        f"{_CORE_ANCHOR_GUIDANCE}\n\n"
        f"{_EVIDENCE_CONTEXT_GUIDANCE}\n\n"
        f"{_METHOD_DETAIL_GUIDANCE}\n\n"
        "Seed query guidelines:\n"
        "1. Use exactly 3 AND (+) groups using this structure: "
        "G1 broad expert field/evidence context, G2 core application/topic anchor, G3 bounded facet.\n"
        "2. G2 is NOT a narrow method/tool group or a mandatory study-design group. Put the stable application, object, "
        "material, target system, source context, or research context from the expert description in G2.\n"
        "3. Do not simply copy long lists of technical details from the expert description. Compress methods, "
        "metrics, and workflows into broader searchable method families.\n"
        "4. Put concrete tools/databases/algorithms inside G1 only as a minority of alternatives under a broader method-family anchor.\n"
        "5. G3 should be broad enough to retrieve literature: use a compact family of outcomes, objects, "
        "applications, evidence contexts, or mechanisms, not a single over-specific endpoint.\n"
        "6. For the seed query, avoid low-yield micro-facets such as single indicators, named tools, "
        "validation metrics, narrow analysis techniques, or demographic strata unless they define the expert field.\n"
        "7. Do not make study-design/population/evidence-context terms mandatory in every query; use them only when they add needed precision.\n"
        "8. Keep at least two strong domain anchors across G1/G2/G3, so recall improves without drifting into a generic field.\n"
        "9. Use `|` between alternatives. Never write the word OR in the query.\n"
        "10. Use prefix wildcards only when they are domain-specific and improve title/abstract recall without broad drift.\n"
        "11. Each `|` group: ≥2 terms; include multi-word phrases where possible.\n"
        "12. Avoid NOT (-) on the first query.\n"
        "13. The query MUST be in English.\n"
        "14. "
        f"{_JSON_OUTPUT}"
    )


def _build_facet_pivot_prompt(
    expert_name: str,
    expert_description: str,
    query_yields: list[tuple[str, int]],
    cumulative_total: int,
    *,
    min_papers: int,
    max_papers: int,
    facet_index: int,
    force_anchor_shift: bool = False,
) -> str:
    recall_expansion = cumulative_total < max(10_000, min_papers // 5)
    recall_instruction = (
        "Anchor-pivot mode is ON. You must change either G1 or G2 to a different anchor family; "
        "use the remaining groups to preserve precision and domain relevance."
        if force_anchor_shift
        else "Recall-expansion mode is ON because the cumulative count is still far below target. "
        "You may broaden G2 from an overly narrow phrase to its parent topic/application family, "
        "but G1 and G3 must remain meaningful domain anchors so the query does not drift."
        if recall_expansion
        else
        "Recall-expansion mode is OFF. Keep G2 stable and replace G3 only."
    )
    rule_2 = (
        "2. Because this is anchor-pivot mode, changing only G3 is INVALID. "
        "The new query must change G1 or G2 to a different title/abstract-observable CORE anchor family. "
        "Do not satisfy anchor-pivot by moving an external outcome, disease, population, or setting into G1/G2.\n"
        if force_anchor_shift
        else (
            "2. Prefer broad, high-yield facets early: parent objects, major outcome families, evidence contexts, "
            "measurement contexts, or application subareas. Avoid low-yield micro-facets until the target is close.\n"
        )
    )
    intro = (
        "Prior queries reused the same main anchors and are now low-yield. "
        "Generate an ANCHOR-PIVOT query with a different G1/G2 role combination.\n\n"
        if force_anchor_shift
        else "Prior queries for the current facet are saturated or exhausted. "
    )
    return (
        "You are a literature retrieval specialist. "
        f"{intro}"
        "Generate a query for a DIFFERENT sub-topic within the same expert field.\n\n"
        f"Expert Name: {expert_name}\n"
        f"Expert Description: {expert_description}\n\n"
        f"{_format_target_band(cumulative_total, min_papers, max_papers)}\n\n"
        f"Cumulative unique papers so far: {cumulative_total:,}\n\n"
        "Query yield history:\n"
        f"{_format_yield_history(query_yields, min_papers=min_papers)}\n\n"
        f"Last query: {query_yields[-1][0] if query_yields else ''}\n\n"
        f"{recall_instruction}\n\n"
        f"{_coverage_stage_instruction(cumulative_total, min_papers, facet_index)}\n\n"
        f"{_anchor_rotation_instruction(facet_index, query_yields, min_papers=min_papers, force_anchor_shift=force_anchor_shift)}\n\n"
        f"{_TITLE_ABSTRACT_TERM_GUIDANCE}\n\n"
        f"{_WILDCARD_GUIDANCE}\n\n"
        f"{_CORE_ANCHOR_GUIDANCE}\n\n"
        f"{_EVIDENCE_CONTEXT_GUIDANCE}\n\n"
        f"{_METHOD_DETAIL_GUIDANCE}\n\n"
        "Anchored facet-pivot strategy:\n"
        "  1. Keep G1/G2 as core-domain anchors: expert field, core object/material/system, method/process family, "
        "data/source/sample context, or application family.\n"
        "  2. Prefer keeping G2 stable in normal facet-pivot mode. In anchor-pivot mode, change G1 or G2 only to another "
        "core-domain anchor family, not to a loose external outcome category.\n"
        "  3. Follow the anchor-rotation target. If yield history shows several low-new or saturated pivots with the same G1/G2, change ONE main anchor "
        "(G1 or G2) to a broader or complementary parent concept family; keep the other main anchor plus G3 specific.\n"
        "  4. Put external outcomes, diseases, populations, settings, study designs, and downstream use cases in G3 unless they are the expert's core domain.\n"
        "  5. Use exactly 3 AND (+) groups — do NOT drop to 2 groups.\n"
        f"  6. Keep every `|` group at ≤{_MAX_OR_TERMS_PER_GROUP} terms.\n"
        "  7. Keep at least two strong domain anchors across G1/G2/G3; do NOT drift into an unrelated application area.\n\n"
        f"{_QUERY_SYNTAX}\n\n"
        f"{_QUERY_SHAPE}\n\n"
        "Rules:\n"
        "1. G3 must differ substantively from prior queries, not synonym swaps only.\n"
        f"{rule_2}"
        "3. Do not repeat a prior facet at a narrower granularity when the prior facet yielded few new papers; "
        "switch to a different major facet family instead.\n"
        "4. Do not chase novelty alone: a useful pivot should be different AND likely high-yield.\n"
        "5. Do not simply copy long lists of methods, metrics, or workflow terms from the expert description; "
        "compress them into broader searchable method families.\n"
        "6. If prior queries under-collected, loosen fixed study-design/evidence-context terms and method/tool details before broadening into external topics.\n"
        "7. Do not keep the same study-design/population group as a fixed anchor across repeated pivots.\n"
        "8. If anchor-pivot is active, the query should still clearly express the expert's core domain after removing G3.\n"
        "9. Every term should plausibly appear in a paper title or abstract. Replace abstract profile language with common literature terms.\n"
        "10. In resume searches, avoid facets already implied by the yield history even if they are central; prioritize missed high-yield families.\n"
        "11. All terms must be field-specific; no generic filler terms.\n"
        "12. Use prefix wildcards only for domain-specific stems that remain meaningful after truncation.\n"
        "13. Use `|` between alternatives. Never write the word OR in the query.\n"
        "14. The query MUST be in English.\n"
        "15. "
        f"{_JSON_OUTPUT}"
    )


def _build_broaden_prompt(
    expert_name: str,
    expert_description: str,
    query_yields: list[tuple[str, int]],
    last_query: str,
    last_query_count: int,
    cumulative_total: int,
    *,
    min_papers: int,
    max_papers: int,
    broaden_round: int,
) -> str:
    return (
        "You are a literature retrieval specialist. "
        "The last query under-collected NEW papers for this facet. "
        "Apply ONE in-group broaden step — do not remove AND groups.\n\n"
        f"Expert Name: {expert_name}\n"
        f"Expert Description: {expert_description}\n\n"
        f"{_format_target_band(cumulative_total, min_papers, max_papers)}\n\n"
        f"Last query: {last_query}\n"
        f"New papers from last query: {last_query_count:,}\n"
        f"Cumulative unique papers so far: {cumulative_total:,}\n\n"
        "Query yield history:\n"
        f"{_format_yield_history(query_yields, min_papers=min_papers)}\n\n"
        f"{_QUERY_SYNTAX}\n\n"
        f"{_QUERY_SHAPE}\n\n"
        f"{_TITLE_ABSTRACT_TERM_GUIDANCE}\n\n"
        f"{_WILDCARD_GUIDANCE}\n\n"
        f"{_CORE_ANCHOR_GUIDANCE}\n\n"
        f"{_EVIDENCE_CONTEXT_GUIDANCE}\n\n"
        f"{_METHOD_DETAIL_GUIDANCE}\n\n"
        f"{_broaden_ladder_instruction(broaden_round)}\n\n"
        "Rules:\n"
        "1. Keep exactly 3 AND (+) groups with roles G1 broad field/evidence context, G2 core topic anchor, G3 facet.\n"
        "2. Preserve G2. If the previous query used a method-only G2, rewrite it so methods move into G1 and G2 becomes the core topic anchor.\n"
        f"3. Keep every `|` group at ≤{_MAX_OR_TERMS_PER_GROUP} terms; replace weak terms instead of creating long lists.\n"
        "4. If the previous query under-collected because any group was too method/tool-heavy or too study-design/evidence-context-heavy, "
        "replace details with a parent method family or broader source/context phrase while keeping two domain anchors.\n"
        "5. The new query MUST differ from all previously used queries.\n"
        "6. Do not replace specific terms with generic ones.\n"
        "7. Stay within this expert's connected topical neighborhood; do not broaden into unrelated disciplines.\n"
        "8. Use only terms likely to appear in titles or abstracts; avoid profile-only wording.\n"
        "9. Do not add more named tools/databases/algorithms when broadening; use method families instead.\n"
        "10. Do not keep the same study-design/population evidence group fixed across repeated low-yield pivots.\n"
        "11. Add or preserve prefix wildcards only on specific domain stems; remove broad wildcards that may drift.\n"
        "12. Use `|` between alternatives. Never write the word OR in the query.\n"
        "13. The query MUST be in English.\n"
        "14. "
        f"{_JSON_OUTPUT}"
    )


def _build_narrow_prompt(
    expert_name: str,
    expert_description: str,
    previous_queries: list[str],
    too_many_queries: list[str],
    last_error: str,
    *,
    rejected_query: str,
) -> str:
    prev_qs = "\n".join(f"  [{i}] {q}" for i, q in enumerate(previous_queries, 1))
    too_many = "\n".join(f"  [{i}] {q}" for i, q in enumerate(too_many_queries, 1))
    return (
        "You are a literature retrieval specialist. "
        "The last query was rejected because it returned too many hits (>10M). "
        "Generate a NARROWER query that stays under the bulk-search limit but remains useful.\n\n"
        f"Expert Name: {expert_name}\n"
        f"Expert Description: {expert_description}\n"
        f"Rejected query: {rejected_query}\n"
        f"Last error: {last_error}\n"
        f"Queries already used:\n{prev_qs}\n"
        f"Too-broad queries:\n{too_many}\n\n"
        f"{_QUERY_SYNTAX}\n\n"
        f"{_QUERY_SHAPE}\n\n"
        f"{_TITLE_ABSTRACT_TERM_GUIDANCE}\n\n"
        f"{_WILDCARD_GUIDANCE}\n\n"
        f"{_EVIDENCE_CONTEXT_GUIDANCE}\n\n"
        f"{_METHOD_DETAIL_GUIDANCE}\n\n"
        "Narrowing tactics (apply 1–2 only):\n"
        "1. Keep exactly 3 AND (+) groups; do NOT add a 4th group.\n"
        "2. Remove the broadest `|` alternatives from one or more groups.\n"
        "3. Replace an ambiguous, generic, or drift-prone wildcard with a title/abstract-observable phrase.\n"
        "4. Narrow one existing group to a more specific parent-child concept family while preserving the other anchors.\n"
        "5. Do not narrow by replacing a method family with a long list of named tools, databases, metrics, or algorithms.\n"
        "6. Do not narrow by making study-design/population/evidence-context terms mandatory unless they define the expert domain.\n\n"
        "Rules:\n"
        "1. The new query MUST differ from all previously used queries.\n"
        "2. Do not narrow so much that recall becomes near-zero; exactly 3 AND groups are mandatory even if the prior query was >10M.\n"
        "3. The query MUST be in English.\n"
        "4. "
        f"{_JSON_OUTPUT}"
    )


_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def _extract_query_from_llm_text(raw: str) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None

    candidates: list[str] = []
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    for candidate in candidates:
        try:
            obj = json.loads(candidate.strip())
        except Exception:
            continue
        if isinstance(obj, dict) and "query" in obj:
            query = str(obj["query"]).strip()
            if query:
                return query
    return None


def _request_llm_query_text(client, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You generate Semantic Scholar bulk-search queries."},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
        extra_body={"thinking": {"type": "enabled"}},
        reasoning_effort="high",
    )
    content = resp.choices[0].message.content
    return (content or "").strip()


def _build_query_json_repair_prompt(raw: str) -> str:
    return (
        "Your previous response is not valid JSON with a single \"query\" field. "
        "Repair that response. Keep the intended Semantic Scholar query text; "
        "only fix truncation, fences, escaping, or other JSON formatting problems.\n\n"
        "Previous response:\n"
        f"{raw}\n\n"
        f"{_JSON_OUTPUT}"
    )


def _call_llm_for_query(client, model: str, prompt: str) -> str:
    raw = _request_llm_query_text(client, model, prompt)
    query = _extract_query_from_llm_text(raw)
    if query:
        return query

    print("  LLM query JSON invalid; asking the model to repair its own output ...")
    repaired = _request_llm_query_text(client, model, _build_query_json_repair_prompt(raw))
    query = _extract_query_from_llm_text(repaired)
    if query:
        return query

    raise ValueError(
        "LLM response does not contain a valid JSON query after one repair attempt: "
        f"{repaired or raw}"
    )


class TooManyHitsError(RuntimeError):
    def __init__(self, query: str, error_msg: str) -> None:
        super().__init__(error_msg)
        self.query = query
        self.error_msg = error_msg


def _is_too_many_hits(error_msg: str) -> bool:
    msg = (error_msg or "").lower()
    return "too many hits" in msg and "10000000" in msg


def _generate_query(
    client,
    model: str,
    expert_name: str,
    expert_description: str,
    *,
    min_papers: int,
    max_papers: int,
    query_yields: Optional[list[tuple[str, int]]] = None,
    last_query: str = "",
    last_query_count: int = 0,
    cumulative_total: int = 0,
    broaden_round: int = 0,
    facet_pivot: bool = False,
    anchor_pivot: bool = False,
    facet_index: int = 0,
) -> str:
    repair_instruction = ""
    for attempt in range(3):
        if not query_yields:
            prompt = _build_seed_prompt(
                expert_name,
                expert_description,
                min_papers=min_papers,
                max_papers=max_papers,
                cumulative_total=cumulative_total,
            )
        elif facet_pivot:
            prompt = _build_facet_pivot_prompt(
                expert_name,
                expert_description,
                query_yields,
                cumulative_total=cumulative_total,
                min_papers=min_papers,
                max_papers=max_papers,
                facet_index=facet_index,
                force_anchor_shift=anchor_pivot,
            )
        else:
            prompt = _build_broaden_prompt(
                expert_name,
                expert_description,
                query_yields,
                last_query=last_query or query_yields[-1][0],
                last_query_count=last_query_count,
                cumulative_total=cumulative_total,
                min_papers=min_papers,
                max_papers=max_papers,
                broaden_round=broaden_round,
            )
        if repair_instruction:
            prompt = f"{prompt}\n\n{repair_instruction}"
        query = _normalize_query_syntax(_call_llm_for_query(client, model, prompt))
        failure_reason = _query_shape_failure_reason(query, require_three_groups=True)
        if not failure_reason:
            return query
        if attempt < 2:
            print(
                f"  Query shape check failed ({failure_reason}); retrying ..."
            )
            repair_instruction = (
                "The previous query was rejected by local shape validation:\n"
                f"{failure_reason}\n"
                f"Rejected query: {query}\n"
                "Rewrite it as exactly 3 AND (+) groups: G1 broad field/evidence-context terms, "
                "G2 stable core application/topic anchor, G3 bounded rotating facet. "
                f"Keep each `|` group short (≤{_MAX_OR_TERMS_PER_GROUP} terms), connected to "
                "the expert's topical neighborhood, avoid category-atlas lists, keep at least two strong domain anchors, "
                "use terms likely to appear in paper titles or abstracts, and keep external outcomes/settings as G3 "
                "unless they are the expert's core domain. Use method families instead of tool/detail checklists. "
                "Do not keep study-design/population evidence terms as fixed anchors across low-yield pivots. "
                "Use prefix wildcards only on specific domain stems, never broad stems."
            )
    return query


def _generate_narrow_query(
    client,
    model: str,
    expert_name: str,
    expert_description: str,
    previous_queries: list[str],
    too_many_queries: list[str],
    last_error: str,
    *,
    rejected_query: str,
) -> str:
    repair_instruction = ""
    query = ""
    for attempt in range(3):
        prompt = _build_narrow_prompt(
            expert_name,
            expert_description,
            previous_queries,
            too_many_queries,
            last_error,
            rejected_query=rejected_query,
        )
        if repair_instruction:
            prompt = f"{prompt}\n\n{repair_instruction}"
        query = _normalize_query_syntax(_call_llm_for_query(client, model, prompt))
        failure_reason = _query_shape_failure_reason(query, require_three_groups=True)
        if not failure_reason:
            return query
        if attempt < 2:
            print(f"  Narrow query shape check failed ({failure_reason}); retrying ...")
            repair_instruction = (
                "The previous narrowed query was rejected by local shape validation:\n"
                f"{failure_reason}\n"
                f"Rejected query: {query}\n"
                "Rewrite it as exactly 3 AND (+) groups. Do NOT add a 4th group. "
                "Narrow by removing broad alternatives or replacing ambiguous terms inside the existing 3 groups. "
                "Remove any broad wildcard; keep wildcards only on specific domain stems. "
                "Do not narrow by adding a long list of named tools or algorithms."
            )
    return query


def _load_seen_ids_from_jsonl(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            paper_id = rec.get("paperId")
            if paper_id:
                seen.add(str(paper_id))
    return seen


def _fetch_all_for_query(
    keyword: str,
    s2_headers: dict,
    seen_ids: set[str],
    out_path: Path,
    *,
    publication_date_or_year: str = ":2022-12",
    fields: str = "paperId,title,abstract,year,authors",
) -> tuple[int, int]:
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    params: dict = {
        "query": keyword,
        "fields": fields,
        "publicationDateOrYear": publication_date_or_year,
        "sort": "publicationDate:desc",
    }

    new_count = 0
    total_returned = 0
    page = 0
    stale_streak = 0
    transient_retries = 0
    with out_path.open("a", encoding="utf-8") as out:
        while True:
            resp = requests.get(base_url, params=params, headers=s2_headers, timeout=60)
            if resp.status_code == 429:
                # wait = 2 ** min(page, 8) + (time.monotonic() % 1)
                wait = 30
                print(f"  Rate-limited, waiting {wait:.1f}s ...")
                time.sleep(wait)
                continue
            if resp.status_code in _TRANSIENT_STATUS_CODES:
                transient_retries += 1
                if transient_retries > _MAX_TRANSIENT_RETRIES:
                    print(
                        f"  Transient S2 error {resp.status_code} after "
                        f"{_MAX_TRANSIENT_RETRIES} retries; skipping rest of this query."
                    )
                    break
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else 10 * transient_retries
                except ValueError:
                    wait = 10 * transient_retries
                print(
                    f"  Transient S2 error {resp.status_code}, "
                    f"waiting {wait:.1f}s before retry {transient_retries}/{_MAX_TRANSIENT_RETRIES} ..."
                )
                time.sleep(wait)
                continue
            transient_retries = 0

            body = {}
            try:
                body = resp.json()
            except Exception:
                body = {}

            error_msg = body.get("error", "") if isinstance(body, dict) else ""
            if error_msg:
                if _is_too_many_hits(error_msg):
                    raise TooManyHitsError(keyword, error_msg)
            resp.raise_for_status()
            if not body:
                body = resp.json()

            papers = body.get("data", [])
            total_returned += len(papers)
            added_this_page = 0
            for paper in papers:
                pid = paper.get("paperId")
                if pid and str(pid) not in seen_ids:
                    seen_ids.add(str(pid))
                    # paper.pop("abstract", None)
                    out.write(json.dumps(paper, ensure_ascii=False) + "\n")
                    new_count += 1
                    added_this_page += 1

            page += 1
            print(f"  Page {page}: {len(papers)} returned, {added_this_page} new (unique: {len(seen_ids)})")

            if added_this_page == 0:
                stale_streak += 1
                if stale_streak >= 2:
                    break
            else:
                stale_streak = 0

            if "token" not in body:
                break
            params["token"] = body["token"]
            time.sleep(5)

    return new_count, total_returned


def search_papers(
    expert_name: str,
    expert_description: str,
    *,
    output_path: str | Path = ".",
    start_query: Optional[str] = None,
    model: str = "gpt-4.1",
    min_papers_per_query: int = 100_000,
    max_papers_per_query: int = 1_000_000,
    max_queries: int = _DEFAULT_MAX_QUERIES,
) -> Path:
    out_dir = Path(output_path) / expert_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "search_paper.jsonl"
    query_history_file = out_dir / _QUERY_HISTORY_NAME

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE_URL")
    if not api_key and not start_query:
        raise RuntimeError("OPENAI_API_KEY is required when start_query is not provided.")

    openai_client = None
    if api_key:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("openai package is required for query generation") from exc
        openai_client = OpenAI(api_key=api_key, base_url=base_url)

    s2_headers: dict = {}
    s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if s2_key:
        s2_headers["x-api-key"] = s2_key
        print(f"Using Semantic Scholar API key: {s2_key}")

    seen_ids: set[str] = set()
    too_many_queries: list[str] = []
    if out_file.exists():
        seen_ids.update(_load_seen_ids_from_jsonl(out_file))
        print(f"Loaded {len(seen_ids)} existing paperId(s) from {out_file}")
    else:
        print(f"Starting new jsonl output -> {out_file}")

    query_history = _load_query_history(query_history_file)
    if query_history:
        print(f"Loaded {len(query_history)} prior query record(s) from {query_history_file}")

    query_yields: list[tuple[str, int]] = [
        (str(rec.get("query") or ""), int(rec.get("new_count") or 0))
        for rec in query_history
        if str(rec.get("query") or "").strip()
    ]
    last_record = query_history[-1] if query_history else {}
    last_query_new_count = int(last_record.get("new_count") or 0)
    last_query = str(last_record.get("query") or "")
    last_mode = str(last_record.get("mode") or "")
    broaden_round = 0
    low_yield_streak = 0
    facet_index = sum(
        1 for rec in query_history if rec.get("mode") in ("facet-pivot", "anchor-pivot")
    )
    last_query_saturated = (
        int(last_record.get("total_returned") or 0) > 100
        and int(last_record.get("new_count") or 0)
        / max(1, int(last_record.get("total_returned") or 0))
        < _SATURATION_NEW_RATIO
    )
    yield_threshold = _low_yield_threshold(min_papers_per_query)
    for _, prior_new_count in query_yields[-_FACET_PIVOT_STREAK:]:
        if prior_new_count < yield_threshold:
            low_yield_streak += 1
        else:
            low_yield_streak = 0

    for attempt in range(max_queries):
        if start_query and attempt == 0:
            current_query = _normalize_query_syntax(start_query)
            query_mode = "start"
        else:
            if openai_client is None:
                raise RuntimeError("openai client not available (missing OPENAI_API_KEY or openai package).")
            last_productive = last_query_new_count >= yield_threshold
            query_mode = _choose_query_mode(
                query_yields=query_yields,
                low_yield_streak=low_yield_streak,
                last_mode=last_mode,
                broaden_round=broaden_round,
                last_query_productive=last_productive,
                last_query_saturated=last_query_saturated,
            )
            use_facet_pivot = query_mode in ("facet-pivot", "anchor-pivot")
            current_query = _generate_query(
                client=openai_client,
                model=model,
                expert_name=expert_name,
                expert_description=expert_description,
                min_papers=min_papers_per_query,
                max_papers=max_papers_per_query,
                query_yields=query_yields if query_yields else None,
                last_query=last_query,
                last_query_count=last_query_new_count,
                cumulative_total=len(seen_ids),
                broaden_round=broaden_round,
                facet_pivot=use_facet_pivot,
                anchor_pivot=query_mode == "anchor-pivot",
                facet_index=facet_index,
            )
            if query_mode == "broaden":
                broaden_round += 1
            elif query_mode in ("facet-pivot", "anchor-pivot"):
                facet_index += 1
                broaden_round = 0
        if not current_query:
            continue
        last_query = current_query
        mode_tag = f" ({query_mode})" if query_mode not in ("start", "seed") else ""
        print(f"Query [{attempt + 1}/{max_queries}]{mode_tag}: {current_query}")

        try:
            new_count, total_returned = _fetch_all_for_query(
                keyword=current_query,
                s2_headers=s2_headers,
                seen_ids=seen_ids,
                out_path=out_file,
            )
        except TooManyHitsError as exc:
            too_many_queries.append(exc.query)
            if openai_client is None:
                raise RuntimeError("openai client not available for narrowing query.") from exc
            narrowed_query = _generate_narrow_query(
                client=openai_client,
                model=model,
                expert_name=expert_name,
                expert_description=expert_description,
                previous_queries=[q for q, _ in query_yields],
                too_many_queries=too_many_queries,
                last_error=exc.error_msg,
                rejected_query=exc.query,
            )
            last_query = narrowed_query
            print(f"Query [{attempt + 1}/{max_queries}] (narrowed): {narrowed_query}")
            new_count, total_returned = _fetch_all_for_query(
                keyword=narrowed_query,
                s2_headers=s2_headers,
                seen_ids=seen_ids,
                out_path=out_file,
            )
            _append_query_history(
                query_history_file,
                attempt=attempt + 1,
                mode="narrowed",
                query=narrowed_query,
                new_count=new_count,
                total_returned=total_returned,
                cumulative_total=len(seen_ids),
            )
            query_yields.append((narrowed_query, new_count))
            last_mode = "narrowed"
            broaden_round = 0

        else:
            _append_query_history(
                query_history_file,
                attempt=attempt + 1,
                mode=query_mode,
                query=last_query,
                new_count=new_count,
                total_returned=total_returned,
                cumulative_total=len(seen_ids),
            )
            query_yields.append((last_query, new_count))
            last_mode = query_mode
            if query_mode in ("seed", "start"):
                broaden_round = 0

        last_query_new_count = new_count
        last_query_saturated = (
            total_returned > 100
            and new_count / total_returned < _SATURATION_NEW_RATIO
        )
        if last_query_saturated:
            print(
                f"  Facet saturated ({new_count}/{total_returned} new); "
                "will pivot on next attempt if any remain."
            )
        if new_count < yield_threshold:
            low_yield_streak += 1
        else:
            low_yield_streak = 0

        if len(seen_ids) >= max_papers_per_query:
            print(f"  Reached max papers cap ({max_papers_per_query:,}).")
            break
        if len(seen_ids) >= min_papers_per_query:
            print(f"  Reached min papers target ({min_papers_per_query:,}).")
            break
        if new_count == 0:
            print("  No new papers from this query; will pivot/broaden on next attempt if any remain.")
            continue

    print(f"Done - {len(seen_ids)} unique papers, saved to {out_file}")
    return out_file

