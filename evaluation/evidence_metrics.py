import os
import re


def normalize_text(text: str) -> str:
    """Normalize English text so punctuation and hyphenation do not affect matching."""
    text = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return " ".join(text.split())


def document_key(item: dict) -> tuple:
    """Stable identity for a candidate during ranking/fusion."""
    document = item["document"]
    return (
        os.path.basename(document.get("source", "")).lower(),
        document.get("chunk_id"),
        normalize_text(document.get("text", "")),
    )


def _source_matches(item: dict, expected_source: str | None) -> bool:
    if not expected_source:
        return True

    actual_source = os.path.basename(
        item.get("document", {}).get("source", "")
    )

    return actual_source.casefold() == os.path.basename(
        expected_source
    ).casefold()


def evidence_is_covered(
    results: list[dict],
    evidence_groups: list[list[str]],
    expected_source: str | None = None,
) -> bool:
    """
    Return True only when every required evidence group is covered.

    Each group contains alternative phrases; one alternative must match for
    each group. Required groups may be found across different retrieved chunks.
    """
    if not evidence_groups:
        return False

    relevant_texts = [
        normalize_text(item.get("document", {}).get("text", ""))
        for item in results
        if _source_matches(item, expected_source)
    ]

    if not relevant_texts:
        return False

    for group in evidence_groups:
        alternatives = [
            normalize_text(phrase)
            for phrase in group
            if normalize_text(phrase)
        ]

        if not alternatives:
            return False

        group_found = any(
            phrase in text
            for phrase in alternatives
            for text in relevant_texts
        )

        if not group_found:
            return False

    return True


def hit_at(results: list[dict], case: dict, k: int) -> int:
    """1 if the top-k results collectively contain all required evidence."""
    return int(
        evidence_is_covered(
            results[:max(0, k)],
            case.get("expected_evidence", []),
            case.get("expected_source"),
        )
    )


def first_complete_evidence_rank(results: list[dict], case: dict) -> int | None:
    """Rank at which all required evidence has appeared for the first time."""
    evidence_groups = case.get("expected_evidence", [])
    expected_source = case.get("expected_source")

    for rank in range(1, len(results) + 1):
        if evidence_is_covered(
            results[:rank],
            evidence_groups,
            expected_source,
        ):
            return rank

    return None


def mrr(results: list[dict], case: dict) -> float:
    """Reciprocal rank of the first prefix containing complete evidence."""
    rank = first_complete_evidence_rank(results, case)
    return 1.0 / rank if rank else 0.0
