import unittest

from evaluation.evidence_metrics import (
    evidence_is_covered,
    first_complete_evidence_rank,
    hit_at,
    mrr,
)


SOURCE = "The-Little-Prince-for-Level-2-PDF.pdf"


def result(text, source=SOURCE, chunk_id=0):
    return {
        "document": {
            "source": f"data/pdf/{source}",
            "chunk_id": chunk_id,
            "text": text,
        },
        "score": 0.5,
    }


class EvidenceMetricsTests(unittest.TestCase):

    def test_single_evidence_group_ignores_punctuation_and_hyphenation(self):
        case = {
            "expected_source": SOURCE,
            "expected_evidence": [["forty-three times"]],
        }
        results = [result("One day I saw the sunset forty three times.")]

        self.assertEqual(hit_at(results, case, 1), 1)
        self.assertEqual(mrr(results, case), 1.0)

    def test_all_required_groups_must_be_present(self):
        case = {
            "expected_source": SOURCE,
            "expected_evidence": [["most handsome"], ["richest"]],
        }
        results = [
            result("He was the most handsome man.", chunk_id=1),
            result("He believed he was the richest man.", chunk_id=2),
        ]

        self.assertEqual(hit_at(results, case, 1), 0)
        self.assertEqual(hit_at(results, case, 2), 1)
        self.assertEqual(first_complete_evidence_rank(results, case), 2)
        self.assertEqual(mrr(results, case), 0.5)

    def test_alternative_phrases_within_a_group_are_or_conditions(self):
        case = {
            "expected_source": SOURCE,
            "expected_evidence": [
                ["sheep jumped over the fence", "sheep can come to the flower"]
            ],
        }
        results = [result("The sheep can come to the flower at night.")]

        self.assertTrue(
            evidence_is_covered(
                results,
                case["expected_evidence"],
                case["expected_source"],
            )
        )

    def test_wrong_source_does_not_count_as_evidence(self):
        case = {
            "expected_source": SOURCE,
            "expected_evidence": [["draw me a sheep"]],
        }
        results = [
            result(
                "The little prince said draw me a sheep.",
                source="another-document.pdf",
            )
        ]

        self.assertEqual(hit_at(results, case, 1), 0)
        self.assertEqual(mrr(results, case), 0.0)

    def test_empty_evidence_is_not_automatically_a_hit(self):
        self.assertFalse(evidence_is_covered([], []))


if __name__ == "__main__":
    unittest.main()
