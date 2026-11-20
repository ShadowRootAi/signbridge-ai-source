"""Focused scorer tests; synthetic token examples, no corpus/model inference."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evaluate_scaled_phrases import edit_counts, evaluate_paired_rows, evaluate_rows, normalize_text


def example(identity="a", **changes):
    row = {"id": identity, "signer": "6", "category": "unseen_template_known_glosses",
           "composition": "unseen_template", "split": "test", "gold_glosses": ["ΕΓΩ", "ΘΕΛΩ", "ΒΙΒΛΙΟ"],
           "predicted_glosses": ["ΕΓΩ", "ΘΕΛΩ", "ΒΙΒΛΙΟ"], "reference": "Εγώ θέλω βιβλίο.",
           "text": "εγώ θέλω βιβλίο", "input_video_id": identity}
    return {**row, **changes}


class ScorerTests(unittest.TestCase):
    def test_known_substitution_deletion_and_insertion(self):
        self.assertEqual(edit_counts(["a", "b", "c"], ["a", "d", "c", "e"]),
                         {"correct": 2, "substitutions": 1, "deletions": 0, "insertions": 1, "edits": 2})
        self.assertEqual(edit_counts(["a", "b", "c"], ["a", "c"])["deletions"], 1)
        self.assertEqual(edit_counts([], ["x", "y"])["insertions"], 2)
        self.assertEqual(edit_counts(["x", "y"], [])["deletions"], 2)

    def test_alignment_conservation_in_ambiguous_cases(self):
        for gold, pred in [(["a", "b"], ["b", "a"]), (["a", "a", "b"], ["a", "b", "b"]), ([], [])]:
            counts = edit_counts(gold, pred)
            self.assertEqual(counts["correct"] + counts["substitutions"] + counts["deletions"], len(gold))
            self.assertEqual(counts["correct"] + counts["substitutions"] + counts["insertions"], len(pred))
            self.assertEqual(counts["edits"], sum(counts[k] for k in ("substitutions", "deletions", "insertions")))

    def test_unicode_and_names_are_retained(self):
        self.assertEqual(normalize_text("  ΓΕΙΑ,  COVID-19! "), "γεια covid 19")
        self.assertEqual(normalize_text("θε\u0301λω"), "θέλω")
        self.assertNotEqual(normalize_text("θέλω"), normalize_text("θελω"))
        self.assertNotEqual(normalize_text("θέλω COVID"), normalize_text("θέλω"))

    def test_exact_normalized_and_verbatim_distinguished(self):
        score = evaluate_rows([example()])
        self.assertEqual(score["gloss_sequence_exact"], 1)
        self.assertEqual(score["greek_reference_exact"], 1)
        self.assertEqual(score["greek_verbatim_exact"], 0)
        self.assertEqual(score["accepted_text_exact_rate"], 1)

    def test_abstention_has_full_word_deletions_and_no_false_precision(self):
        score = evaluate_rows([example(text=None, predicted_glosses=[])])
        self.assertEqual(score["greek_wer"], 1)
        self.assertEqual(score["gloss_wer"], 1)
        self.assertEqual(score["greek_edit_counts"]["deletions"], 3)
        self.assertEqual(score["text_coverage"], 0)
        self.assertEqual(score["greek_reference_exact"], 0)
        self.assertIsNone(score["accepted_text_exact_rate"])

    def test_insertions_can_make_wer_exceed_one(self):
        score = evaluate_rows([example(reference="ναι", text="όχι όχι όχι")])
        self.assertEqual(score["greek_wer"], 3)
        self.assertEqual(score["incorrect_accepted_texts"], 1)

    def test_corpus_micro_average_and_disjoint_subgroups(self):
        rows = [example(), example("b", signer=7, category="unseen_template_unseen_glosses",
                                  reference="ναι", text=None, gold_glosses=["ΝΑΙ"], predicted_glosses=[])]
        score = evaluate_rows(rows)
        self.assertEqual(score["greek_wer"], .25)
        self.assertEqual(score["gloss_wer"], .25)
        self.assertEqual(score["text_coverage"], .5)
        self.assertEqual(score["by_signer"]["7"]["greek_wer"], 1)
        self.assertEqual(score["by_category"]["unseen_template_known_glosses"]["utterances"], 1)
        self.assertEqual(score["by_composition"]["unseen_template"]["utterances"], 2)

    def test_explicit_unspecified_metadata_and_no_mutation(self):
        row = example()
        for key in ("signer", "category", "composition", "split"):
            row.pop(key)
        original = copy.deepcopy(row)
        self.assertEqual(evaluate_rows([row])["by_signer"]["unspecified"]["utterances"], 1)
        self.assertEqual(row, original)

    def test_rejects_invalid_inputs_and_duplicate_ids(self):
        invalid = [[], [example(), example()], [example(text=3)], [example(reference="...")],
                   [example(gold_glosses=[])], [example(predicted_glosses="ΝΑΙ")],
                   [example(predicted_glosses=["ΔΥΟ ΛΕΞΕΙΣ"])], [example(accepted=False)],
                   [example(text=None, accepted=True)], [example(accepted=1)]]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate_rows(rows)

    def test_refuses_to_pool_counterfactual_conditions(self):
        with self.assertRaises(ValueError):
            evaluate_rows([example(condition="correct"), example("b", condition="mismatched")])

    def test_paired_improvement_and_source_checks(self):
        correct = [example(condition="correct", input_frames=48)]
        wrong = [example(condition="mismatched", input_video_id="donor", input_frames=48,
                         input_signer="6", input_reference="όχι ευχαριστώ", text="όχι", predicted_glosses=["ΟΧΙ"])]
        score = evaluate_paired_rows(correct, wrong)
        self.assertEqual(score["pairs"], 1)
        self.assertEqual(score["greek_correct_video_better"], 1)
        self.assertEqual(score["gloss_correct_video_better"], 1)
        self.assertEqual(score["verified_equal_input_frames"], 1)
        self.assertEqual(score["verified_distinct_input_references"], 1)
        self.assertEqual(score["verified_same_input_signer"], 1)

    def test_identical_counterfactual_outputs_not_claimed_as_grounding(self):
        score = evaluate_paired_rows([example()], [example(input_video_id="donor")])
        self.assertEqual(score["correct_video"]["greek_reference_exact"], 1)
        self.assertEqual(score["mismatched_video"]["greek_reference_exact"], 1)
        self.assertEqual(score["greek_ties"], 1)
        self.assertEqual(score["same_nonempty_text_predictions"], 1)
        self.assertEqual(score["verified_same_input_signer"], 0)
        empty = evaluate_paired_rows([example(text=None)], [example(text=None, input_video_id="donor")])
        self.assertEqual(empty["same_text_predictions"], 1)
        self.assertEqual(empty["same_nonempty_text_predictions"], 0)

    def test_paired_targets_and_controls_must_match(self):
        invalid = [example("other", input_video_id="donor"), example(),
                   example(input_video_id="donor", reference="άλλη αναφορά"),
                   example(input_video_id="donor", category="other"),
                   example(input_video_id="donor", input_reference="Εγώ θέλω βιβλίο!"),
                   example(input_video_id="donor", input_signer="7"),
                   example(input_video_id="donor", input_frames=47)]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                evaluate_paired_rows([example(input_frames=48)], [row])

    def test_cli_binds_frozen_provenance_and_fails_before_output_on_bad_hash(self):
        workspace = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="scaled-scorer-test-", dir=workspace / ".runtime-tmp") as folder:
            folder = Path(folder)
            self.assertTrue(folder.resolve().is_relative_to(workspace))
            freeze, predictions, output = [folder / name for name in ("freeze.json", "predictions.json", "score.json")]
            freeze.write_text('{"fitted": true}', encoding="utf-8")
            predictions.write_text(json.dumps([example()], ensure_ascii=False), encoding="utf-8")
            checksum = hashlib.sha256(freeze.read_bytes()).hexdigest()
            command = [sys.executable, "-B", str(Path(__file__).with_name("evaluate_scaled_phrases.py")),
                       "--predictions", str(predictions), "--freeze", str(freeze), "--output", str(output),
                       "--expected-freeze-sha256"]
            bad = subprocess.run(command + ["0" * 64], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertFalse(output.exists())
            good = subprocess.run(command + [checksum], capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(json.loads(output.read_bytes())["provenance"]["freeze_sha256"], checksum)


if __name__ == "__main__":
    unittest.main()
