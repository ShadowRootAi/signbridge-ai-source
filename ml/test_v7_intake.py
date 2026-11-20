"""Contract tests using existing real GSL videos, never new evaluation data.

Temporary metadata intentionally exercise declarations; they do not establish
new participants or real consent. A tiny injected registry enables positive
contract cases; the full default registry correctly rejects every GSL fixture.
"""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from v7_intake import ROOT, digest, validate_manifest


class IntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.videos = sorted((ROOT / 'data/open-set-v2/clips').glob('*.mp4'))[:4]
        if len(cls.videos) != 4:
            raise unittest.SkipTest('Existing real GSL video fixtures unavailable; no media generated.')

    def setUp(self):
        runtime = (ROOT / '.runtime-tmp').resolve()
        runtime.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='v7-intake-test-', dir=runtime)
        self.base = Path(self.temporary.name).resolve()

        def cleanup():
            assert self.base.is_relative_to(runtime)
            self.temporary.cleanup()
        self.addCleanup(cleanup)
        self.collection = self.base / 'collection'
        self.collection.mkdir()
        # Distinct real media, copied without modification; no generated signs.
        for i, source in enumerate(self.videos[1:]):
            shutil.copyfile(source, self.collection / f'fixture-{i}.mp4')
        self.reference = self.base / 'test-only-reference.json'
        self.reference.write_text(json.dumps({'samples': [{'sha256': digest(self.videos[0])}]}), encoding='utf-8')
        self.manifest = self.collection / 'manifest.json'
        self.document = {'schema_version': 1, 'novel_final_unknown_categories': ['u_final_only'], 'samples': []}
        for i, split in enumerate(('train', 'calibration', 'final')):
            filename = f'fixture-{i}.mp4'
            self.document['samples'].append({
                'clip_id': f'clip_{i}', 'signer_id': f'P{i+1:03}', 'split': split,
                'consent_local_ml': True, 'human_recorded': True, 'new_participant_declared': True,
                'label': {'kind': 'known', 'id': 'ΓΕΙΑ'} if i == 0 else
                         {'kind': 'unknown', 'category_id': 'u_calibration' if i == 1 else 'u_final_only'},
                'source_path': filename, 'source_sha256': digest(self.collection / filename),
                'clip_path': filename, 'clip_sha256': digest(self.collection / filename),
            })

    def validate(self, document=None, full_registry=False):
        self.manifest.write_text(json.dumps(self.document if document is None else document, ensure_ascii=False), encoding='utf-8')
        return validate_manifest(self.manifest, None if full_registry else [self.reference])

    def codes(self, report):
        return {issue['code'] for issue in report['issues']}

    def test_contract_pass_is_not_identity_or_independence_verification(self):
        report = self.validate()
        self.assertTrue(report['valid'])
        self.assertEqual(report['verified_media_file_count'], 3)
        self.assertEqual(report['split_counts'], {'train': 1, 'calibration': 1, 'final': 1})
        self.assertFalse(report['independent_evaluation_verified'])
        text = json.dumps(report)
        for private in ('P001', 'fixture-0.mp4', str(self.base)):
            self.assertNotIn(private, text)

    def test_real_gsl_fixtures_are_rejected_by_default_registry(self):
        result = self.validate(full_registry=True)
        self.assertFalse(result['valid'])
        self.assertIn('existing_gsl_source_detected', self.codes(result))

    def test_cli_rejects_gsl_with_exit_two_and_no_private_paths(self):
        self.validate()
        command = [sys.executable, '-B', str(ROOT / 'ml/v7_intake.py'), str(self.manifest)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stderr, '')
        self.assertFalse(json.loads(completed.stdout)['valid'])
        self.assertNotIn(str(self.base), completed.stdout)
        self.assertNotIn('P001', completed.stdout)

    def test_true_boolean_declarations_required(self):
        for key in ('consent_local_ml', 'human_recorded', 'new_participant_declared'):
            for value in (False, 'true', 1):
                with self.subTest(key=key, value=value):
                    doc = copy.deepcopy(self.document); doc['samples'][0][key] = value
                    self.assertIn(key + '_required', self.codes(self.validate(doc)))

    def test_both_original_and_clip_hashes_are_recomputed(self):
        for key in ('source', 'clip'):
            with self.subTest(key=key):
                doc = copy.deepcopy(self.document); doc['samples'][0][key + '_sha256'] = '0' * 64
                self.assertIn(key + '_digest_mismatch', self.codes(self.validate(doc)))

    def test_renaming_media_does_not_hide_exact_duplicates(self):
        shutil.copyfile(self.collection / 'fixture-0.mp4', self.collection / 'renamed.mp4')
        second = self.document['samples'][1]
        for key in ('source', 'clip'):
            second[key + '_path'] = 'renamed.mp4'
            second[key + '_sha256'] = digest(self.collection / 'renamed.mp4')
        self.assertIn('duplicate_clip_or_original_source', self.codes(self.validate()))

    def test_shared_original_is_rejected_even_with_distinct_clips(self):
        self.document['samples'][1]['source_path'] = self.document['samples'][0]['source_path']
        self.document['samples'][1]['source_sha256'] = self.document['samples'][0]['source_sha256']
        self.assertIn('duplicate_clip_or_original_source', self.codes(self.validate()))

    def test_one_signer_cannot_cross_splits(self):
        self.document['samples'][1]['signer_id'] = 'P001'
        self.assertIn('signer_crosses_splits', self.codes(self.validate()))

    def test_duplicate_clip_id(self):
        self.document['samples'][1]['clip_id'] = 'clip_0'
        self.assertIn('duplicate_clip_id', self.codes(self.validate()))

    def test_known_and_unknown_label_contracts_are_distinct(self):
        for label, code in [({'kind': 'known', 'id': 'ΑΛΛΟ'}, 'known_label_invalid'),
                            ({'kind': 'unknown', 'category_id': 'ΓΕΙΑ'}, 'unknown_category_invalid'),
                            ({'kind': 'unknown', 'category_id': 'u_ok', 'id': 'ΓΕΙΑ'}, 'unknown_category_invalid')]:
            with self.subTest(code=code):
                doc = copy.deepcopy(self.document); doc['samples'][0]['label'] = label
                self.assertIn(code, self.codes(self.validate(doc)))

    def test_novel_final_category_excluded_from_train_and_calibration(self):
        for index in (0, 1):
            with self.subTest(index=index):
                doc = copy.deepcopy(self.document)
                doc['samples'][index]['label'] = {'kind': 'unknown', 'category_id': 'u_final_only'}
                self.assertIn('novel_final_category_in_development', self.codes(self.validate(doc)))

    def test_final_category_must_be_registered(self):
        self.document['novel_final_unknown_categories'] = []
        self.assertIn('final_unknown_not_declared_novel', self.codes(self.validate()))

    def test_collection_path_escape_is_rejected_without_disclosing_path(self):
        outside = self.base / 'private-name.mp4'; shutil.copyfile(self.videos[1], outside)
        self.document['samples'][0]['clip_path'] = '../private-name.mp4'
        result = self.validate()
        self.assertIn('clip_path_outside_collection', self.codes(result))
        self.assertNotIn('private-name', json.dumps(result))

    def test_missing_registry_fails_closed(self):
        self.validate()
        result = validate_manifest(self.manifest, [self.base / 'missing.json'])
        self.assertEqual(self.codes(result), {'gsl_reference_registry_unavailable_or_invalid'})

    def test_duplicate_json_keys_fail_closed(self):
        self.manifest.write_text('{"schema_version":1,"schema_version":1}', encoding='utf-8')
        self.assertEqual(self.codes(validate_manifest(self.manifest)), {'manifest_unreadable_or_invalid_json'})

    def test_empty_template_is_not_accepted_as_data(self):
        result = self.validate({'schema_version': 1, 'novel_final_unknown_categories': [], 'samples': []})
        self.assertIn('samples_must_be_nonempty_list', self.codes(result))


if __name__ == '__main__':
    unittest.main()
