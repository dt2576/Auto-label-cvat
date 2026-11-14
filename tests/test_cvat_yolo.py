import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvat_agent import annotate_batch
from cvat_agent.yolo_infer import label_mapping


class CvatYoloTests(unittest.TestCase):
    def test_annotation_policies_and_empty_reset(self):
        original = {'version': 4, 'tags': [], 'tracks': [], 'shapes': [
            {'id': 1, 'frame': 0, 'label_id': 1, 'type': 'rectangle', 'points': [0, 0, 5, 5]},
            {'id': 2, 'frame': 0, 'label_id': 2, 'type': 'polygon', 'points': [0, 0, 9, 0, 9, 9]}]}
        for policy, empty, expected_added, expected_cleared in [
                ('keep', False, 0, 0), ('append', False, 1, 0),
                ('replace_selected', False, 1, 1), ('replace_selected', True, 0, 1)]:
            with self.subTest(policy=policy, empty=empty), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory) / 'predictions'
                folder.mkdir()
                items = [] if empty else [{'label': 'bus', 'type': 'rectangle', 'points': [1, 1, 6, 6]}]
                (folder / 'frame_0000.json').write_text(json.dumps(items))
                calls = []
                with patch.object(annotate_batch, 'unlock_and_show_labels'), \
                     patch.object(annotate_batch, 'evaluate', side_effect=lambda value: calls.append(value) or 200), \
                     patch.object(annotate_batch, 'fetch_json', side_effect=[original, {
                         'results': [{'name': 'bus', 'id': 1}], 'next': None}]):
                    result = annotate_batch.apply(1, folder, '', ['bus'], annotation_policy=policy)
                self.assertEqual(result['added_shapes'], expected_added)
                self.assertEqual(result['cleared_shapes'], expected_cleared)
                expression = next(value for value in calls if "method:'PUT'" in value)
                payload = json.loads(expression.split('body:JSON.stringify(', 1)[1].split(')}).then', 1)[0])
                self.assertIn(annotate_batch.without_server_ids(original['shapes'][1]), payload['shapes'])
                self.assertTrue(Path(result['backup']).exists())

    def test_sam3_empty_detection_preserves_existing_annotations_and_backup(self):
        original = {'version': 1, 'tags': [], 'tracks': [],
                    'shapes': [{'id': 7, 'frame': 0, 'label_id': 1, 'type': 'rectangle', 'points': [1, 2, 3, 4]}]}
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / 'predictions'
            predictions.mkdir()
            (predictions / 'frame_0000.json').write_text('[]')
            with patch.object(annotate_batch, 'unlock_and_show_labels'), \
                 patch.object(annotate_batch, 'evaluate', side_effect=lambda expression: calls.append(expression) or 200), \
                 patch.object(annotate_batch, 'fetch_json', side_effect=[original, {
                     'results': [{'name': 'bus', 'id': 1}], 'next': None}]):
                result = annotate_batch.apply(1, predictions, '', ['bus'], preserve_empty=True)
            self.assertEqual(result['cleared_shapes'], 0)
            backup = next(Path(directory).glob('annotations_before_upload_*.json'))
            self.assertEqual(json.loads(backup.read_text()), original)
        expression = next(call for call in calls if "method:'PUT'" in call)
        payload = json.loads(expression.split('body:JSON.stringify(', 1)[1].split(')}).then', 1)[0])
        self.assertEqual(payload['shapes'], annotate_batch.without_server_ids(original)['shapes'])

    def test_alias_and_unsupported_classes(self):
        self.assertEqual(label_mapping({0: 'person', 1: 'car', 2: 'stop sign'},
                                       ['pedestrian', 'car', 'road']),
                         {0: 'pedestrian', 1: 'car'})

    def test_upload_preserves_other_labels_frames_and_tracks(self):
        original = {
            'version': 7, 'tags': [{'id': 9, 'frame': 0, 'label_id': 3}],
            'tracks': [{'id': 10, 'label_id': 1, 'shapes': [{'id': 11, 'frame': 0}],
                        'elements': [{'id': 12, 'label_id': 4, 'shapes': [{'id': 13, 'frame': 0}]}]}],
            'shapes': [{'id': 1, 'frame': 0, 'label_id': 1},
                       {'id': 2, 'frame': 0, 'label_id': 2},
                       {'id': 3, 'frame': 1, 'label_id': 1}],
        }
        calls = []

        def evaluate(expression):
            calls.append(expression)
            return 200

        with tempfile.TemporaryDirectory() as directory:
            prediction_dir = Path(directory) / 'predictions'
            prediction_dir.mkdir()
            (prediction_dir / 'frame_0000.json').write_text(json.dumps([
                {'label': 'car', 'type': 'rectangle', 'points': [1, 2, 3, 4]}
            ]))
            with patch.object(annotate_batch, 'unlock_and_show_labels'), \
                 patch.object(annotate_batch, 'evaluate', side_effect=evaluate), \
                 patch.object(annotate_batch, 'fetch_json', side_effect=[original, {
                     'results': [{'name': 'car', 'id': 1}, {'name': 'road', 'id': 2}],
                     'next': None,
                 }]):
                result = annotate_batch.apply(123, prediction_dir, '', ['car'], annotation_policy='replace_selected')
        expression = next(call for call in calls if "method:'PUT'" in call)
        payload = json.loads(expression.split('body:JSON.stringify(', 1)[1].split(')}).then', 1)[0])
        clean = annotate_batch.without_server_ids(original)
        self.assertEqual(payload['shapes'][:2], clean['shapes'][1:])
        self.assertEqual(payload['tags'], clean['tags'])
        self.assertEqual(payload['tracks'], clean['tracks'])
        self.assertEqual(payload['tracks'][0]['elements'][0]['label_id'], 4)
        self.assertNotIn('"id":', json.dumps(payload))
        self.assertEqual(result['cleared_shapes'], 1)
        self.assertEqual(result['added_shapes'], 1)


if __name__ == '__main__':
    unittest.main()
