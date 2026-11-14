import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from cvat_agent.mask_format import encode_mask, validate_mask
from cvat_agent.sam3_backend import convert_result
from cvat_agent import annotate_batch

class MaskTests(unittest.TestCase):
    def test_pixel_exact_roundtrip(self):
        masks = [np.ones((1, 1), bool), np.eye(9, dtype=bool)]
        ring = np.zeros((12, 13), bool)
        ring[2:10, 3:11] = True
        ring[4:8, 5:9] = False
        masks.append(ring)
        for bitmap in masks:
            points = encode_mask(bitmap)
            validate_mask(points)
            *runs, left, top, right, bottom = points
            crop = np.repeat(np.arange(len(runs)) % 2, runs).reshape(bottom-top+1, right-left+1)
            decoded = np.zeros_like(bitmap)
            decoded[top:bottom+1,left:right+1] = crop
            np.testing.assert_array_equal(bitmap, decoded)
        self.assertIsNone(encode_mask(np.zeros((3, 3), bool)))

    def test_mask_conversion_and_upload(self):
        bitmap = torch.eye(5, dtype=torch.bool)
        result = {'scores': torch.tensor([0.9]), 'masks': bitmap[None], 'boxes': torch.tensor([[0, 0, 5, 5]])}
        shapes = convert_result(result, 'road', 'mask', 1)
        self.assertEqual(shapes[0]['type'], 'mask')
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            predictions = Path(folder) / 'predictions'
            predictions.mkdir()
            (predictions / 'frame_0000.json').write_text(json.dumps(shapes))
            with patch.object(annotate_batch, 'unlock_and_show_labels'), patch.object(annotate_batch, 'evaluate', side_effect=lambda x: calls.append(x) or 200), patch.object(annotate_batch, 'fetch_json', side_effect=[{'version': 1, 'shapes': [], 'tracks': [], 'tags': []}, {'results': [{'id': 1, 'name': 'road'}]}]):
                uploaded = annotate_batch.apply(1, predictions, '', ['road'])
            self.assertEqual(uploaded['added_shapes'], 1)
        expression = next(x for x in calls if "method:'PUT'" in x)
        payload = json.loads(expression.split('body:JSON.stringify(', 1)[1].split(')}).then', 1)[0])
        self.assertEqual(payload['shapes'][0]['points'], shapes[0]['points'])
        self.assertEqual(payload['shapes'][0]['type'], 'mask')

    def test_malformed_mask_rejected(self):
        with self.assertRaises(ValueError):
            validate_mask([0, 99, 0, 0, 1, 1])
