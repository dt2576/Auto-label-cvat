import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image
from transformers import BatchEncoding

from cvat_agent import sam3_backend
from cvat_agent.job_runner import main
from cvat_agent.ui import PAGE, validate_model


class Sam3Tests(unittest.TestCase):
    def test_disconnected_regions_are_not_discarded(self):
        mask = torch.zeros((1, 40, 60), dtype=torch.bool)
        mask[:, 2:15, 2:20] = True
        mask[:, 23:35, 35:55] = True
        result = {'scores': torch.tensor([0.8]), 'boxes': torch.tensor([[2., 2., 55., 35.]]), 'masks': mask}
        items = sam3_backend.convert_result(result, 'vegetation', 'segmentation', 12)
        self.assertEqual(len(items), 2)
        self.assertTrue(all(6 <= len(item['points']) <= 24 for item in items))

    def test_prompt_duplicates_merge_without_merging_distinct_objects(self):
        masks = torch.zeros((2, 20, 40), dtype=torch.bool)
        masks[0, 2:10, 2:10] = True
        masks[1, 2:10, 25:35] = True
        first = {'scores': torch.tensor([0.8, 0.6]),
                 'boxes': torch.tensor([[2., 2., 10., 10.], [25., 2., 35., 10.]]), 'masks': masks}
        duplicate = {'scores': torch.tensor([0.7]), 'boxes': first['boxes'][:1], 'masks': masks[:1]}
        merged = sam3_backend.merge_results([first, duplicate])
        self.assertEqual(len(merged['scores']), 2)
        # Different labels are converted independently: a person inside a bus survives.
        bus = sam3_backend.convert_result(merged, 'bus', 'bbox', 12)
        person = sam3_backend.convert_result(duplicate, 'person', 'bbox', 12)
        self.assertEqual(len(bus + person), 3)

    def test_bus_prompts_include_short_object_names(self):
        self.assertEqual(sam3_backend.prompt_variants({'label': 'bus', 'prompt': 'a bus or coach'}),
                         ['bus', 'city bus', 'coach'])

    def test_empty_prompt_masks_can_have_different_resolution(self):
        empty = {'scores': torch.empty(0), 'boxes': torch.empty((0, 4)),
                 'masks': torch.empty((0, 288, 288), dtype=torch.bool)}
        detected = {'scores': torch.tensor([0.7]), 'boxes': torch.tensor([[1., 1., 10., 10.]]),
                    'masks': torch.ones((1, 20, 30), dtype=torch.bool)}
        self.assertEqual(sam3_backend.merge_results([empty, detected])['masks'].shape, (1, 20, 30))
        self.assertEqual(len(sam3_backend.merge_results([empty, empty])['scores']), 0)

    def test_missing_weights_stops_before_cvat_access(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'SAM3_MODEL_DIR': directory}), \
             patch('cvat_agent.job_runner.ensure_ui') as launch:
            with self.assertRaisesRegex(RuntimeError, 'download-sam3.cmd'):
                main('http://localhost/tasks/1/jobs/2', '', 'sam3', 'segmentation', directory)
            launch.assert_not_called()
            with self.assertRaises(RuntimeError):
                validate_model({'model': 'sam3'})

    def test_text_prompts_reuse_vision_and_write_cvat_polygons_offline(self):
        mask = torch.zeros((1, 20, 30), dtype=torch.bool)
        mask[:, 3:16, 4:24] = True
        result = {'scores': torch.tensor([0.9]), 'boxes': torch.tensor([[4., 3., 24., 16.]]), 'masks': mask}
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        model.get_vision_features.return_value = 'cached-features'
        processor = MagicMock()

        def encode(**kwargs):
            if 'images' in kwargs:
                return BatchEncoding({'pixel_values': torch.zeros(1, 3, 20, 30),
                                      'original_sizes': torch.tensor([[20, 30]])})
            return BatchEncoding({'input_ids': torch.ones((1, 4), dtype=torch.long),
                                  'attention_mask': torch.ones((1, 4), dtype=torch.long)})

        processor.side_effect = encode
        processor.post_process_instance_segmentation.return_value = [result]
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'SAM3_DEVICE': 'cpu'}), \
             patch.object(sam3_backend, 'require_model', return_value=Path(directory)), \
             patch('transformers.Sam3Model.from_pretrained', return_value=model) as load_model, \
             patch('transformers.Sam3Processor.from_pretrained', return_value=processor) as load_processor:
            images = Path(directory) / 'images'
            images.mkdir()
            Image.new('RGB', (30, 20)).save(images / 'frame_0020.png')
            output = Path(directory) / 'output'
            labels = sam3_backend.infer_job(images, output, [
                {'label': 'road', 'prompt': 'road surface'},
                {'label': 'vegetation', 'prompt': 'green vegetation'}], 'segmentation', 12)
            items = json.loads((output / 'frame_0020.json').read_text())
            self.assertEqual(labels, ['road', 'vegetation'])
            self.assertEqual([item['label'] for item in items], labels)
            self.assertTrue(all(item['type'] == 'polygon' and 6 <= len(item['points']) <= 24 for item in items))
            self.assertEqual(model.get_vision_features.call_count, 1)
            self.assertEqual(model.call_count, 6)
            self.assertEqual(processor.post_process_instance_segmentation.call_args.kwargs['threshold'], 0.3)
            self.assertTrue(load_model.call_args.kwargs['local_files_only'])
            self.assertTrue(load_processor.call_args.kwargs['local_files_only'])
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
            self.assertEqual(os.environ['HF_HUB_DISABLE_TELEMETRY'], '1')
        boxes = sam3_backend.convert_result(result, 'car', 'bbox', 12)
        self.assertEqual(boxes[0]['points'], [4., 3., 24., 16.])

    def test_ui_exposes_model_selection(self):
        self.assertIn('id="model"', PAGE)
        self.assertIn('id="confidence"', PAGE)
        self.assertIn('model:$(\'model\').value', PAGE)
        self.assertIn('download-sam3.cmd', PAGE)


if __name__ == '__main__':
    unittest.main()
