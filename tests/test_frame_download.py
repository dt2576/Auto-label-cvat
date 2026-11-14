import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvat_agent.job_runner import collect_images


class FrameDownloadTests(unittest.TestCase):
    def test_retry_then_nested_mixed_formats_in_numeric_order(self):
        calls = []

        def download(job_id, destination, chunk):
            calls.append(chunk)
            if len(calls) == 1:
                raise TimeoutError('network timeout')
            source = destination / f'job_{job_id}_chunk_{chunk}' / 'nested'
            source.mkdir(parents=True)
            (source / 'image_10.JPG').write_bytes(b'ten')
            (source / 'image_2.png').write_bytes(b'two')

        with tempfile.TemporaryDirectory() as directory, \
             patch('cvat_agent.job_runner.save_chunk', side_effect=download), \
             patch('cvat_agent.job_runner.time.sleep'):
            images = collect_images(9, 20, 2, Path(directory))
            self.assertEqual((images / 'frame_0020.png').read_bytes(), b'two')
            self.assertEqual((images / 'frame_0021.jpg').read_bytes(), b'ten')
        self.assertEqual(calls, [0, 0])

    def test_failure_reports_original_error(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('cvat_agent.job_runner.save_chunk', side_effect=RuntimeError('HTTP 403')), \
             patch('cvat_agent.job_runner.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 403'):
                collect_images(9, 20, 2, Path(directory))


if __name__ == '__main__':
    unittest.main()
