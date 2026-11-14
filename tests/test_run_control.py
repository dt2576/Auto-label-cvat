import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from cvat_agent.run_control import checkpoint

class PauseTests(unittest.TestCase):
    def test_pause_waits_until_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'pause'
            marker.touch()
            done = threading.Event()
            with patch.dict(os.environ, {'CVAT_PAUSE_FILE': str(marker)}):
                worker = threading.Thread(target=lambda: (checkpoint(), done.set()), daemon=True)
                worker.start()
                try:
                    self.assertFalse(done.wait(0.1))
                finally:
                    marker.unlink(missing_ok=True)
                self.assertTrue(done.wait(2))
                worker.join()

    def test_without_pause_marker_returns(self):
        with patch.dict(os.environ, {'CVAT_PAUSE_FILE': ''}):
            checkpoint()
