"""Cooperative pause at safe boundaries, retaining inference state in memory."""
import os
import time
from pathlib import Path


def checkpoint():
    filename = os.environ.get('CVAT_PAUSE_FILE')
    if not filename:
        return
    marker = Path(filename)
    if not marker.exists():
        return
    print('[control] paused', flush=True)
    while marker.exists():
        time.sleep(0.25)
    print('[control] resumed', flush=True)
