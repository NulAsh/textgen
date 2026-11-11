import json
import subprocess
import sys
import threading

import pytest

from modules.vault_runtime import flush_pending_save


def test_lock_flushes_pending_edits_without_replaying_old_notebook_input():
    saved = []
    timer = threading.Timer(60, lambda: saved.append('edited input'))
    timer.start()
    try:
        flush_pending_save(timer, lambda: saved.append('edited input'))
        assert saved == ['edited input']
        # A generation may have persisted newer output since that timer ran.
        saved.append('new generated output')
        flush_pending_save(timer, lambda: saved.append('stale input'))
        flush_pending_save(None, lambda: saved.append('stale input'))
        assert saved[-1] == 'new generated output'
    finally:
        timer.cancel()
        timer.join()


def test_queue_join_and_stream_without_construction_event_loop():
    pytest.importorskip('gradio')
    result = subprocess.run([sys.executable, '-B', 'tests/vault/probe_queue.py'],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary['queued_streams_completed'] == 2
