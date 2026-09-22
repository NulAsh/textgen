import threading

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
