"""Publish complete encrypted snapshots to an optional persistent folder."""

import os
import tempfile
from pathlib import Path


def save_encrypted_backup(directory, content):
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.vault-snapshot-', dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        latest = directory / 'vault-latest.sqlite3'
        previous = directory / 'vault-previous.sqlite3'
        # Finish writing before changing either recoverable snapshot. A
        # cloud-backed filesystem can still interrupt or delay synchronization.
        if latest.exists():
            os.replace(latest, previous)
        os.replace(temporary, latest)
    finally:
        temporary.unlink(missing_ok=True)
