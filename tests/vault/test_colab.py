import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from modules.vault_colab import ColabIsolation, verify_sandbox
from modules.vault_gateway import parse_args
from modules.vault_store import VaultError


@pytest.mark.parametrize('origin', ['https://session-7860-colab.googleusercontent.com', 'https://7860-session.us-central1.colab.dev'])
def test_exact_colab_https_origin(origin):
    options, backend = parse_args(['--vault-colab-origin', origin, '--vault-isolation', 'colab', '--portable'])
    assert options.vault_colab_origin == origin
    assert backend == ['--portable']


@pytest.mark.parametrize('origin', ['http://session-colab.googleusercontent.com', 'https://evil.example',
                                  'https://session-colab.googleusercontent.com.evil.example',
                                  'https://session-colab.googleusercontent.com/path',
                                  'https://user:secret@session-colab.googleusercontent.com'])
def test_invalid_colab_origin_is_rejected(origin):
    with pytest.raises(SystemExit):
        parse_args(['--vault-colab-origin', origin])


def test_live_database_cannot_be_backup_destination(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(['--vault', str(tmp_path / 'vault-latest.sqlite3'), '--vault-backup-dir', str(tmp_path)])


def test_colab_does_not_silently_run_as_notebook_user(monkeypatch):
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    with pytest.raises(VaultError, match='unprivileged worker UID'):
        ColabIsolation()


def test_colab_worker_refuses_unconfined_entry():
    ram = Path(tempfile.mkdtemp(prefix='textgen-colab-entry-', dir='/dev/shm'))
    try:
        with pytest.raises(VaultError):
            verify_sandbox(ram, os.getpid())
    finally:
        shutil.rmtree(ram)


@pytest.mark.skipif(os.geteuid() != 0, reason='Colab uses a root notebook runtime; run this test under sudo in CI.')
def test_native_colab_filesystem_isolation():
    isolation = ColabIsolation()
    try:
        isolation.preflight()
    finally:
        isolation.close()


def test_colab_notebook_uses_fork_private_gateway_and_no_cached_output():
    notebook = json.loads(Path('Colab-Encrypted-TextGen.ipynb').read_text())
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    for cell in cells:
        code = ''.join(cell['source'])
        compile(code, 'Colab notebook', 'exec')
        assert cell['outputs'] == [] and cell['execution_count'] is None
        assert '--share' not in code and 'silence.m4a' not in code
    source = '\n'.join(''.join(cell['source']) for cell in cells)
    assert 'https://github.com/NulAsh/textgen.git' in source
    assert 'cache_in_notebook=False' in source
    assert '--vault-isolation' in source and '--vault-colab-origin' in source
    assert '--vault-backup-dir' in source
    assert 'modules.vault_colab' in source
    # Password entry belongs to the gateway, never an executed/saved cell.
    assert 'getpass' not in source and 'input(' not in source
