import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.vault_colab import ColabIsolation, create_syscall_filter, verify_sandbox
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


@pytest.mark.parametrize('expose', ['none', 'disk', 'proc-sys'])
def test_real_mount_verifier_distinguishes_hidden_and_writable_mounts(tmp_path, expose):
    binary = shutil.which('bwrap')
    if not binary:
        pytest.skip('Install bubblewrap to test actual mount layouts.')
    ram = Path(tempfile.mkdtemp(prefix='textgen-mount-test-', dir='/dev/shm'))
    filter_fd = create_syscall_filter()
    try:
        # Exercise the mount verifier under the current test identity; the
        # separate root CI tests additionally check the production UID drop.
        code = ('import ctypes, errno; libc = ctypes.CDLL(None, use_errno=True); '
                'assert libc.unshare(0x10000000) == -1 and ctypes.get_errno() == errno.EPERM; '
                f'from modules.vault_colab import verify_mounts; verify_mounts({str(ram)!r})')
        command = ColabIsolation.command(SimpleNamespace(binary=binary, filter_fd=filter_fd), ram, [sys.executable, '-B', '-c', code])
        if expose != 'none':
            index = command.index('--')
            command[index:index] = (['--bind', str(tmp_path), str(tmp_path)] if expose == 'disk'
                                    else ['--tmpfs', '/proc/sys'])
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30, pass_fds=(filter_fd,))
        if expose != 'none':
            assert result.returncode != 0
            assert ('unexpected writable mount' if expose == 'disk' else 'global procfs controls') in result.stderr
            assert (str(tmp_path) if expose == 'disk' else '/proc/sys') in result.stderr
            if expose == 'disk':
                assert 'type=' in result.stderr and 'flags=' in result.stderr
        else:
            assert result.returncode == 0, result.stderr
    finally:
        os.close(filter_fd)
        shutil.rmtree(ram)


@pytest.mark.parametrize('mask', ['null', 'file', 'nested-directory'])
def test_inherited_proc_masks_stay_readonly_without_blocking_thread_names(tmp_path, mask):
    """Build container-style procfs submounts before constructing the worker."""
    binary = shutil.which('bwrap')
    if not binary:
        pytest.skip('Install bubblewrap to test inherited container mounts.')
    apparmor = Path('/proc/sys/kernel/apparmor_restrict_unprivileged_userns')
    if os.geteuid() != 0 and apparmor.exists() and apparmor.read_text().strip() == '1':
        pytest.skip('The container-mount fixture needs root with AppArmor userns restrictions; CI runs it under sudo.')
    if mask == 'nested-directory' and not Path('/proc/tty').is_dir():
        pytest.skip('This fixture uses the kernel tty directory as a mountpoint.')
    ram = Path(tempfile.mkdtemp(prefix='textgen-proc-masks-', dir='/dev/shm'))
    outside = tmp_path / 'outside'
    outside.write_text('unchanged')
    outside.chmod(0o666)
    directory = tmp_path / 'mask-directory'
    directory.mkdir()
    (directory / 'nested file').touch()
    try:
        if os.geteuid() == 0:
            # Construct the simulated container in a private mount namespace.
            # Ubuntu's bwrap profile intentionally denies capabilities to its
            # children, so nesting bwrap cannot construct this outer fixture.
            # The helper still drops to the real dedicated UID for the worker.
            command = ['unshare', '--mount', '--propagation', 'private', '--', sys.executable, '-B',
                       'tests/vault/probe_proc_masks.py', '--fixture', mask, str(ram), str(outside), str(directory)]
        else:
            # Local non-root hosts without that AppArmor restriction can build
            # the same container fixture with an outer bubblewrap namespace.
            command = [binary, '--unshare-user', '--ro-bind', '/', '/', '--bind', '/proc', '/proc',
                       '--bind', str(ram), str(ram)]
            if mask == 'null':
                command += ['--dev-bind', '/dev/null', '/proc/keys']
                paths = ['/proc/keys']
            elif mask == 'file':
                command += ['--bind', str(outside), '/proc/keys']
                paths = ['/proc/keys']
            else:
                command += ['--bind', str(directory), '/proc/tty',
                            '--bind', str(outside), '/proc/tty/nested file']
                paths = ['/proc/tty', '/proc/tty/nested file']
            command += ['--', sys.executable, '-B', 'tests/vault/probe_proc_masks.py', str(ram), *paths]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == 'proc masks protected; thread names writable'
        assert outside.read_text() == 'unchanged'
        assert not (directory / 'forbidden').exists()
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
    # The iframe must open the exact configured origin. The Colab helper
    # obtains another proxy URL internally; an absolute path pins our URL.
    assert 'path=bootstrap_url' in source
    assert 'TEXTGEN_COLAB_BOOTSTRAP_TOKEN' in source and 'vault-bootstrap' in source
    # Password entry belongs to the gateway, never an executed/saved cell.
    assert 'getpass' not in source and 'input(' not in source
