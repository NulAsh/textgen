import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.vault_linux import landlock_abi, verify_tmpfs
from modules.vault_store import VaultError


def test_disk_directory_is_refused(tmp_path):
    with pytest.raises(VaultError, match='tmpfs'):
        verify_tmpfs(tmp_path)


def test_ram_directory_symlink_is_refused(tmp_path):
    link = tmp_path / 'link'
    link.symlink_to('/dev/shm')
    with pytest.raises(VaultError, match='symlink'):
        verify_tmpfs(link)


def test_native_write_isolation_and_inheritance(tmp_path):
    try:
        landlock_abi()
    except VaultError as error:
        pytest.skip(str(error))
    ram = Path(tempfile.mkdtemp(prefix='textgen-landlock-test-', dir='/dev/shm'))
    outside = tmp_path / 'forbidden'
    try:
        code = '''
import pathlib,subprocess,sys
from modules.vault_linux import confine_writes
ram, outside = map(pathlib.Path, sys.argv[1:])
confine_writes(ram)
(ram/'allowed').write_text('allowed')
try:
 outside.write_text('forbidden')
 raise AssertionError('Unconfined write succeeded')
except PermissionError:
 pass
child=subprocess.run([sys.executable,'-B','-c', 'from pathlib import Path; Path('+repr(str(outside))+').write_text("forbidden")'],capture_output=True)
assert child.returncode != 0 and not outside.exists()
print('isolated')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code, str(ram), str(outside)], capture_output=True, text=True, timeout=20, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'isolated'
    finally:
        shutil.rmtree(ram)


def test_real_textgen_interface_and_persistence(tmp_path):
    pytest.importorskip('gradio')
    result = subprocess.run([sys.executable, '-B', 'tests/vault/probe_textgen.py', str(tmp_path / 'vault.sqlite3')],
                            capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary['components'] > 100
    assert summary['config'] == 200 and summary['upload'] == 200
    assert summary['all_content_restored'] is True


@pytest.mark.asyncio
async def test_native_gateway_and_worker(tmp_path, unused_tcp_port):
    """Exercise the production worker, Landlock and Unix socket together."""
    try:
        landlock_abi()
    except VaultError as error:
        pytest.skip(str(error))
    pytest.importorskip('gradio')
    import aiohttp
    from aiohttp.test_utils import TestClient, TestServer

    from modules.vault_gateway import Supervisor, create_app
    from modules.vault_store import Vault

    models = tmp_path / 'models'
    models.mkdir()
    options = SimpleNamespace(vault=str(tmp_path / 'native.sqlite3'), vault_ram_dir='/dev/shm',
                              vault_import=None, listen_port=unused_tcp_port, model_dir=str(models),
                              image_model_dir=str(models), lora_dir=str(models))
    supervisor = Supervisor(options, ['--portable'])
    client = TestClient(TestServer(create_app(supervisor), port=unused_tcp_port),
                        cookie_jar=aiohttp.CookieJar(unsafe=True))
    await client.start_server()
    client.session.headers.update({'Origin': f'http://127.0.0.1:{unused_tcp_port}', 'X-Vault-CSRF': supervisor.csrf})
    password = 'native synthetic test password'
    secret = 'SYNTHETIC-NATIVE-ENCRYPTED-UPLOAD-927018'
    try:
        response = await client.post('/vault/unlock', json={'password': password, 'confirmation': password})
        assert response.status == 200, await response.text()
        for _ in range(120):
            if supervisor.ready or supervisor.error:
                break
            await asyncio.sleep(0.5)
        diagnostics = (supervisor.workspace / 'data/private_logs/backend.log').read_text()
        assert supervisor.ready and not supervisor.error, diagnostics
        response = await client.get('/ui/config')
        assert response.status == 200, await response.text()
        assert len((await response.json())['components']) > 100
        upload = aiohttp.FormData()
        upload.add_field('files', secret.encode(), filename='synthetic.txt', content_type='text/plain')
        response = await client.post('/ui/upload', data=upload)
        assert response.status == 200, await response.text()
        uploaded = Path((await response.json())[0])
        relative = uploaded.relative_to(supervisor.workspace / 'data')
        assert uploaded.read_text() == secret
        response = await client.post('/vault/lock', json={})
        assert response.status == 200, await response.text()
        assert not supervisor.workspace.exists()
        vault = Vault.open(options.vault, password)
        try:
            assert secret.encode() not in vault.backup()
            # Test-only synthetic restore; production restores only to tmpfs.
            vault.restore(tmp_path / 'restored')
            assert (tmp_path / 'restored' / relative).read_text() == secret
        finally:
            vault.close()
    finally:
        await client.close()
