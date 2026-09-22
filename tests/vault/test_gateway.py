import asyncio
from types import SimpleNamespace

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from modules import vault_gateway as gateway
from modules.vault_store import VaultError

PASSWORD = 'long password for tests only'


@pytest_asyncio.fixture
async def app(tmp_path, monkeypatch, unused_tcp_port):
    # This container lacks Landlock. These are gateway/storage tests only;
    # never start an unconfined TextGen worker. The production guard has no
    # environment variable or command-line switch to disable it.
    monkeypatch.setattr(gateway, 'landlock_abi', lambda: 3)
    options = SimpleNamespace(vault=str(tmp_path / 'vault.sqlite3'), vault_ram_dir='/dev/shm',
                              vault_import=None, listen_port=unused_tcp_port, model_dir=str(tmp_path),
                              image_model_dir=str(tmp_path), lora_dir=str(tmp_path))
    supervisor = gateway.Supervisor(options, [])
    async def no_worker():
        pass
    monkeypatch.setattr(supervisor, 'start_worker', no_worker)
    client = TestClient(TestServer(gateway.create_app(supervisor), port=unused_tcp_port), cookie_jar=aiohttp.CookieJar(unsafe=True))
    await client.start_server()
    client.session.headers.update({'Origin': f'http://127.0.0.1:{unused_tcp_port}', 'X-Vault-CSRF': supervisor.csrf})
    try:
        yield supervisor, client
    finally:
        await client.close()


async def unlock(client):
    response = await client.post('/vault/unlock', json={'password': PASSWORD, 'confirmation': PASSWORD})
    assert response.status == 200, await response.text()


@pytest.mark.asyncio
async def test_locked_endpoints_reject_access(app):
    supervisor, client = app
    for path in ('/vault/status', '/vault/diagnostics', '/ui/config', '/ui/file=/etc/passwd'):
        response = await client.get(path)
        assert response.status == 401
    response = await client.post('/vault/backup', json={})
    assert response.status == 401
    response = await client.get('/')
    assert response.status == 200
    assert response.headers['Cache-Control'].startswith('no-store')
    assert supervisor.vault is None


@pytest.mark.asyncio
async def test_unlock_save_lock_reopen(app):
    supervisor, client = app
    await unlock(client)
    secret = 'CHAT-SYSTEM-OUTPUT-SENTINEL-927018'
    (supervisor.workspace / 'data/settings.yaml').write_text(secret)
    response = await client.post('/vault/backup', json={})
    assert response.status == 200
    assert secret.encode() not in await response.read()
    token = supervisor.session
    response = await client.post('/vault/lock', json={})
    assert response.status == 200
    assert supervisor.session is None and supervisor.vault is None
    assert not supervisor.workspace.exists()
    response = await client.get('/ui/config', cookies={gateway.COOKIE: token})
    assert response.status == 401
    supervisor.last_attempt = 0
    await unlock(client)
    assert (supervisor.workspace / 'data/settings.yaml').read_text() == secret


@pytest.mark.asyncio
async def test_host_origin_and_csrf(app):
    _, client = app
    body = {'password': PASSWORD, 'confirmation': PASSWORD}
    response = await client.post('/vault/unlock', json=body, headers={'Host': 'evil.example'})
    assert response.status == 403
    response = await client.post('/vault/unlock', json=body, headers={'Origin': 'https://evil.example'})
    assert response.status == 403
    response = await client.post('/vault/unlock', json=body, headers={'X-Vault-CSRF': 'wrong'})
    assert response.status == 403


@pytest.mark.asyncio
async def test_password_change_requires_old_password(app):
    supervisor, client = app
    await unlock(client)
    response = await client.post('/vault/password', json={'oldPassword': 'incorrect', 'newPassword': 'new long password', 'confirmation': 'new long password'})
    assert response.status == 400
    assert supervisor.vault.check_password(PASSWORD)
    response = await client.post('/vault/password', json={'oldPassword': PASSWORD, 'newPassword': 'new long password', 'confirmation': 'new long password'})
    assert response.status == 200
    assert supervisor.vault.check_password('new long password')


@pytest.mark.asyncio
async def test_failed_lock_retains_ram_and_does_not_claim_success(app, monkeypatch):
    supervisor, client = app
    await unlock(client)
    original_sync = supervisor.vault.sync
    def fail(_):
        raise OSError('disk full')
    monkeypatch.setattr(supervisor.vault, 'sync', fail)
    response = await client.post('/vault/lock', json={})
    assert response.status == 400
    assert supervisor.error and supervisor.workspace.exists() and supervisor.vault is not None
    assert supervisor.session is None
    monkeypatch.setattr(supervisor.vault, 'sync', original_sync)
    await unlock(client)
    response = await client.post('/vault/lock', json={})
    assert response.status == 200


@pytest.mark.asyncio
async def test_streaming_proxy_and_upload_stay_authenticated(app, monkeypatch, unused_tcp_port_factory):
    supervisor, client = app
    await unlock(client)
    received = []
    async def stream(request):
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        await response.write(b'data: first\n\n')
        await asyncio.sleep(0.01)
        await response.write(b'data: second\n\n')
        return response
    async def upload(request):
        received.append(await request.read())
        assert gateway.COOKIE not in request.headers.get('Cookie', '')
        return web.json_response({'ok': True})
    backend = web.Application()
    backend.router.add_get('/queue/data', stream)
    backend.router.add_post('/upload', upload)
    runner = web.AppRunner(backend)
    await runner.setup()
    # Transport-only stand-in: AF_UNIX is prohibited in the CI container.
    # Exercise the real streaming proxy over a loopback socket using only
    # synthetic data. No production switch bypasses the private UDS.
    backend_port = unused_tcp_port_factory()
    site = web.TCPSite(runner, '127.0.0.1', backend_port)
    await site.start()
    class TestConnector(aiohttp.TCPConnector):
        async def _resolve_host(self, host, port, traces=None):
            return await super()._resolve_host('127.0.0.1', backend_port, traces=traces)
    monkeypatch.setattr(gateway.aiohttp, 'UnixConnector', lambda **kwargs: TestConnector())
    supervisor.ready = True
    try:
        response = await client.get('/ui/queue/data')
        assert response.status == 200
        assert response.headers['Cache-Control'].startswith('no-store')
        assert await response.read() == b'data: first\n\ndata: second\n\n'
        response = await client.post('/ui/upload', data=b'private upload contents')
        assert response.status == 200
        assert received == [b'private upload contents']
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_wrong_password_cannot_steal_unlocked_session(app):
    supervisor, client = app
    await unlock(client)
    token = supervisor.session
    response = await client.post('/vault/unlock', json={'password': 'wrong'})
    assert response.status == 400
    assert supervisor.session == token


@pytest.mark.asyncio
async def test_empty_password_creates_no_database(app):
    supervisor, client = app
    response = await client.post('/vault/unlock', json={'password': '', 'confirmation': ''})
    assert response.status == 400
    assert not supervisor.path.exists()


@pytest.mark.asyncio
async def test_background_save_failure_stops_worker_and_retains_data(app, monkeypatch):
    supervisor, client = app
    await unlock(client)
    original = supervisor.vault.sync
    stopped = asyncio.Event()
    async def stop_worker():
        supervisor.ready = False
        stopped.set()
    def fail(_):
        raise OSError('simulated storage failure')
    monkeypatch.setattr(supervisor, 'stop_worker', stop_worker)
    monkeypatch.setattr(supervisor.vault, 'sync', fail)
    await asyncio.wait_for(stopped.wait(), timeout=3)
    assert supervisor.error and supervisor.workspace.exists() and supervisor.vault is not None
    monkeypatch.setattr(supervisor.vault, 'sync', original)


def test_production_refuses_missing_landlock(tmp_path, monkeypatch):
    options = SimpleNamespace(vault=str(tmp_path / 'vault.sqlite3'), vault_ram_dir='/dev/shm')
    supervisor = gateway.Supervisor(options, [])
    def unsupported():
        raise VaultError('Landlock unavailable')
    monkeypatch.setattr(gateway, 'landlock_abi', unsupported)
    with pytest.raises(VaultError, match='Landlock'):
        supervisor.acquire()
    assert supervisor.process is None and not supervisor.path.exists()


@pytest.mark.parametrize('flag', ['--api', '--extensions', '--user-data-dir=/tmp', '--set=/tmp/file', '--extra-flags=--log-file', '--share'])
def test_unsafe_launch_flags_are_rejected(flag):
    with pytest.raises(SystemExit):
        gateway.parse_args([flag])
