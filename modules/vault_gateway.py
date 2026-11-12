"""Single-user browser unlock, supervisor and authenticated streaming proxy."""

import argparse
import asyncio
import base64
import contextlib
import fcntl
import json
import os
import secrets
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from modules.vault_linux import (
    disable_core_dumps,
    landlock_abi,
    make_runtime,
    runtime_path,
)
from modules.vault_store import (
    SnapshotBusy,
    Vault,
    VaultError,
    read_tree,
    safe_name,
    unpack_record,
)

PROJECT = Path(__file__).resolve().parent.parent
COOKIE = "textgen_vault_session"
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}
FORBIDDEN_FLAGS = {
    "--extensions", "--api", "--public-api", "--nowebui", "--share", "--multi-user",
    "--verbose", "--trust-remote-code", "--settings", "--listen", "--listen-host",
    "--gradio-auth", "--gradio-auth-path", "--subpath", "--extra-flags", "--disk-cache-dir",
    "--chat-template-file", "--model-menu",
}


class Supervisor:
    def __init__(self, options, backend_args):
        self.options = options
        self.backend_args = backend_args
        self.path = Path(options.vault).resolve()
        self.workspace = runtime_path(self.path, options.vault_ram_dir)
        self.vault = None
        self.process = None
        self.session = None
        self.csrf = secrets.token_urlsafe(32)
        self.error = None
        self.ready = False
        self.last_saved = None
        self.last_attempt = 0
        self.failures = 0
        self.guard = asyncio.Lock()
        self.proxy_tasks = set()
        self.lock_fd = None
        self.isolation = None
        self.colab_origin = getattr(options, 'vault_colab_origin', None)
        self.backup_dir = getattr(options, 'vault_backup_dir', None)
        self.last_backup = None
        self.backup_pending = False

    def acquire(self):
        if getattr(self.options, 'vault_isolation', 'landlock') == 'colab':
            from modules.vault_colab import ColabIsolation
            self.isolation = ColabIsolation()
            try:
                self.isolation.preflight(self.options.vault_ram_dir)
            except BaseException:
                self.isolation.close()
                raise
        else:
            landlock_abi()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock_fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lock_fd)
            self.lock_fd = None
            raise VaultError("This vault is already open in another TextGen process.") from None
        if self.path.is_symlink():
            raise VaultError("The vault database must not be a symlink.")

    def _seed(self, import_existing):
        data = self.workspace / "data"
        data.mkdir(mode=0o700, exist_ok=True)
        assets = json.loads((PROJECT / "modules/vault_defaults.json").read_text())
        for name, encoded in assets.items():
            destination = data / safe_name(name)
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            destination.write_bytes(base64.b64decode(encoded, validate=True))
        for name in ("logs", "cache", "loras", "tools", "extensions", "training/datasets", "training/formats", "image_outputs"):
            (data / name).mkdir(mode=0o700, parents=True, exist_ok=True)
        if import_existing:
            if not self.options.vault_import:
                raise VaultError("No import directory was provided at startup.")
            source = Path(self.options.vault_import).resolve()
            if source == data or source in self.workspace.parents:
                raise VaultError("Invalid import directory.")
            # Model weights stay outside the vault; templates and model-specific
            # overrides belong inside. Never execute imported command flags/code.
            allowed = {"logs", "characters", "users", "presets", "instruction-templates", "grammars", "cache", "image_outputs", "training"}
            for item in source.iterdir():
                if item.name not in allowed and item.name != "settings.yaml":
                    continue
                if item.is_symlink():
                    raise VaultError("Symlinks in imported user data are not supported.")
                if item.is_dir():
                    for name, record in read_tree(item).items():
                        _, kind, content = unpack_record(record)
                        destination = data / item.name / safe_name(name)
                        if kind == b"d":
                            destination.mkdir(mode=0o700, parents=True, exist_ok=True)
                        else:
                            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                            destination.write_bytes(content)
                else:
                    if item.stat().st_size > 16 * 1024 * 1024:
                        raise VaultError("Imported settings are too large.")
                    (data / item.name).write_bytes(item.read_bytes())
            config = source / "models" / "config-user.yaml"
            if config.is_file() and not config.is_symlink():
                (data / "model-settings.yaml").write_bytes(config.read_bytes())

    async def unlock(self, password, confirmation=None, import_existing=False):
        async with self.guard:
            if not isinstance(password, str) or not password or len(password.encode('utf-8')) > 4096:
                raise VaultError("Enter a password of at most 4096 UTF-8 bytes.")
            wait = min(30, self.failures) - (time.monotonic() - self.last_attempt)
            if wait > 0:
                raise VaultError("Please wait before trying another password.")
            self.last_attempt = time.monotonic()
            creating = not self.path.exists()
            if creating and password != confirmation:
                raise VaultError("The two passwords do not match.")
            try:
                if self.vault is not None:
                    if not await asyncio.to_thread(self.vault.check_password, password):
                        raise VaultError("Incorrect password.")
                else:
                    if creating:
                        # Validate import and create the RAM tree before creating
                        # an empty persistent vault, so migration can be retried.
                        make_runtime(self.workspace, allow_worker_group=self.isolation is not None)
                        await asyncio.to_thread(self._seed, import_existing)
                        self.vault = await asyncio.to_thread(Vault.create, self.path, password)
                        await asyncio.to_thread(self.vault.sync, self.workspace / "data")
                    else:
                        opened = await asyncio.to_thread(Vault.open, self.path, password)
                        created_workspace = False
                        try:
                            make_runtime(self.workspace, allow_worker_group=self.isolation is not None)
                            created_workspace = True
                            await asyncio.to_thread(opened.restore, self.workspace / "data")
                        except BaseException:
                            opened.close()
                            if created_workspace:
                                shutil.rmtree(self.workspace)
                            raise
                        self.vault = opened
                    self.last_saved = time.time()
                    self.backup_pending = True
                    await self.save_backup(force=True)
                    await self.start_worker()
                self.failures = 0
                self.session = secrets.token_urlsafe(32)
                return self.session
            except Exception:
                self.failures += 1
                if self.vault is not None and self.process is None:
                    self.error = "Startup or save failed. Re-enter your password, then use Save and lock to preserve the data and retry."
                raise

    async def start_worker(self):
        log_dir = self.workspace / "data/private_logs"
        log_dir.mkdir(mode=0o700, exist_ok=True)
        backend = 'colab' if self.isolation is not None else 'landlock'
        command = [sys.executable, "-B", "-m", "modules.vault_runtime", str(self.workspace), str(os.getpid()), backend]
        command += self.backend_args + ["--user-data-dir", str(self.workspace / "data"),
                                       "--model-dir", str(Path(self.options.model_dir).resolve()),
                                       "--image-model-dir", str(Path(self.options.image_model_dir).resolve()),
                                       "--lora-dir", str(Path(self.options.lora_dir).resolve()), "--no-electron"]
        subprocess_options = {}
        if self.isolation is not None:
            self.isolation.claim(self.workspace)
            command = self.isolation.command(self.workspace, command)
            subprocess_options = self.isolation.subprocess_options()
        with (log_dir / "backend.log").open("ab", buffering=0) as log:
            self.process = await asyncio.create_subprocess_exec(*command, cwd=PROJECT, stdin=asyncio.subprocess.DEVNULL,
                                                                stdout=log, stderr=log, start_new_session=True, **subprocess_options)
        self.error = None
        self.ready = False

    async def stop_worker(self):
        self.ready = False
        process, self.process = self.process, None
        if process is None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGINT)
        try:
            await asyncio.wait_for(process.wait(), 12)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        finally:
            # Also stop any model subprocess that outlived the Python worker.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)

    async def lock(self):
        async with self.guard:
            self.session = None
            for task in tuple(self.proxy_tasks):
                task.cancel()
            await self.stop_worker()
            if self.vault is not None:
                try:
                    await asyncio.to_thread(self.vault.sync, self.workspace / "data")
                    await self.save_backup(force=True)
                except Exception:  # noqa: BLE001 - retain plaintext RAM on any persistence failure
                    self.error = "Save failed. The RAM workspace has been retained. Free disk space and retry Save and lock."
                    raise VaultError(self.error) from None
                self.vault.close()
                self.vault = None
                shutil.rmtree(self.workspace)
            self.error = None
            self.last_saved = None

    async def monitor(self):
        while True:
            await asyncio.sleep(1)
            async with self.guard:
                if self.vault is None or self.error:
                    continue
                try:
                    changed = await asyncio.to_thread(self.vault.sync, self.workspace / "data")
                    if changed:
                        self.last_saved = time.time()
                        self.backup_pending = True
                    await self.save_backup()
                    if self.process is not None and self.process.returncode is not None:
                        self.error = "TextGen stopped. Save and lock, then unlock to restart. Diagnostics are available below."
                        self.ready = False
                    elif not self.ready and (self.workspace / "volatile/backend.sock").exists():
                        connector = aiohttp.UnixConnector(path=str(self.workspace / "volatile/backend.sock"))
                        async with aiohttp.ClientSession(connector=connector) as client, client.get("http://localhost/config", timeout=aiohttp.ClientTimeout(total=2)) as response:
                            self.ready = response.status == 200
                except (TimeoutError, SnapshotBusy, aiohttp.ClientError):
                    # A changing file or a starting socket is retried next tick.
                    continue
                except Exception:  # noqa: BLE001 - fail closed without logging plaintext exception details
                    self.error = "Encrypted save failed. TextGen has stopped and the RAM workspace is retained. Retry Save and lock after correcting the storage problem."
                    await self.stop_worker()

    async def save_backup(self, force=False):
        if self.backup_dir and self.vault is not None and (force or (
                self.backup_pending and time.time() - (self.last_backup or 0) >= 60)):
            from modules.vault_backup import save_encrypted_backup
            content = await asyncio.to_thread(self.vault.backup)
            await asyncio.to_thread(save_encrypted_backup, self.backup_dir, content)
            self.last_backup = time.time()
            self.backup_pending = False

    def authenticated(self, request):
        token = request.cookies.get(COOKIE, "")
        return self.session is not None and secrets.compare_digest(token, self.session)

    def record_proxy_error(self, error):
        """Keep exception details only in the unlocked, encrypted working data."""
        if self.vault is None:
            return
        try:
            path = self.workspace / 'data/private_logs/gateway.log'
            path.parent.mkdir(mode=0o700, exist_ok=True)
            with path.open('a', encoding='utf-8') as log:
                log.write(time.strftime('\n%Y-%m-%d %H:%M:%S UTC\n', time.gmtime()))
                traceback.print_exception(type(error), error, error.__traceback__, file=log)
        except OSError:
            # Disk/RAM failures must not send exception details to the console.
            pass


def create_app(supervisor):
    @web.middleware
    async def security(request, handler):
        expected = {f"127.0.0.1:{supervisor.options.listen_port}", f"localhost:{supervisor.options.listen_port}"}
        if supervisor.colab_origin:
            expected.add(urlsplit(supervisor.colab_origin).netloc)
        if request.host not in expected:
            raise web.HTTPForbidden(text="Invalid Host.")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = supervisor.colab_origin or f"http://{request.host}"
            if request.headers.get("Origin") != origin:
                raise web.HTTPForbidden(text="Invalid Origin.")
            if request.path.startswith("/vault/") and not secrets.compare_digest(request.headers.get("X-Vault-CSRF", ""), supervisor.csrf):
                raise web.HTTPForbidden(text="Invalid request token.")
        try:
            response = await handler(request)
        except web.HTTPException as error:
            response = web.Response(status=error.status, text=error.text, headers=error.headers)
        except VaultError as error:
            response = web.json_response({"error": str(error)}, status=400)
        except Exception as error:  # noqa: BLE001 - HTTP errors must not expose exception contents
            # Exception strings may include filenames/input. Detailed backend
            # diagnostics stay in the encrypted vault, never in HTTP errors.
            if request.path.startswith('/ui/'):
                supervisor.record_proxy_error(error)
            # Password/control requests are deliberately excluded from detailed
            # logging. The console gets no request bodies, URLs or exceptions.
            print('Vault web request failed. Open Diagnostics in the unlocked vault.', file=sys.stderr)
            response = web.json_response({"error": "The operation failed. Check Diagnostics in the unlocked vault."}, status=500)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        add_frame_policy(response)
        return response

    app = web.Application(middlewares=[security], client_max_size=256 * 1024 * 1024)

    def add_frame_policy(response):
        directives = [part.strip() for part in response.headers.get('Content-Security-Policy', '').split(';')
                      if part.strip() and not part.strip().startswith('frame-ancestors')]
        if supervisor.colab_origin:
            response.headers.pop('X-Frame-Options', None)
            # The full UI is nested inside Colab's notebook output frame.
            directives.append("frame-ancestors 'self' https://colab.research.google.com https://*.googleusercontent.com https://*.colab.dev")
        else:
            response.headers['X-Frame-Options'] = 'SAMEORIGIN'
            directives.append("frame-ancestors 'self'")
        response.headers['Content-Security-Policy'] = '; '.join(directives)

    def set_session_cookie(response, token):
        if supervisor.colab_origin:
            # CHIPS keeps the cookie in this notebook's top-level partition.
            # Use the header directly for Python versions whose Morsel does
            # not yet recognize Partitioned. Tokens are generated URL-safe.
            value = f'{COOKIE}={token or ""}; Path=/; Secure; HttpOnly; SameSite=None; Partitioned'
            if token is None:
                value += '; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT'
            response.headers.add('Set-Cookie', value)
        elif token is None:
            response.del_cookie(COOKIE, path='/')
        else:
            response.set_cookie(COOKIE, token, httponly=True, samesite='Strict', path='/')

    def require_auth(request):
        if not supervisor.authenticated(request):
            raise web.HTTPUnauthorized(text="Unlock the vault first.")

    async def index(request):
        html = (PROJECT / "modules/vault_web.html").read_text(encoding="utf-8")
        nonce = secrets.token_urlsafe(24)
        config = {"csrf": supervisor.csrf, "creating": not supervisor.path.exists(),
                  "authenticated": supervisor.authenticated(request), "canImport": bool(supervisor.options.vault_import)}
        html = html.replace("__CONFIG__", json.dumps(config)).replace("__NONCE__", nonce)
        response = web.Response(text=html, content_type="text/html")
        response.headers["Content-Security-Policy"] = f"default-src 'self'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; frame-src 'self'; object-src 'none'; base-uri 'none'"
        return response

    async def unlock(request):
        if request.content_length is None or request.content_length > 20000:
            raise web.HTTPRequestEntityTooLarge(max_size=20000, actual_size=request.content_length or 0)
        data = await request.json()
        token = await supervisor.unlock(data.get("password"), data.get("confirmation"), data.get("importExisting") is True)
        response = web.json_response({"ok": True})
        set_session_cookie(response, token)
        return response

    async def lock(request):
        require_auth(request)
        await supervisor.lock()
        response = web.json_response({"ok": True})
        set_session_cookie(response, None)
        response.headers["Clear-Site-Data"] = '"cache", "storage"'
        return response

    async def status(request):
        require_auth(request)
        return web.json_response({"ready": supervisor.ready, "error": supervisor.error, "lastSaved": supervisor.last_saved,
                                  "lastBackup": supervisor.last_backup, "backupEnabled": bool(supervisor.backup_dir)})

    async def backup(request):
        require_auth(request)
        async with supervisor.guard:
            if supervisor.vault is None:
                raise web.HTTPConflict(text="Vault is locked.")
            await asyncio.to_thread(supervisor.vault.sync, supervisor.workspace / "data")
            await supervisor.save_backup(force=True)
            data = await asyncio.to_thread(supervisor.vault.backup)
        return web.Response(body=data, content_type="application/octet-stream", headers={"Content-Disposition": 'attachment; filename="textgen-vault-backup.sqlite3"'})

    async def change_password(request):
        require_auth(request)
        if request.content_length is None or request.content_length > 20000:
            raise web.HTTPBadRequest(text="Invalid request size.")
        data = await request.json()
        if data.get("newPassword") != data.get("confirmation"):
            raise VaultError("The two new passwords do not match.")
        async with supervisor.guard:
            await asyncio.to_thread(supervisor.vault.change_password, data.get("oldPassword"), data.get("newPassword"))
            await supervisor.save_backup(force=True)
        return web.json_response({"ok": True})

    async def diagnostics(request):
        require_auth(request)
        sections = []
        for name in ('backend.log', 'gateway.log'):
            path = supervisor.workspace / 'data/private_logs' / name
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 65536))
                    sections.append(f'--- {name} ---\n' + stream.read().decode('utf-8', errors='replace'))
        return web.Response(text='\n\n'.join(sections) or 'No diagnostic output yet.', content_type='text/plain')

    async def proxy(request):
        require_auth(request)
        if not supervisor.ready or supervisor.error:
            raise web.HTTPServiceUnavailable(text="TextGen is starting or has stopped. Return to the vault controls.")
        # Gradio 4 uses HTTP/SSE for its queue. Reject alternate upgraded
        # protocols instead of opening an unguarded WebSocket route.
        if request.headers.get("Upgrade"):
            raise web.HTTPBadRequest(text="WebSocket upgrades are not supported.")
        task = asyncio.current_task()
        supervisor.proxy_tasks.add(task)
        try:
            connector = aiohttp.UnixConnector(path=str(supervisor.workspace / "volatile/backend.sock"))
            async with aiohttp.ClientSession(connector=connector, auto_decompress=False,
                                             timeout=aiohttp.ClientTimeout(total=None, sock_connect=10)) as client:
                headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS | {"host", "cookie", "authorization", "forwarded"} and not k.lower().startswith("x-forwarded-")}
                headers["Host"] = request.host
                if supervisor.colab_origin:
                    # Derive these from explicit configuration, never untrusted
                    # incoming forwarded headers. Gradio needs HTTPS asset URLs.
                    headers['Host'] = urlsplit(supervisor.colab_origin).netloc
                    headers['X-Forwarded-Host'] = headers['Host']
                    headers['X-Forwarded-Proto'] = 'https'
                # Preserve encoded file URLs and query strings exactly.
                from yarl import URL
                target = URL("http://localhost" + request.raw_path[len("/ui"):], encoded=True)
                async with client.request(request.method, target, headers=headers, data=request.content.iter_any(), allow_redirects=False) as upstream:
                    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS | {"set-cookie"}}
                    out_headers.update({"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache", "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff", "X-Frame-Options": "SAMEORIGIN"})
                    response = web.StreamResponse(status=upstream.status, headers=out_headers)
                    add_frame_policy(response)
                    try:
                        await response.prepare(request)
                        async for chunk in upstream.content.iter_any():
                            if not supervisor.authenticated(request):
                                break
                            await response.write(chunk)
                        await response.write_eof()
                    except (ConnectionResetError, BrokenPipeError):
                        # Reloading/closing a page ends its SSE stream. Do not
                        # log a vault failure or attempt a new HTTP 500 after
                        # headers have been sent to a disconnected browser.
                        # A live downstream still needs upstream diagnostics.
                        if request.transport is not None and not request.transport.is_closing():
                            raise
                        response.force_close()
                    return response
        finally:
            supervisor.proxy_tasks.discard(task)

    app.router.add_get("/", index)
    app.router.add_post("/vault/unlock", unlock)
    app.router.add_post("/vault/lock", lock)
    app.router.add_get("/vault/status", status)
    app.router.add_post("/vault/backup", backup)
    app.router.add_post("/vault/password", change_password)
    app.router.add_get("/vault/diagnostics", diagnostics)
    app.router.add_route("*", "/ui/{tail:.*}", proxy)

    async def lifecycle(app):
        supervisor.acquire()
        task = asyncio.create_task(supervisor.monitor())
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        try:
            await supervisor.lock()
        except VaultError:
            print(f"Encrypted save failed. RAM workspace retained at {supervisor.workspace}", file=sys.stderr)
        finally:
            if supervisor.lock_fd is not None:
                os.close(supervisor.lock_fd)
            if supervisor.isolation is not None:
                supervisor.isolation.close()

    app.cleanup_ctx.append(lifecycle)
    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TextGen encrypted vault (Linux, browser interface)", allow_abbrev=False)
    parser.add_argument("--vault", default=str(PROJECT / "user_data/vault.sqlite3"), help="Encrypted vault database")
    parser.add_argument("--vault-import", help="Optional existing user_data directory to import when creating a vault")
    parser.add_argument("--vault-ram-dir", default="/dev/shm", help="Existing tmpfs directory for transient data")
    parser.add_argument("--listen-port", type=int, default=7860)
    parser.add_argument("--model-dir", default=str(PROJECT / "user_data/models"))
    parser.add_argument("--image-model-dir", default=str(PROJECT / "user_data/image_models"))
    parser.add_argument("--lora-dir", default=str(PROJECT / "user_data/loras"))
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument('--vault-isolation', choices=('landlock', 'colab'), default='landlock')
    parser.add_argument('--vault-colab-origin', help='Exact HTTPS origin returned by the private Colab port proxy')
    parser.add_argument('--vault-backup-dir', help='Optional folder for periodic encrypted snapshots, such as mounted Google Drive')
    options, backend = parser.parse_known_args(argv)
    if not 1 <= options.listen_port <= 65535:
        parser.error("--listen-port must be between 1 and 65535")
    if options.vault_colab_origin:
        try:
            parsed = urlsplit(options.vault_colab_origin)
            valid = (parsed.scheme == 'https' and parsed.hostname and parsed.hostname.endswith(('-colab.googleusercontent.com', '.colab.dev'))
                     and not parsed.username and not parsed.password and parsed.port in (None, 443)
                     and parsed.path in ('', '/') and not parsed.query and not parsed.fragment)
        except ValueError:
            valid = False
        if not valid:
            parser.error('--vault-colab-origin must be an exact HTTPS Colab proxy origin')
        options.vault_colab_origin = 'https://' + parsed.hostname
    if options.vault_backup_dir:
        backup_dir = Path(options.vault_backup_dir).resolve()
        if Path(options.vault).resolve() in (backup_dir / 'vault-latest.sqlite3', backup_dir / 'vault-previous.sqlite3'):
            parser.error('Use a local active vault; the backup folder must contain separate snapshots')
    # Reject abbreviations of restricted upstream options as well.
    for argument in backend:
        flag = argument.split("=", 1)[0]
        if flag.startswith("--") and (flag == "--user-data-dir" or any(x.startswith(flag) for x in FORBIDDEN_FLAGS)):
            parser.error(f"{flag} is unavailable in the encrypted browser launcher")
    return options, backend


def main():
    os.umask(0o077)
    disable_core_dumps()
    options, backend = parse_args()
    try:
        supervisor = Supervisor(options, backend)
        if options.vault_isolation == 'landlock':
            landlock_abi()
    except VaultError as error:
        raise SystemExit(str(error)) from None
    if options.auto_launch:
        import threading
        import webbrowser
        threading.Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:{options.listen_port}")).start()
    print(f"TextGen vault: http://127.0.0.1:{options.listen_port}")
    web.run_app(create_app(supervisor), host="127.0.0.1", port=options.listen_port, access_log=None, print=None)


if __name__ == "__main__":
    main()
