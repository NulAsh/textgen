"""Integration between the RAM-confined TextGen worker and the web gateway."""

import os
import runpy
import sys
import threading
from pathlib import Path

ACTIVE = False
WORKSPACE = None
_server = None
_thread = None


def enforce_options(shared):
    """Do not expose alternate servers, dynamic extensions or executable tools."""
    if not ACTIVE:
        return
    shared.args.multi_user = False
    shared.args.extensions = []
    shared.args.api = shared.args.public_api = shared.args.share = False
    shared.args.nowebui = shared.args.verbose = shared.args.trust_remote_code = False
    shared.settings["default_extensions"] = []
    shared.settings["selected_tools"] = []
    shared.settings["mcp_servers"] = ''
    shared.args.disk_cache_dir = str(WORKSPACE / "volatile" / "offload")


def build_app(blocks, allowed_paths):
    """Build the actual Gradio ASGI app, also usable for in-process tests."""
    import gradio as gr
    from fastapi import FastAPI

    app = FastAPI()
    return gr.mount_gradio_app(app, blocks, path="/", root_path="/ui", allowed_paths=allowed_paths,
                              show_error=False, favicon_path="css/icon.png", max_file_size='256mb')


def launch(blocks, allowed_paths):
    """No public Gradio socket: only the gateway can reach this private UDS."""
    import uvicorn

    global _server, _thread
    close()
    socket_path = WORKSPACE / "backend.sock"
    socket_path.unlink(missing_ok=True)
    app = build_app(blocks, allowed_paths)
    _server = uvicorn.Server(uvicorn.Config(app, uds=str(socket_path), log_level="warning", access_log=False))
    _thread = threading.Thread(target=_server.run, daemon=True)
    _thread.start()


def close():
    global _server, _thread
    if _server is not None:
        _server.should_exit = True
        if _thread is not None:
            _thread.join(timeout=10)
        _server = _thread = None


def flush_pending_save(timer, callback):
    """Flush only a pending debounce, never an old completed notebook save."""
    if timer is not None and timer.is_alive():
        timer.cancel()
        timer.join(timeout=2)
        if not timer.is_alive():
            callback()


def main():
    from modules.vault_linux import bind_to_parent, confine_writes, disable_core_dumps
    global ACTIVE, WORKSPACE
    # Only the supervisor launches this entry point. No password or key is
    # passed to the worker; it gets access to the transient plaintext tree.
    WORKSPACE = Path(sys.argv[1]).resolve()
    parent_pid = int(sys.argv[2])
    bind_to_parent(parent_pid)
    disable_core_dumps()
    os.umask(0o077)
    volatile = WORKSPACE / "volatile"
    volatile.mkdir(mode=0o700, exist_ok=True)
    for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR"):
        directory = volatile / name.lower()
        directory.mkdir(mode=0o700, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_TEMP_DIR"] = str(WORKSPACE / 'data/cache/gradio')
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.pop("TEXTGEN_ELECTRON", None)
    sys.dont_write_bytecode = True
    # Must run on the initial, single worker thread BEFORE importing Gradio,
    # model runtimes, or any code that creates threads or subprocesses.
    confine_writes(WORKSPACE)
    ACTIVE = True
    project = Path(__file__).resolve().parent.parent
    sys.argv = [str(project / "server.py")] + sys.argv[3:]
    runpy.run_path(str(project / "server.py"), run_name="__main__")


if __name__ == "__main__":
    # -m executes this file as __main__; ensure all imports share ACTIVE.
    sys.modules["modules.vault_runtime"] = sys.modules[__name__]
    main()
