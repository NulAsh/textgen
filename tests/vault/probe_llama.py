"""Real llama.cpp loader/inference probe; only synthetic input is used.

The parent extracts a fresh wheel outside the RAM workspace and removes both
test directories afterward. Native mode applies production Landlock before
importing the wheel; loader-only mode passes the executable path directly to
exercise the same missing aliases on kernels without Landlock.
"""

import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
installed, model, ram = map(Path, sys.argv[1:4])
confined = sys.argv[4] == '1'
colab_parent = int(sys.argv[5]) if sys.argv[4] == 'colab' else None
sys.path.insert(0, str(installed))
binary_dir = installed / 'llama_cpp_binaries/bin'
manifest = binary_dir / '_symlinks.json'
original = manifest.read_bytes()
data = ram / 'data'
data.mkdir(exist_ok=True)
volatile = ram / 'volatile'
volatile.mkdir(exist_ok=True)
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['TMPDIR'] = str(volatile)
os.environ['XDG_CACHE_HOME'] = str(volatile / 'cache')
sys.argv = ['server.py', '--portable', '--user-data-dir', str(data), '--gpu-layers', '0',
            '--ctx-size', '256', '--batch-size', '32', '--threads', '2']

from modules import vault_runtime

vault_runtime.ACTIVE = True
vault_runtime.WORKSPACE = ram
if confined:
    from modules.vault_linux import confine_writes
    confine_writes(ram)
elif colab_parent:
    from modules.vault_colab import verify_sandbox
    verify_sandbox(ram, colab_parent)

import requests

from modules import shared
from modules.llama_cpp_server import LlamaServer

server_path = None if confined or colab_parent else str(binary_dir / 'llama-server')
with LlamaServer(str(model), server_path=server_path) as backend:
    url = f'http://127.0.0.1:{backend.port}/completion'
    unauthorized = requests.post(url, json={'prompt': 'Once upon a time', 'n_predict': 8}, timeout=10)
    assert unauthorized.status_code == 401
    state = dict(shared.settings)
    state.update(max_new_tokens=8, auto_max_new_tokens=False, truncation_length=256, seed=1)
    chunks = list(backend.generate_with_streaming('Once upon a time', state))
    assert chunks and chunks[-1]
    tokens = backend.last_completion_token_count
    assert tokens > 0
    assert manifest.read_bytes() == original
    assert all(not (binary_dir / name).exists() for name in json.loads(original))
    print(json.dumps({'generated_tokens': tokens, 'unauthenticated_status': unauthorized.status_code,
                      'installation_unchanged': True}))
