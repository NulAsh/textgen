"""In-process integration probe with synthetic data; does not launch a worker.

Separate process because upstream TextGen parses argv and owns global UI state.
This exercises real Gradio routes, uploads, and TextGen persistence while the
kernel-dependent worker tests remain separate.
"""

import base64
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
database = Path(sys.argv[1])
root = Path(tempfile.mkdtemp(prefix='textgen-probe-', dir='/dev/shm'))
try:
    data = root / 'data'
    data.mkdir()
    for name in ('models', 'image_models', 'loras'):
        (root / name).mkdir()
    for name, encoded in json.loads((PROJECT / 'modules/vault_defaults.json').read_text()).items():
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(encoded))
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    os.environ['GRADIO_TEMP_DIR'] = str(data / 'cache/gradio')
    os.environ['TMPDIR'] = str(root)
    sys.argv = ['server.py', '--portable', '--user-data-dir', str(data), '--model-dir', str(root / 'models')]
    from modules import vault_runtime
    vault_runtime.ACTIVE = True
    vault_runtime.WORKSPACE = root
    captured = {}
    vault_runtime.launch = lambda blocks, paths: captured.update(blocks=blocks, paths=paths)
    import server
    from modules import chat, models_settings, ui, ui_image_generation, ui_notebook
    from modules.vault_store import Vault, read_tree

    secret = 'SYNTHETIC-PRIVATE-CONTENT-TEST-927018'
    server.shared.settings['custom_system_message'] = secret
    server.create_interface()
    from fastapi.testclient import TestClient
    app = vault_runtime.build_app(captured['blocks'], captured['paths'])
    with TestClient(app) as client:
        config = client.get('/config')
        assert secret in config.text
        upload = client.post('/upload', files={'files': ('synthetic.txt', secret.encode(), 'text/plain')})
        upload_path = Path(upload.json()[0])
        assert upload_path.is_relative_to(data) and upload_path.read_text() == secret
        history = {'internal': [[secret, secret + '-GENERATED']], 'visible': [[secret, secret + '-GENERATED']]}
        chat.add_message_attachment(history, 0, str(upload_path))
        chat.save_history(history, 'synthetic-chat', 'Assistant', 'instruct')
        ui_notebook.safe_autosave_prompt(secret + '-NOTEBOOK', 'synthetic-notebook')
        list(models_settings.save_instruction_template('synthetic.gguf', secret + '-TEMPLATE'))
        state = dict(server.shared.settings)
        state.update({'prompt_menu-default': 'synthetic-notebook', 'prompt_menu-notebook': 'synthetic-notebook',
                      'character_menu': 'Assistant', 'user_menu': 'Default'})
        ui._last_interface_state = state
        ui._last_preset = 'Top-P'
        ui._last_extensions = []
        ui._last_show_controls = True
        ui._last_theme_state = 'dark'
        ui._perform_debounced_save()
        from PIL import Image
        image_paths = ui_image_generation.save_generated_images([Image.new('RGB', (2, 2), 'red')], {'image_prompt': secret + '-IMAGE'}, 123)
        assert secret in Path(image_paths[0]).read_bytes().decode('latin-1')
    assert not (root / 'models/config-user.yaml').exists()
    vault = Vault.create(database, 'synthetic integration password')
    vault.sync(data)
    assert secret.encode() not in vault.backup()
    expected = read_tree(data)
    vault.close()
    vault = Vault.open(database, 'synthetic integration password')
    vault.restore(root / 'restored')
    assert read_tree(root / 'restored') == expected
    vault.close()
    print(json.dumps({'components': len(captured['blocks'].blocks), 'config': config.status_code,
                      'upload': upload.status_code, 'all_content_restored': True}))
finally:
    shutil.rmtree(root)
