"""Exercise a real queued Gradio event without an implicit construction loop.

Synthetic data only. The UI is deliberately constructed in a thread with no
event loop, matching Python 3.14's behavior before the server starts.
"""

import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
root = Path(tempfile.mkdtemp(prefix='textgen-queue-probe-', dir='/dev/shm'))
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['GRADIO_TEMP_DIR'] = str(root)

try:
    import gradio as gr
    from fastapi.testclient import TestClient

    from modules.vault_runtime import build_app

    def build():
        def stream(text):
            yield text + '-first'
            yield text + '-complete'

        with gr.Blocks() as blocks:
            text = gr.Textbox()
            button = gr.Button()
            button.click(stream, text, text, api_name='queue_probe')
        blocks.queue()
        return build_app(blocks, [])

    with ThreadPoolExecutor(max_workers=1) as pool:
        app = pool.submit(build).result()

    completions = []
    with TestClient(app, raise_server_exceptions=False) as client:
        for turn in range(2):
            session = f'synthetic-session-{turn}'
            text = f'SYNTHETIC-QUEUE-CONTENT-{turn}'
            response = client.post('/queue/join?', json={'data': [text], 'fn_index': 0, 'session_hash': session})
            assert response.status_code == 200, (response.status_code, response.text)
            event_id = response.json()['event_id']
            with client.stream('GET', '/queue/data', params={'session_hash': session}) as events:
                assert events.status_code == 200
                messages = [json.loads(line[6:]) for line in events.iter_lines() if line.startswith('data: ')]
            completed = next(message for message in messages if message['msg'] == 'process_completed')
            assert completed['success'], completed
            assert completed['event_id'] == event_id
            assert completed['output']['data'] == [text + '-complete'], completed
            assert any(message['msg'] == 'process_generating' for message in messages)
            completions.append(completed['success'])
    print(json.dumps({'queued_streams_completed': len(completions)}))
finally:
    shutil.rmtree(root)
