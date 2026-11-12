import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

from modules.vault_linux import landlock_abi
from modules.vault_native import prepare_library_aliases
from modules.vault_store import VaultError


def test_library_alias_chains_are_resolved_without_modifying_installation(tmp_path):
    binary_dir = tmp_path / 'installed/bin'
    binary_dir.mkdir(parents=True)
    library = binary_dir / 'libexample.so.1.2'
    library.write_bytes(b'synthetic library')
    manifest = binary_dir / '_symlinks.json'
    original = json.dumps({'libexample.so': 'libexample.so.1', 'libexample.so.1': library.name})
    manifest.write_text(original)
    workspace = tmp_path / 'ram'
    aliases = prepare_library_aliases(binary_dir / 'llama-server', workspace)
    assert aliases.is_relative_to(workspace / 'volatile')
    assert (aliases / 'libexample.so').resolve() == library
    assert (aliases / 'libexample.so.1').read_bytes() == b'synthetic library'
    assert manifest.read_text() == original
    assert not (binary_dir / 'libexample.so').exists()
    assert prepare_library_aliases(binary_dir / 'llama-server', workspace) == aliases


@pytest.mark.parametrize('aliases', [
    {'../outside': 'library.so'},
    {'library.so': '/outside'},
    {'library.so': '../outside'},
    {'one.so': 'two.so', 'two.so': 'one.so'},
])
def test_invalid_library_manifest_is_refused(tmp_path, aliases):
    (tmp_path / '_symlinks.json').write_text(json.dumps(aliases))
    with pytest.raises(ValueError):
        prepare_library_aliases(tmp_path / 'llama-server', tmp_path / 'ram')
    assert not (tmp_path / 'ram').exists()


@pytest.mark.parametrize('confined', [False, True, 'colab'], ids=['loader', 'landlock', 'colab'])
def test_fresh_llama_wheel_loads_and_generates(tmp_path, confined):
    """Use real release binaries and a tiny public model when supplied by CI."""
    wheel = os.environ.get('TEXTGEN_TEST_LLAMA_WHEEL')
    model = os.environ.get('TEXTGEN_TEST_LLAMA_MODEL')
    if not wheel or not model:
        pytest.skip('Set TEXTGEN_TEST_LLAMA_WHEEL and TEXTGEN_TEST_LLAMA_MODEL for real inference tests.')
    if confined == 'colab' and os.geteuid() != 0:
        pytest.skip('Colab inference needs a root supervisor to allocate the worker UID.')
    if confined is True:
        try:
            landlock_abi()
        except VaultError as error:
            pytest.skip(str(error))
    pytest.importorskip('gradio')
    installed = tmp_path / 'installed'
    # Unpack a pristine wheel for EVERY run: importing it before confinement
    # would restore the links and conceal the first-use failure.
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
        for entry in archive.infolist():
            path = installed / entry.filename
            if path.is_file():
                path.chmod((entry.external_attr >> 16) & 0o777)
    ram = Path(tempfile.mkdtemp(prefix='textgen-llama-test-', dir='/dev/shm'))
    isolation = None
    try:
        command = [sys.executable, '-B', 'tests/vault/probe_llama.py', str(installed),
                   str(Path(model).resolve()), str(ram), 'colab' if confined == 'colab' else str(int(confined))]
        options = {}
        if confined == 'colab':
            from modules.vault_colab import ColabIsolation
            for directory in (tmp_path, tmp_path.parent, tmp_path.parent.parent):
                directory.chmod(0o755)
            public_model = installed / 'probe.gguf'
            shutil.copyfile(model, public_model)
            public_model.chmod(0o644)
            command[4] = str(public_model)
            isolation = ColabIsolation()
            (ram / 'data').mkdir()
            isolation.claim(ram)
            command += [str(os.getpid())]
            command = isolation.command(ram, command)
            options = isolation.subprocess_options()
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False, **options)
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads(result.stdout.strip().splitlines()[-1])
        assert summary['generated_tokens'] > 0
        assert summary['unauthenticated_status'] == 401
        assert summary['installation_unchanged'] is True
    finally:
        shutil.rmtree(ram)
        if isolation is not None:
            isolation.close()
