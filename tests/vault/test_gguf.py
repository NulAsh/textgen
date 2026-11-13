"""GGUF download/metadata regressions for the Colab model-loading failure."""

import io
import struct

import pytest

from modules.gguf_download import check_models, download_model, model_download_url
from modules.metadata_gguf import GGUFError, load_metadata


def string(value):
    if isinstance(value, str):
        value = value.encode()
    return struct.pack('<Q', len(value)) + value


def entry(key, kind, value):
    return string(key) + struct.pack('<I', kind) + value


def gguf(*entries, version=3):
    return b'GGUF' + struct.pack('<IQQ', version, 0, len(entries)) + b''.join(entries)


@pytest.mark.parametrize('version', [2, 3])
def test_supported_metadata_preserves_tokens_templates_and_scalars(tmp_path, version):
    path = tmp_path / 'model.gguf'
    template = '{{ messages }} → {{ eos_token }}'
    path.write_bytes(gguf(
        entry('general.architecture', 8, string('llama')),
        entry('llama.context_length', 4, struct.pack('<I', 8192)),
        entry('tokenizer.chat_template', 8, string(template)),
        entry('tokenizer.ggml.tokens', 9, struct.pack('<IQ', 8, 3) + string('<s>') + string('hello') + string(b'\xff')),
        entry('tokenizer.ggml.scores', 9, struct.pack('<IQff', 6, 2, -1.0, 0.5)),
        entry('tokenizer.ggml.add_bos_token', 7, b'\x01'),
        entry('empty', 9, struct.pack('<IQ', 4, 0)),
        version=version,
    ))
    assert load_metadata(path) == {
        'general.architecture': 'llama',
        'llama.context_length': 8192,
        'tokenizer.chat_template': template,
        'tokenizer.ggml.tokens': ['<s>', 'hello', b'\xff'],
        'tokenizer.ggml.scores': [-1.0, 0.5],
        'tokenizer.ggml.add_bos_token': True,
        'empty': [],
    }


@pytest.mark.parametrize(('payload', 'message'), [
    (b'<!DOCTYPE html>\n<html><head><title>Model file page</title></head></html>', 'HTML page'),
    (b'version https://git-lfs.github.com/spec/v1\noid sha256:0000\nsize 1000000', 'Git LFS pointer'),
    (b'', 'signature'),
    (b'{"error":"Access denied"}', 'signature'),
    (b'GGUF\x03', 'truncated'),
    (gguf(version=1), 'outdated'),
    (gguf(version=4), 'Unsupported GGUF version'),
    (b'GGUF' + struct.pack('>IQQ', 3, 0, 0), 'Big-endian'),
    (b'GGUF' + struct.pack('<IQQ', 3, 0, 2**63), 'length exceeds'),
    (gguf(struct.pack('<Q', 2**63) + b'padding'), 'key length'),
    (gguf(entry('key', 8, struct.pack('<Q', 2**63))), 'length exceeds'),
    (gguf(entry('key', 9, struct.pack('<IQ', 8, 2**63))), 'length exceeds'),
    (gguf(entry('key', 123, b'padding')), 'value type'),
    (gguf(entry('key', 9, struct.pack('<IQ', 9, 0))), 'Nested'),
    (gguf(entry('key', 10, b'\x00')), 'length exceeds'),
])
def test_bad_downloads_and_lengths_fail_without_memory_errors(tmp_path, payload, message):
    path = tmp_path / 'broken.gguf'
    path.write_bytes(payload)
    with pytest.raises(GGUFError, match=message):
        load_metadata(path)


def test_sparse_large_file_cannot_request_unbounded_string_allocation(tmp_path):
    path = tmp_path / 'large.gguf'
    length = 129 * 1024 * 1024
    with path.open('wb') as file:
        file.write(gguf(entry('key', 8, struct.pack('<Q', length))))
        file.truncate(file.tell() + length)
    with pytest.raises(GGUFError, match='128 MiB'):
        load_metadata(path)


def test_array_object_count_is_bounded_even_when_encoded_bytes_fit(tmp_path):
    path = tmp_path / 'array.gguf'
    count = 2_000_001
    with path.open('wb') as file:
        file.write(gguf(entry('key', 9, struct.pack('<IQ', 0, count))))
        file.truncate(file.tell() + count)
    with pytest.raises(GGUFError, match='item safety limit'):
        load_metadata(path)


@pytest.mark.parametrize(('url', 'expected'), [
    ('https://huggingface.co/org/repo/blob/main/model.gguf?download=true',
     'https://huggingface.co/org/repo/resolve/main/model.gguf?download=true'),
    ('https://huggingface.co/org/repo/blob/refs%2Fpr%2F1/sub/model.gguf',
     'https://huggingface.co/org/repo/resolve/refs%2Fpr%2F1/sub/model.gguf'),
    ('https://huggingface.co/org/repo/resolve/main/model.gguf',
     'https://huggingface.co/org/repo/resolve/main/model.gguf'),
    ('https://example.com/org/repo/blob/main/model.gguf',
     'https://example.com/org/repo/blob/main/model.gguf'),
])
def test_download_links_are_normalized_only_for_known_file_page_routes(url, expected):
    assert model_download_url(url) == (expected, 'model.gguf')


@pytest.mark.parametrize('url', [
    'http://example.com/model.gguf', 'file:///tmp/model.gguf',
    'https://example.com/model.html', 'https://user:password@example.com/model.gguf',
    'https://example.com/..%2Fmodel.gguf', 'https://example.com/model%00.gguf',
])
def test_invalid_download_urls_are_rejected(url):
    with pytest.raises(ValueError):
        model_download_url(url)


class Response(io.BytesIO):
    status = 200

    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data) if length is None else length)}

    def geturl(self):
        return 'https://cdn.example.com/model.gguf'


def test_bad_existing_download_is_replaced_only_after_validated_download(tmp_path, monkeypatch):
    old = tmp_path / 'model.gguf'
    old.write_bytes(b'<!DOCTYPE html><html>file page</html>')
    payload = gguf(entry('general.architecture', 8, string('llama')))
    urls = []

    def open_response(request, timeout):
        urls.append(request.full_url)
        return Response(payload)

    monkeypatch.setattr('urllib.request.urlopen', open_response)
    assert download_model('https://huggingface.co/org/repo/blob/main/model.gguf', tmp_path) == old
    assert urls == ['https://huggingface.co/org/repo/resolve/main/model.gguf']
    assert old.read_bytes() == payload
    assert old.stat().st_mode & 0o777 == 0o644
    assert list(tmp_path.iterdir()) == [old]
    # Reuse a valid file; no second network download.
    download_model('https://huggingface.co/org/repo/resolve/main/model.gguf', tmp_path)
    assert len(urls) == 1


@pytest.mark.parametrize(('payload', 'length', 'message'), [
    (b'<!DOCTYPE html><html>still a page</html>', None, 'HTML page'),
    (gguf(entry('key', 8, struct.pack('<Q', 2**63))), None, 'length exceeds'),
    (gguf(), 200, 'Incomplete model download'),
])
def test_failed_download_keeps_existing_file_and_removes_partial(tmp_path, monkeypatch, payload, length, message):
    old = tmp_path / 'model.gguf'
    original = b'original incomplete download'
    old.write_bytes(original)
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: Response(payload, length))
    with pytest.raises(ValueError, match=message):
        download_model('https://example.com/model.gguf', tmp_path)
    assert old.read_bytes() == original
    assert list(tmp_path.iterdir()) == [old]


def test_interrupted_download_is_not_installed(tmp_path, monkeypatch):
    class Interrupted(Response):
        def read(self, n=-1):
            if self.tell() >= 24:
                raise OSError('network interrupted')
            return super().read(n)

    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: Interrupted(gguf()))
    with pytest.raises(OSError, match='network interrupted'):
        download_model('https://example.com/model.gguf', tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_upload_checker_reports_invalid_files_without_dumping_contents(tmp_path, capsys):
    (tmp_path / 'valid.gguf').write_bytes(gguf())
    (tmp_path / 'invalid.gguf').write_bytes(b'<!DOCTYPE html><html>PRIVATE-CONTENT</html>')
    assert not check_models(tmp_path)
    output = capsys.readouterr().out
    assert 'GGUF metadata OK: valid.gguf' in output
    assert 'INVALID:' in output and 'HTML page' in output
    assert 'PRIVATE-CONTENT' not in output
