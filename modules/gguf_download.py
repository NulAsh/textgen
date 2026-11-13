"""Download public model weights outside the vault worker, with GGUF checks."""

import argparse
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from modules.metadata_gguf import GGUFError, load_metadata, validate_header


def model_download_url(url):
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Use a public HTTPS GGUF download URL without embedded credentials.')
    parts = parsed.path.split('/')
    # A Hugging Face file page returns HTML with HTTP 200, even with a .gguf
    # suffix. Only rewrite the known host and repository route, not other URLs.
    if parsed.hostname == 'huggingface.co' and len(parts) >= 6 and parts[3] == 'blob':
        parts[3] = 'resolve'
    filename = urllib.parse.unquote(parts[-1])
    if not filename.lower().endswith('.gguf') or any(c in filename for c in ('/', '\\', '\x00')):
        raise ValueError('Use a download link ending in a GGUF filename.')
    return urllib.parse.urlunsplit(parsed._replace(path='/'.join(parts), fragment='')), filename


def download_model(url, model_dir):
    url, filename = model_download_url(url)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / filename
    if destination.exists():
        try:
            load_metadata(destination)
        except GGUFError as error:
            print(f'Existing file failed validation: {error}', flush=True)
            print('Downloading a replacement; the existing file is kept until the new metadata passes.', flush=True)
        else:
            print(f'GGUF metadata OK: {destination.name} ({destination.stat().st_size:,} bytes)', flush=True)
            return destination

    print(f'Downloading {filename}…', flush=True)
    request = urllib.request.Request(url, headers={'Accept-Encoding': 'identity'})
    partial = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise ValueError(f'Download returned HTTP {response.status}; expected the complete file (HTTP 200).')
            if urllib.parse.urlsplit(response.geturl()).scheme != 'https':
                raise ValueError('The model download redirected to an unencrypted URL.')
            header = response.read(24)
            validate_header(header)
            expected_size = response.headers.get('Content-Length')
            expected_size = int(expected_size) if expected_size is not None else None
            # Use a unique temporary file. Failed/partial downloads never get
            # a .gguf name and cannot replace a previously downloaded model.
            with tempfile.NamedTemporaryFile(prefix=f'.{filename}.', suffix='.part', dir=model_dir, delete=False) as file:
                partial = Path(file.name)
                file.write(header)
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
                actual_size = file.tell()
            if expected_size is not None and actual_size != expected_size:
                raise ValueError(f'Incomplete model download: got {actual_size:,} of {expected_size:,} bytes. Retry the download.')
        print('Checking GGUF metadata…', flush=True)
        load_metadata(partial)
        # The dedicated Colab worker must be able to read public model weights.
        partial.chmod(0o644)
        os.replace(partial, destination)
        print(f'GGUF metadata OK: {destination.name} ({actual_size:,} bytes)', flush=True)
        return destination
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def check_models(model_dir):
    paths = sorted(Path(model_dir).rglob('*.gguf'))
    if not paths:
        print(f'Upload GGUF weights to {model_dir} or rerun cell 2 with a model URL.')
    valid = True
    for path in paths:
        try:
            load_metadata(path)
        except (GGUFError, OSError) as error:
            print(f'INVALID: {error}', flush=True)
            valid = False
        else:
            print(f'GGUF metadata OK: {path.name} ({path.stat().st_size:,} bytes)', flush=True)
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--url', default='', help='Public HTTPS GGUF URL, or omit to check existing files.')
    args = parser.parse_args()
    try:
        if args.url.strip():
            download_model(args.url, args.model_dir)
            return 0
        return 0 if check_models(args.model_dir) else 1
    except (ValueError, OSError) as error:
        print(f'Model preparation failed: {error}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
