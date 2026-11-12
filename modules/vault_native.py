"""Resolve wheel library aliases without modifying the Python installation."""

import hashlib
import json
import os
from pathlib import Path


def prepare_library_aliases(server_path, workspace):
    """Recreate a native wheel's symlink manifest in volatile RAM only.

    llama_cpp_binaries normally restores these links on import. Landlock
    correctly denies those writes to site-packages, so give the dynamic loader
    a private directory containing aliases to the original, read-only libraries.
    Executables and libraries themselves stay in their installed location.
    """
    binary_dir = Path(server_path).resolve().parent
    manifest = binary_dir / '_symlinks.json'
    if not manifest.is_file():
        return None

    aliases = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(aliases, dict):
        raise TypeError('Invalid native library symlink manifest.')
    for name, target in aliases.items():
        if any(not isinstance(part, str) or not part or Path(part).name != part
               or part in ('.', '..') for part in (name, target)):
            raise ValueError('Native library aliases must be filenames within the wheel.')

    resolved = {}
    for name in aliases:
        target = name
        seen = set()
        while target in aliases:
            if target in seen:
                raise ValueError('Cyclic native library symlink manifest.')
            seen.add(target)
            target = aliases[target]
        library = (binary_dir / target).resolve(strict=True)
        if library.parent != binary_dir or not library.is_file():
            raise ValueError('Native library alias points outside the wheel.')
        resolved[name] = library

    # Separate regular/IK wheels (and different installations) on model switches.
    identity = hashlib.sha256(os.fsencode(binary_dir)).hexdigest()[:24]
    directory = Path(workspace) / 'volatile' / 'native-libraries' / identity
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, library in resolved.items():
        alias = directory / name
        if alias.is_symlink() and alias.readlink() == library:
            continue
        alias.unlink(missing_ok=True)
        alias.symlink_to(library)
    return directory
