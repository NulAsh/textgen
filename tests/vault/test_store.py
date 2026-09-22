import json
import os
import sqlite3

import pytest

from modules.vault_store import Vault, VaultError, read_tree

PASSWORD = 'correct horse battery staple — дракон'
SECRET = b'PRIVATE-PROMPT-AND-GENERATED-TEXT-70219'


def fixture_tree(root):
    (root / 'logs/instruct').mkdir(parents=True)
    (root / 'characters').mkdir()
    (root / 'empty-directory').mkdir()
    (root / 'logs/instruct/SECRET-TITLE.json').write_bytes(SECRET)
    (root / 'characters/Лира.yaml').write_text('system: ' + SECRET.decode(), encoding='utf-8')
    (root / 'upload.bin').write_bytes(bytes(range(256)))


def test_roundtrip_and_no_plaintext_in_database_or_journal(tmp_path):
    source = tmp_path / 'source'
    fixture_tree(source)
    vault = Vault.create(tmp_path / 'vault.sqlite3', PASSWORD)
    assert vault.sync(source)
    assert not vault.sync(source)
    for path in tmp_path.glob('vault.sqlite3*'):
        data = path.read_bytes()
        for marker in (SECRET, b'SECRET-TITLE', 'Лира'.encode(), PASSWORD.encode()):
            assert marker not in data
    vault.close()
    reopened = Vault.open(tmp_path / 'vault.sqlite3', PASSWORD)
    reopened.restore(tmp_path / 'restored')
    assert read_tree(source) == read_tree(tmp_path / 'restored')
    assert os.stat(tmp_path / 'vault.sqlite3').st_mode & 0o777 == 0o600
    reopened.close()


def test_wrong_password_does_not_overwrite_data(tmp_path):
    source = tmp_path / 'source'
    fixture_tree(source)
    path = tmp_path / 'vault.sqlite3'
    vault = Vault.create(path, PASSWORD)
    vault.sync(source)
    before = vault.backup()
    vault.close()
    with pytest.raises(VaultError, match='Incorrect password'):
        Vault.open(path, 'wrong password')
    vault = Vault.open(path, PASSWORD)
    assert vault.backup() == before
    vault.close()


@pytest.mark.parametrize('attack', ['flip', 'delete', 'rename_id', 'delete_manifest', 'change_manifest'])
def test_corruption_is_detected_before_restore(tmp_path, attack):
    source = tmp_path / 'source'
    fixture_tree(source)
    path = tmp_path / 'vault.sqlite3'
    vault = Vault.create(path, PASSWORD)
    vault.sync(source)
    vault.close()
    with sqlite3.connect(path) as db:
        key, blob = db.execute('SELECT id,data FROM objects LIMIT 1').fetchone()
        if attack == 'flip':
            damaged = bytearray(blob)
            damaged[-1] ^= 1
            db.execute('UPDATE objects SET data=? WHERE id=?', (bytes(damaged), key))
        elif attack == 'delete':
            db.execute('DELETE FROM objects WHERE id=?', (key,))
        elif attack == 'rename_id':
            db.execute('UPDATE objects SET id=? WHERE id=?', (os.urandom(32), key))
        elif attack == 'delete_manifest':
            db.execute("DELETE FROM metadata WHERE name='manifest'")
        else:
            db.execute("UPDATE metadata SET value=? WHERE name='manifest'", (b'bad',))
    with pytest.raises(VaultError):
        Vault.open(path, PASSWORD)
    assert not (tmp_path / 'restored').exists()


def test_kdf_header_cannot_force_unbounded_allocation(tmp_path):
    path = tmp_path / 'vault.sqlite3'
    Vault.create(path, PASSWORD).close()
    with sqlite3.connect(path) as db:
        header = json.loads(db.execute("SELECT value FROM metadata WHERE name='header'").fetchone()[0])
        header['kdf']['memory_kib'] = 2**60
        db.execute("UPDATE metadata SET value=? WHERE name='header'", (json.dumps(header),))
    with pytest.raises(VaultError):
        Vault.open(path, PASSWORD)


def test_atomic_failed_save_keeps_previous_snapshot(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    fixture_tree(source)
    vault = Vault.create(tmp_path / 'vault.sqlite3', PASSWORD)
    vault.sync(source)
    expected = read_tree(source)
    before = vault.backup()
    (source / 'settings.yaml').write_bytes(SECRET)
    (source / 'upload.bin').unlink()
    def fail(_):
        raise OSError('simulated disk full')
    monkeypatch.setattr(vault, '_save_manifest', fail)
    with pytest.raises(OSError):
        vault.sync(source)
    assert vault.backup() == before
    vault.restore(tmp_path / 'restored')
    assert read_tree(tmp_path / 'restored') == expected
    vault.close()


def test_rename_delete_and_empty_directories(tmp_path):
    source = tmp_path / 'source'
    fixture_tree(source)
    vault = Vault.create(tmp_path / 'vault.sqlite3', PASSWORD)
    vault.sync(source)
    (source / 'upload.bin').rename(source / 'renamed.bin')
    (source / 'characters/Лира.yaml').unlink()
    (source / 'new-empty').mkdir()
    assert vault.sync(source)
    vault.restore(tmp_path / 'restored')
    assert read_tree(source) == read_tree(tmp_path / 'restored')
    vault.close()


@pytest.mark.parametrize('kind', ['symlink_file', 'symlink_dir', 'hardlink', 'fifo', 'missing_root'])
def test_unsafe_files_do_not_destroy_last_snapshot(tmp_path, kind):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'file').write_bytes(SECRET)
    vault = Vault.create(tmp_path / 'vault.sqlite3', PASSWORD)
    vault.sync(source)
    before = vault.backup()
    if kind == 'symlink_file':
        (source / 'link').symlink_to(source / 'file')
    elif kind == 'symlink_dir':
        (source / 'link').symlink_to(tmp_path, target_is_directory=True)
    elif kind == 'hardlink':
        os.link(source / 'file', source / 'link')
    elif kind == 'fifo':
        os.mkfifo(source / 'pipe')
    else:
        source.rename(tmp_path / 'moved')
    with pytest.raises((VaultError, FileNotFoundError)):
        vault.sync(source)
    assert vault.backup() == before
    vault.close()


def test_fresh_nonce_for_each_rewrite(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    path = source / 'file'
    vault = Vault.create(tmp_path / 'vault.sqlite3', PASSWORD)
    ciphertexts = []
    for value in (b'first', b'second', b'first'):
        path.write_bytes(value)
        vault.sync(source)
        ciphertexts.append(vault.db.execute('SELECT data FROM objects').fetchone()[0])
    assert len(set(ciphertexts)) == 3
    assert len({data[:12] for data in ciphertexts}) == 3
    vault.close()


def test_password_change_and_encrypted_backup(tmp_path):
    source = tmp_path / 'source'
    fixture_tree(source)
    vault = Vault.create(tmp_path / 'vault.sqlite3', PASSWORD)
    vault.sync(source)
    backup_path = tmp_path / 'backup.sqlite3'
    backup_path.write_bytes(vault.backup())
    vault.change_password(PASSWORD, 'another long password')
    assert SECRET not in vault.backup()
    with pytest.raises(VaultError):
        vault.check_password(PASSWORD)
    assert vault.check_password('another long password')
    vault.close()
    old = Vault.open(backup_path, PASSWORD)
    old.restore(tmp_path / 'restored')
    assert read_tree(source) == read_tree(tmp_path / 'restored')
    old.close()


def test_existing_vault_is_never_reinitialized(tmp_path):
    path = tmp_path / 'vault.sqlite3'
    Vault.create(path, PASSWORD).close()
    with pytest.raises(FileExistsError):
        Vault.create(path, 'new sufficiently long password')
    Vault.open(path, PASSWORD).close()


def test_failed_initialization_can_be_retried(tmp_path, monkeypatch):
    path = tmp_path / 'vault.sqlite3'
    def fail(self, _):
        raise OSError('disk full during initialization')
    with monkeypatch.context() as patch:
        patch.setattr(Vault, '_save_manifest', fail)
        with pytest.raises(OSError):
            Vault.create(path, PASSWORD)
    assert not list(tmp_path.glob('vault.sqlite3*'))
    Vault.create(path, PASSWORD).close()
