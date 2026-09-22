"""Versioned, authenticated encrypted storage. No TextGen/Gradio imports.

SQLite sees only encrypted records, keyed path hashes, and a password-wrapped
random key. Its WAL, journals and backups therefore also contain ciphertext.
"""

import base64
import hashlib
import hmac
import json
import os
import sqlite3
import stat
import struct
import threading
from pathlib import Path, PurePosixPath

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

FORMAT = "textgen-vault-1"
KDF = {"name": "argon2id", "memory_kib": 65536, "iterations": 3, "lanes": 4}
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


class VaultError(Exception):
    """An error safe to report without including filenames or file contents."""


class SnapshotBusy(VaultError):
    """A file changed during capture; retry without changing the saved state."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def b64(data):
    return base64.b64encode(data).decode("ascii")


def unb64(data):
    return base64.b64decode(data, validate=True)


def derive(password, salt):
    if not isinstance(password, str) or not password or len(password.encode("utf-8")) > 4096:
        raise VaultError("Enter a password of at most 4096 UTF-8 bytes.")
    return Argon2id(salt=salt, length=32, iterations=KDF["iterations"],
                    lanes=KDF["lanes"], memory_cost=KDF["memory_kib"]).derive(password.encode("utf-8"))


def wrap_key(password, root_key, vault_id):
    salt, nonce = os.urandom(16), os.urandom(12)
    header = {"format": FORMAT, "kdf": KDF, "salt": b64(salt), "vault_id": vault_id}
    key = derive(password, salt)
    header["wrapped_key"] = b64(nonce + AESGCM(key).encrypt(nonce, root_key, canonical(header)))
    return header


def unwrap_key(password, header):
    try:
        header = dict(header)
        wrapped = unb64(header.pop("wrapped_key"))
        if set(header) != {"format", "kdf", "salt", "vault_id"} or header["format"] != FORMAT or header["kdf"] != KDF:
            raise ValueError("Unsupported format")
        salt = unb64(header["salt"])
        if len(salt) != 16 or len(wrapped) != 60 or len(bytes.fromhex(header["vault_id"])) != 16:
            raise ValueError("Invalid header")
        return AESGCM(derive(password, salt)).decrypt(wrapped[:12], wrapped[12:], canonical(header))
    except (InvalidTag, ValueError, KeyError, TypeError):
        raise VaultError("Incorrect password or damaged vault.") from None


def safe_name(name):
    path = PurePosixPath(name)
    if not name or "\x00" in name or "\\" in name or path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise VaultError("Invalid path in vault.")
    return path


def pack_record(name, kind, content):
    encoded = str(safe_name(name)).encode("utf-8")
    return struct.pack(">I", len(encoded)) + encoded + kind + content


def unpack_record(data):
    try:
        length, = struct.unpack(">I", data[:4])
        if length > 65536 or 5 + length > len(data):
            raise ValueError()
        name = data[4:4 + length].decode("utf-8")
        safe_name(name)
        kind, content = data[4 + length:5 + length], data[5 + length:]
        if kind not in (b"d", b"f") or (kind == b"d" and content):
            raise ValueError()
        return name, kind, content
    except (ValueError, UnicodeError, struct.error):
        raise VaultError("Damaged vault record.") from None


def read_tree(root):
    """Capture regular files without following symlinks or accepting partial writes."""
    records, total = {}, 0
    root = Path(root)
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode):
        raise VaultError("The vault data directory is unavailable.")
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=lambda e: (_ for _ in ()).throw(e)):
        for filename in sorted(dirs + files):
            path = Path(directory) / filename
            before = path.lstat()
            name = path.relative_to(root).as_posix()
            if stat.S_ISDIR(before.st_mode):
                records[name] = pack_record(name, b"d", b"")
                continue
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise VaultError("Vault data must contain only directories and regular, unlinked files.")
            if before.st_size > MAX_FILE_BYTES:
                raise VaultError("A vault file exceeds the 256 MiB limit.")
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_ino != before.st_ino or opened.st_dev != before.st_dev:
                    raise SnapshotBusy("Data is changing; retrying save.")
                content = stream.read(MAX_FILE_BYTES + 1)
                after = os.fstat(stream.fileno())
            current = path.lstat()
            if len(content) != before.st_size or (before.st_ino, before.st_mtime_ns, before.st_ctime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_ctime_ns, after.st_size) or current.st_ino != before.st_ino:
                raise SnapshotBusy("Data is changing; retrying save.")
            total += len(content)
            if total > MAX_TOTAL_BYTES:
                raise VaultError("Vault data exceeds the 2 GiB limit.")
            records[name] = pack_record(name, b"f", content)
    if root.lstat().st_ino != root_info.st_ino:
        raise SnapshotBusy("Data is changing; retrying save.")
    return records


class Vault:
    def __init__(self, path, connection, root_key, header):
        self.path = Path(path)
        self.db = connection
        self.header = header
        self._root_key = root_key
        keys = HKDF(algorithm=hashes.SHA256(), length=64, salt=bytes.fromhex(header["vault_id"]), info=FORMAT.encode()).derive(root_key)
        self._cipher = AESGCM(keys[:32])
        self._index_key = keys[32:]
        self._lock = threading.RLock()
        self._digests = {}
        self._revision = 0

    @staticmethod
    def _connect(path):
        db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=rw", uri=True, check_same_thread=False, timeout=10)
        db.execute("PRAGMA temp_store=MEMORY")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @classmethod
    def create(cls, path, password):
        if len(password) < 12:
            raise VaultError("Use a password or passphrase of at least 12 characters.")
        path = Path(path)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_key = os.urandom(32)
        header = wrap_key(password, root_key, os.urandom(16).hex())
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        db = None
        try:
            db = cls._connect(path)
            vault = cls(path, db, root_key, header)
            with db:
                db.execute("CREATE TABLE metadata (name TEXT PRIMARY KEY, value BLOB NOT NULL)")
                db.execute("CREATE TABLE objects (id BLOB PRIMARY KEY, data BLOB NOT NULL)")
                db.execute("INSERT INTO metadata VALUES ('header', ?)", (canonical(header),))
                vault._save_manifest({})
            return vault
        except BaseException:
            if db is not None:
                db.close()
            # Only our newly reserved, never-initialized database is removed.
            for suffix in ('', '-wal', '-shm', '-journal'):
                Path(str(path) + suffix).unlink(missing_ok=True)
            raise

    @classmethod
    def open(cls, path, password):
        db = cls._connect(path)
        try:
            row = db.execute("SELECT value FROM metadata WHERE name='header'").fetchone()
            if not row or len(row[0]) > 4096:
                raise VaultError("Damaged vault header.")
            header = json.loads(row[0])
            vault = cls(path, db, unwrap_key(password, header), header)
            vault._verified_records()
            return vault
        except BaseException:
            db.close()
            raise

    def _id(self, name):
        return hmac.digest(self._index_key, name.encode("utf-8"), "sha256")

    def _seal(self, data, context):
        nonce = os.urandom(12)
        aad = FORMAT.encode() + bytes.fromhex(self.header["vault_id"]) + context
        return nonce + self._cipher.encrypt(nonce, data, aad)

    def _unseal(self, data, context):
        try:
            aad = FORMAT.encode() + bytes.fromhex(self.header["vault_id"]) + context
            return self._cipher.decrypt(data[:12], data[12:], aad)
        except (InvalidTag, ValueError):
            raise VaultError("Vault integrity check failed. No data has been restored.") from None

    def _save_manifest(self, hashes_by_id):
        data = canonical({"revision": self._revision, "objects": hashes_by_id})
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('manifest', ?)", (self._seal(data, b"manifest"),))

    def _verified_records(self):
        row = self.db.execute("SELECT value FROM metadata WHERE name='manifest'").fetchone()
        if not row:
            raise VaultError("Missing vault manifest.")
        manifest = json.loads(self._unseal(row[0], b"manifest"))
        objects = self.db.execute("SELECT id, data FROM objects").fetchall()
        actual = {key.hex(): hashlib.sha256(data).hexdigest() for key, data in objects}
        if actual != manifest["objects"]:
            raise VaultError("Vault inventory check failed. No data has been restored.")
        records = {}
        for key, data in objects:
            plain = self._unseal(data, b"object" + key)
            name, kind, content = unpack_record(plain)
            if not hmac.compare_digest(key, self._id(name)) or name in records:
                raise VaultError("Vault path integrity check failed.")
            records[name] = (kind, content)
            self._digests[name] = hashlib.sha256(plain).digest()
        self._revision = manifest["revision"]
        return records

    def restore(self, destination):
        with self._lock:
            records = self._verified_records()
            root = Path(destination)
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if any(root.iterdir()):
                raise VaultError("Restore destination must be empty.")
            for name, (kind, content) in sorted(records.items(), key=lambda x: (len(PurePosixPath(x[0]).parts), x[0])):
                path = root / safe_name(name)
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if kind == b"d":
                    path.mkdir(mode=0o700, exist_ok=True)
                else:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(content)

    def sync(self, source):
        with self._lock:
            try:
                records = read_tree(source)
            except FileNotFoundError:
                if Path(source).is_dir():
                    raise SnapshotBusy('Data is changing; retrying save.') from None
                raise
            digests = {name: hashlib.sha256(data).digest() for name, data in records.items()}
            if digests == self._digests:
                return False
            previous_revision = self._revision
            try:
                with self.db:
                    for name, plain in records.items():
                        if digests[name] != self._digests.get(name):
                            key = self._id(name)
                            self.db.execute("INSERT OR REPLACE INTO objects VALUES (?, ?)", (key, self._seal(plain, b"object" + key)))
                    for name in self._digests.keys() - records.keys():
                        self.db.execute("DELETE FROM objects WHERE id=?", (self._id(name),))
                    hashes_by_id = {key.hex(): hashlib.sha256(data).hexdigest() for key, data in self.db.execute("SELECT id, data FROM objects")}
                    self._revision += 1
                    self._save_manifest(hashes_by_id)
            except BaseException:
                self._revision = previous_revision
                raise
            self._digests = digests
            return True

    def check_password(self, password):
        with self._lock:
            return hmac.compare_digest(unwrap_key(password, self.header), self._root_key)

    def change_password(self, old_password, new_password):
        with self._lock:
            if not self.check_password(old_password):
                raise VaultError("Incorrect password.")
            if len(new_password) < 12:
                raise VaultError("Use at least 12 characters.")
            # Rotate the data key as well as the password wrapper. Old backups
            # retain their historical content, but an old wrapper/key cannot
            # decrypt subsequently written records in this vault.
            records = self._verified_records()
            root_key = os.urandom(32)
            header = wrap_key(new_password, root_key, self.header["vault_id"])
            replacement = Vault(self.path, self.db, root_key, header)
            replacement._revision = self._revision + 1
            with self.db:
                self.db.execute('DELETE FROM objects')
                hashes_by_id = {}
                for name, (kind, content) in records.items():
                    key = replacement._id(name)
                    data = replacement._seal(pack_record(name, kind, content), b'object' + key)
                    self.db.execute('INSERT INTO objects VALUES (?, ?)', (key, data))
                    hashes_by_id[key.hex()] = hashlib.sha256(data).hexdigest()
                replacement._save_manifest(hashes_by_id)
                self.db.execute("UPDATE metadata SET value=? WHERE name='header'", (canonical(header),))
            self.header = header
            self._root_key = replacement._root_key
            self._cipher = replacement._cipher
            self._index_key = replacement._index_key
            self._revision = replacement._revision

    def backup(self):
        """Return a consistent single-file SQLite backup, containing ciphertext only."""
        with self._lock:
            copy = sqlite3.connect(":memory:")
            try:
                self.db.backup(copy)
                return copy.serialize()
            finally:
                copy.close()

    def close(self):
        with self._lock:
            try:
                self.db.close()
            finally:
                self._cipher = self._index_key = self._root_key = None
                self._digests.clear()
