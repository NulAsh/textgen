"""Linux primitives for the vault's transient plaintext workspace.

Require tmpfs and Landlock ABI >= 3 (including truncate mediation). Never fall
back to a normal temporary directory or to an unrestricted worker.
"""

import ctypes
import hashlib
import os
import platform
import resource
import shutil
import stat
import sys
from pathlib import Path

from modules.vault_store import VaultError

TMPFS_MAGIC = 0x01021994
WRITE_FILE = 1 << 1
FS_MUTATIONS = WRITE_FILE | sum(1 << bit for bit in range(4, 15))
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long


def landlock_abi():
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise VaultError("The encrypted launcher requires x86_64 or aarch64 Linux.")
    version = libc.syscall(444, 0, 0, 1)
    if version < 3:
        raise VaultError("Landlock ABI 3 or newer is required. The backend will not run without write isolation.")
    return version


def verify_tmpfs(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise VaultError("The RAM directory must be a real directory, not a symlink.")
    buffer = ctypes.create_string_buffer(512)
    if libc.statfs(os.fsencode(path), ctypes.byref(buffer)) != 0 or ctypes.c_long.from_buffer(buffer).value != TMPFS_MAGIC:
        raise VaultError("The runtime directory must be on tmpfs. Disk-backed temporary files are refused.")


def runtime_path(vault_path, base="/dev/shm"):
    verify_tmpfs(base)
    suffix = hashlib.sha256(os.fsencode(Path(vault_path).resolve())).hexdigest()[:24]
    return Path(base) / f"textgen-vault-{os.getuid()}-{suffix}"


def make_runtime(path):
    path = Path(path)
    verify_tmpfs(path.parent)
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise VaultError("Unsafe existing RAM workspace; refusing to use or remove it.")
        # The gateway holds the exclusive vault lock before calling this.
        shutil.rmtree(path)
    path.mkdir(mode=0o700)
    verify_tmpfs(path)


def disable_core_dumps():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def confine_writes(workspace):
    landlock_abi()
    verify_tmpfs(workspace)

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    ruleset = Ruleset(FS_MUTATIONS)
    fd = libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if fd < 0:
        raise VaultError("Unable to create the filesystem write restrictions.")
    try:
        for path, rights in ((workspace, FS_MUTATIONS), ("/dev/null", WRITE_FILE)):
            path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(rights, path_fd)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) != 0:
                    raise VaultError("Unable to configure the filesystem write restrictions.")
            finally:
                os.close(path_fd)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, fd, 0) != 0:
            raise VaultError("Unable to enforce the filesystem write restrictions.")
    finally:
        os.close(fd)


def bind_to_parent(parent_pid):
    # Linux PR_SET_PDEATHSIG. Set before threads or model subprocesses exist.
    import signal
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent_pid:
        raise VaultError("The vault supervisor is no longer running.")
