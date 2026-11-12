"""Colab write isolation using bubblewrap and a separate host UID.

No new proc filesystem is mounted: Colab blocks that operation. The existing
proc mount is read-only, and a dedicated unprivileged host identity prevents
access through the notebook/supervisor's /proc/PID/root or file descriptors.
Every launch verifies the resulting mount layout before importing TextGen.
"""

import argparse
import ctypes
import fcntl
import json
import os
import pwd
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from modules.vault_linux import disable_core_dumps, verify_tmpfs
from modules.vault_store import VaultError


def verify_sandbox(workspace, supervisor_pid):
    workspace = Path(workspace).resolve()
    verify_tmpfs(workspace)
    libc = ctypes.CDLL(None, use_errno=True)
    if os.getuid() == 0 or libc.prctl(39, 0, 0, 0, 0) != 1:  # PR_GET_NO_NEW_PRIVS
        raise VaultError('Colab worker privilege isolation was not applied.')
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    if any(int(status[name].strip(), 16) for name in ('CapEff', 'CapPrm', 'CapAmb')):
        raise VaultError('Colab worker still has capabilities.')
    if not os.statvfs('/').f_flag & os.ST_RDONLY:
        raise VaultError('Colab worker root filesystem is writable.')
    # Verify every visible writable mount, including mounts below a read-only
    # parent. Only private RAM, private devices, and selected GPU nodes qualify.
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        mount = Path(fields[4].replace('\\040', ' ').replace('\\011', '\t').replace('\\012', '\n').replace('\\134', '\\'))
        if 'rw' not in fields[5].split(','):
            continue
        kind = fields[fields.index('-') + 1]
        if kind == 'tmpfs' and (mount == workspace or workspace in mount.parents or mount == Path('/dev') or Path('/dev') in mount.parents):
            continue
        if kind == 'devpts' and mount == Path('/dev/pts'):
            continue
        if mount.parent == Path('/dev') and mount.name.startswith('nvidia') and stat.S_ISCHR(mount.stat().st_mode):
            continue
        raise VaultError('Colab sandbox contains an unexpected writable mount.')
    # These must never provide a route back to the supervisor's writable root.
    for pid in {supervisor_pid, os.getppid()}:
        try:
            os.readlink(f'/proc/{pid}/root')
        except (PermissionError, FileNotFoundError):
            pass
        else:
            raise VaultError('Colab worker can access an unsandboxed parent through procfs.')


class ColabIsolation:
    def __init__(self):
        if os.geteuid() != 0:
            raise VaultError('Colab isolation requires the root notebook runtime to allocate an unprivileged worker UID.')
        self.binary = shutil.which('bwrap')
        if self.binary is None:
            raise VaultError('Install bubblewrap before starting the Colab vault.')
        # Allocate an unused host identity, not just a different UID inside a
        # user namespace. Never reuse nobody or an existing notebook identity.
        used = {entry.pw_uid for entry in pwd.getpwall()}
        for path in Path('/proc').glob('[0-9]*/status'):
            try:
                for line in path.read_text().splitlines():
                    if line.startswith('Uid:'):
                        used.update(map(int, line.split()[1:]))
            except (OSError, ValueError):
                pass
        self.uid = None
        self.lock_fd = None
        verify_tmpfs('/dev/shm')
        for uid in range(60000 + secrets.randbelow(10000), 100000):
            if uid in used:
                continue
            fd = os.open(f'/dev/shm/textgen-colab-uid-{uid}.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_nlink != 1:
                os.close(fd)
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            self.uid, self.lock_fd = uid, fd
            break
        if self.uid is None:
            raise VaultError('Cannot allocate an isolated Colab worker identity.')

    def close(self):
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None

    def claim(self, workspace):
        """Give the worker only its private RAM tree; never chmod host paths."""
        workspace = Path(workspace)
        verify_tmpfs(workspace)
        (workspace / 'volatile').mkdir(mode=0o700, exist_ok=True)
        # Keep the outer directory owned by the supervisor so crash recovery
        # can authenticate/remove it. The worker can traverse but not replace
        # its data/volatile roots; their contents belong to its private UID.
        os.chown(workspace, 0, self.uid)
        os.chmod(workspace, 0o710)
        for root, directories, files in os.walk(workspace):
            for name in directories + files:
                path = Path(root) / name
                if path.is_symlink():
                    raise VaultError('Unexpected symlink before starting the Colab worker.')
                os.chown(path, self.uid, self.uid)

    def command(self, workspace, command):
        workspace = str(Path(workspace).resolve())
        args = [self.binary, '--unshare-user', '--disable-userns', '--cap-drop', 'ALL',
                '--die-with-parent', '--ro-bind', '/', '/', '--dev', '/dev',
                '--bind', workspace, workspace]
        # No block devices, host shared memory, terminals or broad /dev bind.
        for path in sorted(Path('/dev').glob('nvidia*')):
            if stat.S_ISCHR(path.stat().st_mode):
                args += ['--dev-bind', str(path), str(path)]
            elif path.name == 'nvidia-caps' and path.is_dir():
                for device in sorted(path.iterdir()):
                    if stat.S_ISCHR(device.stat().st_mode):
                        args += ['--dev-bind', str(device), str(device)]
        return args + ['--'] + list(command)

    def subprocess_options(self):
        return {'user': self.uid, 'group': self.uid, 'extra_groups': []}

    def preflight(self, ram_dir='/dev/shm'):
        """Test real isolation, including child inheritance, before decrypting."""
        verify_tmpfs(ram_dir)
        ram = Path(tempfile.mkdtemp(prefix='textgen-colab-check-', dir=ram_dir))
        outside = None
        try:
            fd, name = tempfile.mkstemp(prefix='textgen-colab-outside-')
            outside = Path(name)
            with os.fdopen(fd, 'w') as stream:
                stream.write('unchanged')
            outside.chmod(0o666)  # Must be denied by isolation, not ordinary DAC.
            self.claim(ram)
            command = [sys.executable, '-B', '-m', 'modules.vault_colab', '--probe', str(ram), str(os.getpid()), str(outside)]
            result = subprocess.run(self.command(ram, command), capture_output=True, text=True, timeout=30,
                                    check=False, **self.subprocess_options())
            if result.returncode or result.stdout.strip() != 'isolated' or outside.read_text() != 'unchanged':
                raise VaultError('Colab filesystem isolation failed its startup test. No vault was unlocked. '
                                 'The runtime must support unprivileged user namespaces and bubblewrap. '
                                 + result.stderr[-2000:])
        except (OSError, subprocess.TimeoutExpired) as error:
            raise VaultError('Unable to start the Colab filesystem sandbox. No vault was unlocked.') from error
        finally:
            shutil.rmtree(ram)
            if outside is not None:
                outside.unlink(missing_ok=True)


def probe(workspace, parent, outside):
    verify_sandbox(workspace, parent)
    (Path(workspace) / 'volatile/allowed').write_text('RAM only')
    for path in (outside, f'/proc/{parent}/root{outside}', f'/proc/{os.getppid()}/root{outside}'):
        try:
            Path(path).write_text('forbidden')
        except OSError:
            pass
        else:
            raise AssertionError('Unconfined write succeeded')
    code = 'from pathlib import Path; Path(' + repr(str(outside)) + ').write_text("forbidden")'
    child = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, check=False)
    if child.returncode == 0:
        raise AssertionError('Child did not inherit isolation')
    print('isolated')


def main():
    parser = argparse.ArgumentParser(description='Verify Colab vault isolation without opening a vault')
    parser.add_argument('--probe', nargs=3, metavar=('RAM', 'PARENT', 'OUTSIDE'))
    args = parser.parse_args()
    disable_core_dumps()
    if args.probe:
        ram, parent, outside = args.probe
        probe(ram, int(parent), outside)
    else:
        isolation = ColabIsolation()
        try:
            isolation.preflight()
        finally:
            isolation.close()
        print(json.dumps({'isolation': 'bubblewrap', 'verified': True}))


if __name__ == '__main__':
    main()
