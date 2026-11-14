"""Colab write isolation using bubblewrap and a separate host UID.

No new proc filesystem is mounted: Colab blocks that operation. Process-local
procfs controls stay writable for CUDA thread names; global controls are
read-only. A dedicated unprivileged host identity prevents
access through the notebook/supervisor's /proc/PID/root or file descriptors.
Every launch verifies the resulting mount layout before importing TextGen.
"""

import argparse
import ctypes
import errno
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

PROC_READONLY = tuple(Path('/proc') / name for name in
                      ('sys', 'sysrq-trigger', 'irq', 'bus', 'fs', 'driver', 'acpi', 'scsi'))


def has_nvidia_gpu():
    return any(path.name[6:].isdigit() and stat.S_ISCHR(path.stat().st_mode)
               for path in Path('/dev').glob('nvidia[0-9]*'))


def create_syscall_filter():
    """Block namespace/mount changes without writing container sysctls.

    Bubblewrap loads this filter before executing Python. Colab can expose a
    read-only /proc/sys, where bubblewrap's --disable-userns cannot work.
    """
    class Argument(ctypes.Structure):
        _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_int),
                    ('datum_a', ctypes.c_uint64), ('datum_b', ctypes.c_uint64)]

    try:
        library = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    except OSError as error:
        raise VaultError('Install libseccomp2 before starting the Colab vault.') from error
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                             ctypes.c_uint, ctypes.POINTER(Argument)]
    library.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    context = library.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW; other architectures are refused.
    if not context:
        raise VaultError('Unable to initialize the Colab syscall filter.')
    fd = None
    try:
        blocked = ('unshare', 'setns', 'mount', 'umount2', 'pivot_root', 'move_mount',
                   'open_tree', 'fsopen', 'fsconfig', 'fsmount', 'fspick', 'mount_setattr')
        for name in (*blocked, 'clone', 'clone3'):
            number = library.seccomp_syscall_resolve_name(name.encode())
            if number < 0:
                raise VaultError('libseccomp is too old for the required Colab syscall filter.')
            # clone3 has pointer-based arguments: return ENOSYS so libraries
            # fall back to clone(), whose CLONE_NEWUSER bit can be checked.
            action = 0x00050000 | (errno.ENOSYS if name == 'clone3' else errno.EPERM)
            argument = Argument(0, 7, 0x10000000, 0x10000000)  # SCMP_CMP_MASKED_EQ, CLONE_NEWUSER
            count = int(name == 'clone')
            if library.seccomp_rule_add_array(context, action, number, count, ctypes.byref(argument) if count else None) < 0:
                raise VaultError('Unable to build the Colab syscall filter.')
        fd = os.memfd_create('textgen-vault-seccomp', os.MFD_CLOEXEC)
        if library.seccomp_export_bpf(context, fd) < 0:
            raise VaultError('Unable to export the Colab syscall filter.')
        # Only a read descriptor is inherited by bubblewrap, which consumes it.
        readonly = os.open(f'/proc/self/fd/{fd}', os.O_RDONLY | os.O_CLOEXEC)
        return readonly
    finally:
        library.seccomp_release(context)
        if fd is not None:
            os.close(fd)


def verify_mounts(workspace):
    workspace = Path(workspace).resolve()
    if not os.statvfs('/').f_flag & os.ST_RDONLY:
        raise VaultError('Colab worker root filesystem is writable.')
    for path in PROC_READONLY:
        if path.exists() and not os.statvfs(path).f_flag & os.ST_RDONLY:
            raise VaultError('Colab global procfs controls are writable.')
    # Verify every visible writable mount, including mounts below a read-only
    # parent. Exceptions are private RAM/devices, selected GPU nodes and the
    # procfs process metadata described below; persistent files stay read-only.
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        mount = Path(fields[4].replace('\\040', ' ').replace('\\011', '\t').replace('\\012', '\n').replace('\\134', '\\'))
        if 'rw' not in fields[5].split(','):
            continue
        # mountinfo also lists hidden mounts below overmounts. Resolve the
        # actual reachable mount ID instead of mistaking an obscured host
        # mount for the worker's view. No capabilities or new namespaces can
        # be used to uncover these mounts after setup.
        try:
            fd = os.open(mount, os.O_PATH | os.O_CLOEXEC)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
        try:
            info = dict(line.split(':', 1) for line in Path(f'/proc/self/fdinfo/{fd}').read_text().splitlines())
            if info['mnt_id'].strip() != fields[0]:
                continue
        finally:
            os.close(fd)
        kind = fields[fields.index('-') + 1]
        if kind == 'proc' and mount == Path('/proc') and fields[3] == '/':
            # Kernel process metadata, not a persistent filesystem. The
            # private host UID and verify_sandbox's parent checks are essential:
            # proc magic links must not expose an unsandboxed writable root.
            continue
        if kind == 'tmpfs' and (mount == workspace or workspace in mount.parents or mount == Path('/dev') or Path('/dev') in mount.parents):
            continue
        if kind == 'devpts' and mount == Path('/dev/pts'):
            continue
        device = ((mount.parent == Path('/dev') and (mount.name in {'null', 'zero', 'full', 'random', 'urandom', 'tty'}
                   or mount.name.startswith('nvidia'))) or mount.parent == Path('/dev/nvidia-caps'))
        if device and stat.S_ISCHR(mount.stat().st_mode):
            continue
        raise VaultError('Colab sandbox contains an unexpected writable mount.')


def verify_sandbox(workspace, supervisor_pid):
    verify_tmpfs(workspace)
    libc = ctypes.CDLL(None, use_errno=True)
    if os.getuid() == 0 or libc.prctl(39, 0, 0, 0, 0) != 1:  # PR_GET_NO_NEW_PRIVS
        raise VaultError('Colab worker privilege isolation was not applied.')
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    if any(int(status[name].strip(), 16) for name in ('CapEff', 'CapPrm', 'CapAmb')):
        raise VaultError('Colab worker still has capabilities.')
    # Probe in a child so a rejected unconfined entry never changes the
    # caller's namespace. Children must inherit the same irreversible filter.
    code = ('import ctypes, errno, sys; libc = ctypes.CDLL(None, use_errno=True); '
            'sys.exit(0 if libc.unshare(0x10000000) == -1 and ctypes.get_errno() == errno.EPERM else 1)')
    if status['Seccomp'].strip() != '2' or subprocess.run(
            [sys.executable, '-B', '-c', code], capture_output=True, check=False, timeout=10).returncode:
        raise VaultError('Colab worker can create new user namespaces.')
    verify_mounts(workspace)
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
        self.filter_fd = None
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
        try:
            self.filter_fd = create_syscall_filter()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
        if self.filter_fd is not None:
            os.close(self.filter_fd)
            self.filter_fd = None

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
        os.lseek(self.filter_fd, 0, os.SEEK_SET)
        args = [self.binary, '--unshare-user', '--seccomp', str(self.filter_fd), '--cap-drop', 'ALL',
                '--die-with-parent', '--ro-bind', '/', '/', '--dev', '/dev',
                '--bind', workspace, workspace, '--bind', '/proc', '/proc']
        # CUDA's cuInit may rename threads through /proc/PID/task/TID/comm.
        # Reuse procfs (Colab forbids a fresh proc mount) without making global
        # kernel controls writable. All disk mounts remain read-only.
        for path in PROC_READONLY:
            if path.exists():
                args += ['--ro-bind', str(path), str(path)]
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
        return {'user': self.uid, 'group': self.uid, 'extra_groups': [], 'pass_fds': (self.filter_fd,)}

    def preflight(self, ram_dir='/dev/shm', check_cuda=False):
        """Test real isolation, including child inheritance, before decrypting."""
        verify_tmpfs(ram_dir)
        ram = Path(tempfile.mkdtemp(prefix='textgen-colab-check-', dir=ram_dir))
        outside = None
        cuda_info = None
        try:
            fd, name = tempfile.mkstemp(prefix='textgen-colab-outside-')
            outside = Path(name)
            with os.fdopen(fd, 'w') as stream:
                stream.write('unchanged')
            outside.chmod(0o666)  # Must be denied by isolation, not ordinary DAC.
            env = os.environ.copy()
            if check_cuda:
                cache = ram / 'volatile/cuda-check'
                cache.mkdir(parents=True)
                env['CUDA_CACHE_PATH'] = str(cache)
                env['TMPDIR'] = str(cache)
                # Initialize before enumerating device nodes: nvidia-uvm may
                # be created lazily. Never initialize CUDA in the supervisor
                # that will later hold the vault key.
                host = subprocess.run([sys.executable, '-B', '-m', 'modules.vault_cuda'],
                                      capture_output=True, text=True, timeout=30, check=False, env=env)
                if host.returncode:
                    raise VaultError('CUDA failed in the notebook runtime before sandboxing. '
                                     'Check that a GPU is assigned and its driver works. ' + host.stderr[-2000:])
            self.claim(ram)
            command = [sys.executable, '-B', '-m', 'modules.vault_colab', '--probe', str(ram), str(os.getpid()), str(outside)]
            if check_cuda:
                command.append('--check-cuda')
            result = subprocess.run(self.command(ram, command), capture_output=True, text=True, timeout=30,
                                    check=False, env=env, **self.subprocess_options())
            if result.returncode or outside.read_text() != 'unchanged':
                raise VaultError('Colab filesystem isolation failed its startup test. No vault was unlocked. '
                                 'The runtime must support the sandbox and, when requested, CUDA inside it. '
                                 + result.stderr[-2000:])
            report = json.loads(result.stdout)
            if report.get('isolation') != 'verified' or (check_cuda and not report.get('cuda')):
                raise VaultError('Colab startup probe returned an incomplete report.')
            cuda_info = report.get('cuda')
        except (OSError, subprocess.TimeoutExpired) as error:
            raise VaultError('Unable to start the Colab filesystem sandbox. No vault was unlocked.') from error
        finally:
            shutil.rmtree(ram)
            if outside is not None:
                outside.unlink(missing_ok=True)
        return cuda_info


def probe_thread_names():
    """Exercise the exact procfs operation used by NVIDIA, also on CPU-only CI."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import get_native_id

    def rename():
        path = Path(f'/proc/self/task/{get_native_id()}/comm')
        original = path.read_text().rstrip('\n')
        try:
            path.write_text('textgen-cuda')
            if path.read_text().strip() != 'textgen-cuda':
                raise VaultError('Cannot rename CUDA worker threads through procfs.')
        finally:
            path.write_text(original)

    rename()
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(rename).result()


def probe(workspace, parent, outside, check_cuda=False):
    verify_sandbox(workspace, parent)
    probe_thread_names()
    (Path(workspace) / 'volatile/allowed').write_text('RAM only')
    # Check both direct paths and procfs magic links. An fd opened through
    # the read-only mount must not be reopened for writing through procfs.
    outside_fd = os.open(outside, os.O_RDONLY | os.O_CLOEXEC)
    try:
        paths = (outside, f'/proc/self/root{outside}', f'/proc/self/fd/{outside_fd}',
                 f'/proc/{parent}/root{outside}', f'/proc/{os.getppid()}/root{outside}')
        for path in paths:
            try:
                Path(path).write_text('forbidden')
            except OSError:
                pass
            else:
                raise AssertionError('Unconfined write succeeded')
    finally:
        os.close(outside_fd)
    # Process names are mutable kernel memory; unrelated host process metadata
    # must remain protected by the dedicated UID as well.
    for pid in {parent, os.getppid()}:
        try:
            Path(f'/proc/{pid}/comm').write_text('forbidden')
        except OSError:
            pass
        else:
            raise AssertionError('Unconfined process metadata write succeeded')
    code = ('import ctypes, errno\nfrom pathlib import Path\n'
            'from modules.vault_colab import probe_thread_names\nprobe_thread_names()\n'
            'libc = ctypes.CDLL(None, use_errno=True)\n'
            'assert libc.unshare(0x10000000) == -1 and ctypes.get_errno() == errno.EPERM\n'
            'try:\n    Path(' + repr(str(outside)) + ').write_text("forbidden")\n'
            'except OSError:\n    print("isolated")\n'
            'else:\n    raise AssertionError("Child write succeeded")\n')
    child = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, check=False)
    if child.returncode or child.stdout.strip() != 'isolated':
        raise AssertionError('Child did not inherit isolation')
    cuda_info = None
    if check_cuda:
        from modules.vault_cuda import probe_cuda
        cuda_info = probe_cuda()
    print(json.dumps({'isolation': 'verified', 'cuda': cuda_info}))


def main():
    parser = argparse.ArgumentParser(description='Verify Colab vault isolation without opening a vault')
    parser.add_argument('--probe', nargs=3, metavar=('RAM', 'PARENT', 'OUTSIDE'))
    parser.add_argument('--check-cuda', action='store_true', help='Require CUDA initialization and a small VRAM allocation inside the sandbox.')
    args = parser.parse_args()
    disable_core_dumps()
    if args.probe:
        ram, parent, outside = args.probe
        probe(ram, int(parent), outside, check_cuda=args.check_cuda)
    else:
        isolation = ColabIsolation()
        try:
            cuda_info = isolation.preflight(check_cuda=args.check_cuda)
        finally:
            isolation.close()
        print(json.dumps({'isolation': 'bubblewrap', 'verified': True, 'cuda': cuda_info}))


if __name__ == '__main__':
    main()
