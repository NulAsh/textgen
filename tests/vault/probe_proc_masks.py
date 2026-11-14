"""Enter the production mount layout from a container-style procfs fixture."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.vault_colab import ColabIsolation, create_syscall_filter

probe = '''
import os
import sys
from pathlib import Path
from modules.vault_colab import verify_mounts, verify_sandbox, probe_thread_names

ram = Path(sys.argv[1])
parent = int(sys.argv[2])
if parent:
    verify_sandbox(ram, parent)
else:
    verify_mounts(ram)
probe_thread_names()
assert not os.statvfs('/proc').f_flag & os.ST_RDONLY
for name in sys.argv[3:]:
    path = Path(name)
    assert os.statvfs(path).f_flag & os.ST_RDONLY, name
    target = path / 'forbidden' if path.is_dir() else path
    try:
        target.write_text('forbidden')
    except OSError:
        pass
    else:
        raise AssertionError('Inherited proc mount allowed a write: ' + name)
(ram / 'volatile' / 'allowed').write_text('synthetic RAM content')
print('proc masks protected; thread names writable')
'''

fixture = sys.argv[1] == '--fixture'
if fixture:
    # The caller must have entered a private mount namespace. Never install
    # fixture mounts into the test runner's namespace or change its policy.
    assert os.geteuid() == 0
    assert os.readlink('/proc/self/ns/mnt') != os.readlink(f'/proc/{os.getppid()}/ns/mnt')
    mask, ram_name, outside, directory = sys.argv[2:]
    if mask == 'nested-directory':
        mounts = [(directory, '/proc/tty'), (outside, '/proc/tty/nested file')]
    else:
        mounts = [('/dev/null' if mask == 'null' else outside, '/proc/keys')]
    for source, target in mounts:
        subprocess.run(['mount', '--bind', source, target], check=True)
    paths = [target for _, target in mounts]
else:
    ram_name, *paths = sys.argv[1:]

ram = Path(ram_name)
(ram / 'volatile').mkdir(exist_ok=True)
isolation = ColabIsolation() if fixture else SimpleNamespace(binary=shutil.which('bwrap'), filter_fd=create_syscall_filter())
try:
    if fixture:
        isolation.claim(ram)
    parent = os.getpid() if fixture else 0
    command = ColabIsolation.command(isolation, ram, [sys.executable, '-B', '-c', probe, str(ram), str(parent), *paths])
    options = isolation.subprocess_options() if fixture else {'pass_fds': (isolation.filter_fd,)}
    result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False, **options)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    raise SystemExit(result.returncode)
finally:
    if fixture:
        isolation.close()
    else:
        os.close(isolation.filter_fd)
