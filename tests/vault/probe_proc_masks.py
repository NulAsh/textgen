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
from modules.vault_colab import verify_mounts, probe_thread_names

ram = Path(sys.argv[1])
verify_mounts(ram)
probe_thread_names()
assert not os.statvfs('/proc').f_flag & os.ST_RDONLY
for name in sys.argv[2:]:
    path = Path(name)
    assert os.statvfs(path).f_flag & os.ST_RDONLY, name
    target = path / 'forbidden' if path.is_dir() else path
    try:
        target.write_text('forbidden')
    except OSError:
        pass
    else:
        raise AssertionError('Inherited proc mount allowed a write: ' + name)
(ram / 'allowed').write_text('synthetic RAM content')
print('proc masks protected; thread names writable')
'''

ram = Path(sys.argv[1])
fd = create_syscall_filter()
try:
    command = ColabIsolation.command(SimpleNamespace(binary=shutil.which('bwrap'), filter_fd=fd), ram,
                                     [sys.executable, '-B', '-c', probe, *sys.argv[1:]])
    result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False, pass_fds=(fd,))
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    raise SystemExit(result.returncode)
finally:
    os.close(fd)
