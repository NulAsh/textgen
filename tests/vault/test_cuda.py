"""Real procfs/ctypes regressions using an explicitly simulated CUDA driver.

CI has no NVIDIA GPU. The stub performs NVIDIA's observed thread-name syscall
and a host-memory round trip; it does NOT claim to test real CUDA hardware.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.vault_colab import ColabIsolation, create_syscall_filter


@pytest.fixture
def simulated_cuda(tmp_path, monkeypatch):
    compiler = shutil.which('cc')
    if compiler is None:
        pytest.skip('A C compiler is needed for the simulated CUDA driver.')
    source = tmp_path / 'driver.c'
    source.write_text(r'''
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

static int fail(const char *stage) {
    const char *value = getenv("TEXTGEN_TEST_CUDA_FAILURE");
    return value && strcmp(value, stage) == 0;
}
int cuInit(unsigned int flags) {
    char path[128];
    snprintf(path, sizeof(path), "/proc/self/task/%ld/comm", syscall(SYS_gettid));
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) return 304;
    int result = write(fd, "cuda-simulated", 14) == 14 ? 0 : 304;
    close(fd);
    return fail("init") ? 304 : result;
}
int cuDeviceGetCount(int *count) { *count = fail("no-gpu") ? 0 : 1; return 0; }
int cuDeviceGet(int *device, int ordinal) { *device = ordinal; return 0; }
int cuDeviceGetName(char *name, int size, int device) {
    snprintf(name, size, "Simulated test GPU"); return 0;
}
int cuDevicePrimaryCtxRetain(void **ctx, int device) { *ctx = (void *)1; return 0; }
int cuDevicePrimaryCtxRelease_v2(int device) { return 0; }
int cuCtxSetCurrent(void *ctx) { return 0; }
int cuMemGetInfo_v2(size_t *free_mem, size_t *total) {
    *total = 1024 * 1024 * 1024; *free_mem = *total / 2; return 0;
}
int cuMemAlloc_v2(uint64_t *ptr, size_t size) {
    if (fail("allocate")) return 2;
    *ptr = (uintptr_t)malloc(size); return *ptr ? 0 : 2;
}
int cuMemsetD8_v2(uint64_t ptr, unsigned char value, size_t size) {
    memset((void *)(uintptr_t)ptr, value, size); return 0;
}
int cuMemcpyDtoH_v2(void *host, uint64_t ptr, size_t size) {
    memcpy(host, (void *)(uintptr_t)ptr, size);
    if (fail("round-trip")) ((char *)host)[0] = 0;
    return 0;
}
int cuMemFree_v2(uint64_t ptr) { free((void *)(uintptr_t)ptr); return 0; }
int cuGetErrorName(int result, const char **name) {
    *name = result == 304 ? "CUDA_ERROR_OPERATING_SYSTEM" : "CUDA_ERROR_OUT_OF_MEMORY"; return 0;
}
int cuGetErrorString(int result, const char **text) { *text = "simulated driver error"; return 0; }
''')
    result = subprocess.run([compiler, '-shared', '-fPIC', str(source), '-o', str(tmp_path / 'libcuda.so.1')],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    for path in (tmp_path, tmp_path.parent, tmp_path.parent.parent):
        path.chmod(0o755)
    monkeypatch.setenv('LD_LIBRARY_PATH', str(tmp_path))
    return tmp_path


@pytest.mark.parametrize('readonly_proc', [True, False])
def test_cuda_thread_name_syscall_inside_real_sandbox(simulated_cuda, readonly_proc):
    binary = shutil.which('bwrap')
    if not binary:
        pytest.skip('Install bubblewrap to test CUDA procfs compatibility.')
    ram = Path(tempfile.mkdtemp(prefix='textgen-cuda-proc-', dir='/dev/shm'))
    fd = create_syscall_filter()
    try:
        command = ColabIsolation.command(SimpleNamespace(binary=binary, filter_fd=fd), ram,
                                         [sys.executable, '-B', '-m', 'modules.vault_cuda'])
        if readonly_proc:
            index = command.index('--')
            command[index:index] = ['--ro-bind', '/proc', '/proc']
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False, pass_fds=(fd,))
        if readonly_proc:
            assert result.returncode != 0
            assert 'cuInit: CUDA_ERROR_OPERATING_SYSTEM (304)' in result.stderr
        else:
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout)[0]['vram_check'] == 'passed'
    finally:
        os.close(fd)
        shutil.rmtree(ram)


@pytest.mark.parametrize(('stage', 'message'), [
    ('init', 'CUDA_ERROR_OPERATING_SYSTEM (304)'),
    ('no-gpu', 'CUDA reports no GPU'),
    ('allocate', 'cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY'),
    ('round-trip', 'CUDA memory round-trip verification failed'),
])
def test_cuda_probe_failure_is_reported_without_cpu_fallback(simulated_cuda, monkeypatch, stage, message):
    monkeypatch.setenv('TEXTGEN_TEST_CUDA_FAILURE', stage)
    result = subprocess.run([sys.executable, '-B', '-m', 'modules.vault_cuda'],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode != 0
    assert message in result.stderr
    assert not result.stdout


@pytest.mark.skipif(os.geteuid() != 0, reason='The full Colab preflight needs a root supervisor.')
def test_cuda_preflight_with_simulated_driver_and_real_worker_identity(simulated_cuda):
    isolation = ColabIsolation()
    try:
        result = isolation.preflight(check_cuda=True)
        assert result == [{'device': 0, 'name': 'Simulated test GPU', 'total_mib': 1024,
                           'free_mib': 512, 'vram_check': 'passed'}]
    finally:
        isolation.close()
