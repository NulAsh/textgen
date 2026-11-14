"""Small CUDA driver/VRAM probe; contains no model, vault or user content.

Run in a short-lived subprocess before unlock. nvidia-smi alone uses NVML and
does not prove that CUDA can initialize or allocate memory inside the sandbox.
"""

import ctypes as C
import json
import sys

from modules.vault_store import VaultError


def probe_cuda():
    try:
        cuda = C.CDLL('libcuda.so.1')
    except OSError as error:
        raise VaultError(f'Cannot load the NVIDIA CUDA driver: {error}') from error

    signatures = {
        'cuInit': [C.c_uint],
        'cuDeviceGetCount': [C.POINTER(C.c_int)],
        'cuDeviceGet': [C.POINTER(C.c_int), C.c_int],
        'cuDeviceGetName': [C.c_void_p, C.c_int, C.c_int],
        'cuDevicePrimaryCtxRetain': [C.POINTER(C.c_void_p), C.c_int],
        'cuDevicePrimaryCtxRelease_v2': [C.c_int],
        'cuCtxSetCurrent': [C.c_void_p],
        'cuMemGetInfo_v2': [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)],
        'cuMemAlloc_v2': [C.POINTER(C.c_uint64), C.c_size_t],
        'cuMemsetD8_v2': [C.c_uint64, C.c_ubyte, C.c_size_t],
        'cuMemcpyDtoH_v2': [C.c_void_p, C.c_uint64, C.c_size_t],
        'cuMemFree_v2': [C.c_uint64],
        'cuGetErrorName': [C.c_int, C.POINTER(C.c_char_p)],
        'cuGetErrorString': [C.c_int, C.POINTER(C.c_char_p)],
    }
    try:
        for name, signature in signatures.items():
            function = getattr(cuda, name)
            function.argtypes = signature
            function.restype = C.c_int
    except AttributeError as error:
        raise VaultError('The NVIDIA driver is missing a required CUDA API.') from error

    def call(name, *args):
        result = getattr(cuda, name)(*args)
        if result:
            label, detail = C.c_char_p(), C.c_char_p()
            cuda.cuGetErrorName(result, C.byref(label))
            cuda.cuGetErrorString(result, C.byref(detail))
            label = label.value.decode(errors='replace') if label.value else 'CUDA error'
            detail = detail.value.decode(errors='replace') if detail.value else 'unknown error'
            raise VaultError(f'{name}: {label} ({result}): {detail}')

    call('cuInit', 0)
    count = C.c_int()
    call('cuDeviceGetCount', C.byref(count))
    if count.value < 1:
        raise VaultError('CUDA reports no GPU. Select a GPU runtime and check that Colab has assigned one.')

    devices = []
    for ordinal in range(count.value):
        device, context = C.c_int(), C.c_void_p()
        call('cuDeviceGet', C.byref(device), ordinal)
        name = C.create_string_buffer(256)
        call('cuDeviceGetName', name, len(name), device)
        call('cuDevicePrimaryCtxRetain', C.byref(context), device)
        try:
            call('cuCtxSetCurrent', context)
            free, total = C.c_size_t(), C.c_size_t()
            call('cuMemGetInfo_v2', C.byref(free), C.byref(total))
            pointer = C.c_uint64()
            call('cuMemAlloc_v2', C.byref(pointer), 4096)
            try:
                call('cuMemsetD8_v2', pointer, 0x5A, 4096)
                host = C.create_string_buffer(4096)
                call('cuMemcpyDtoH_v2', host, pointer, len(host))
                if host.raw != b'\x5a' * len(host):
                    raise VaultError('CUDA memory round-trip verification failed.')
            finally:
                call('cuMemFree_v2', pointer)
            devices.append({'device': ordinal, 'name': name.value.decode(errors='replace'),
                            'total_mib': total.value // (1024 * 1024),
                            'free_mib': free.value // (1024 * 1024), 'vram_check': 'passed'})
        finally:
            # The probe owns this process/context; always release its VRAM.
            try:
                call('cuCtxSetCurrent', None)
            finally:
                call('cuDevicePrimaryCtxRelease_v2', device)
    return devices


def main():
    try:
        print(json.dumps(probe_cuda()))
    except VaultError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
