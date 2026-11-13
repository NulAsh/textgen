import os
import struct
from enum import IntEnum
from pathlib import Path


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_simple_value_packing = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
    GGUFValueType.BOOL: "?",
}

value_type_info = {
    GGUFValueType.UINT8: 1,
    GGUFValueType.INT8: 1,
    GGUFValueType.UINT16: 2,
    GGUFValueType.INT16: 2,
    GGUFValueType.UINT32: 4,
    GGUFValueType.INT32: 4,
    GGUFValueType.FLOAT32: 4,
    GGUFValueType.UINT64: 8,
    GGUFValueType.INT64: 8,
    GGUFValueType.FLOAT64: 8,
    GGUFValueType.BOOL: 1,
}


class GGUFError(ValueError):
    """An invalid or unsupported model, safe to describe in the interface."""


# Metadata is small compared with tensor data. Bound both encoded bytes and
# Python objects, even when a corrupt length would fit inside a multi-GB file.
_MAX_METADATA_BYTES = 128 * 1024 * 1024
_MAX_METADATA_ITEMS = 2_000_000
_MAX_KEY_BYTES = 65535


def validate_header(header):
    """Validate the first 24 bytes without trusting any file-provided lengths."""
    if header[:4] != b'GGUF':
        prefix = header.lstrip().lower()
        if prefix.startswith((b'<!doctype', b'<html', b'<head', b'<body')):
            detail = 'This is an HTML page, not model weights. Use the file download link (on Hugging Face: /resolve/, not /blob/).'
        elif prefix.startswith(b'version https://git-lfs'):
            detail = 'This is a Git LFS pointer, not model weights. Download the actual GGUF file.'
        else:
            detail = 'The GGUF signature is missing. Download the actual GGUF weights again; renaming another file does not convert it.'
        raise GGUFError(detail)
    if len(header) < 24:
        raise GGUFError('The GGUF header is truncated. Download the model again.')
    version = struct.unpack_from('<I', header, 4)[0]
    if version == 1:
        raise GGUFError('GGUF version 1 is outdated. Download a version 2 or 3 model.')
    if version not in (2, 3):
        if struct.unpack_from('>I', header, 4)[0] in (2, 3):
            raise GGUFError('Big-endian GGUF is not supported by this loader. Download a little-endian GGUF model.')
        raise GGUFError(f'Unsupported GGUF version {version}. This loader supports versions 2 and 3.')


class _MetadataReader:
    def __init__(self, file):
        self.file = file
        self.size = os.fstat(file.fileno()).st_size
        self.items_left = _MAX_METADATA_ITEMS

    def check_bytes(self, length):
        position = self.file.tell()
        if length > self.size - position:
            raise GGUFError('Truncated or corrupt GGUF metadata: a length exceeds the remaining file. Download the model again.')
        if length > _MAX_METADATA_BYTES - position:
            raise GGUFError('GGUF metadata exceeds the 128 MiB safety limit. Check the model file and its source.')

    def read(self, length):
        self.check_bytes(length)
        value = self.file.read(length)
        if len(value) != length:
            raise GGUFError('The GGUF file changed or was truncated while reading. Finish downloading before loading it.')
        return value

    def unpack(self, packing):
        return struct.unpack(packing, self.read(struct.calcsize(packing)))[0]

    def count(self, length, minimum_bytes):
        self.check_bytes(length * minimum_bytes)
        if length > self.items_left:
            raise GGUFError('GGUF metadata exceeds the 2,000,000-item safety limit. Check the model file and its source.')
        self.items_left -= length

    def value_type(self):
        value = self.unpack('<I')
        try:
            return GGUFValueType(value)
        except ValueError:
            raise GGUFError(f'Invalid GGUF metadata value type {value}. The model may be corrupt or unsupported.') from None

    def single(self, value_type):
        if value_type == GGUFValueType.STRING:
            value = self.read(self.unpack('<Q'))
            try:
                return value.decode('utf-8')
            except UnicodeDecodeError:
                # Some tokenizers store raw byte tokens; preserve the old API.
                return value
        return self.unpack(_simple_value_packing[value_type])


def load_metadata(fname):
    metadata = {}
    with open(fname, 'rb') as file:
        try:
            header = file.read(24)
            validate_header(header)
            kv_data_count = struct.unpack_from('<Q', header, 16)[0]
            reader = _MetadataReader(file)
            # Even an empty key and the smallest scalar need 8 + 4 + 1 bytes.
            reader.count(kv_data_count, 13)
            for _ in range(kv_data_count):
                key_length = reader.unpack('<Q')
                if key_length > _MAX_KEY_BYTES:
                    raise GGUFError('Invalid GGUF metadata key length (maximum 65535 bytes). The file is corrupt or is not model weights.')
                try:
                    key = reader.read(key_length).decode('utf-8')
                except UnicodeDecodeError:
                    raise GGUFError('Invalid GGUF metadata key encoding. Download the model again.') from None
                value_type = reader.value_type()
                if value_type == GGUFValueType.ARRAY:
                    element_type = reader.value_type()
                    if element_type == GGUFValueType.ARRAY:
                        raise GGUFError('Nested GGUF metadata arrays are not supported by this loader.')
                    length = reader.unpack('<Q')
                    reader.count(length, 8 if element_type == GGUFValueType.STRING else value_type_info[element_type])
                    metadata[key] = [reader.single(element_type) for _ in range(length)]
                else:
                    metadata[key] = reader.single(value_type)
        except GGUFError as error:
            raise GGUFError(f'{Path(fname).name}: {error}') from None

    return metadata
