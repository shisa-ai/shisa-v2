"""Monkeypatch Megatron-LM checkpoint writing to handle PyTorch 2.5 signature changes."""

import os
import sys

# Ensure Megatron-LM sources are importable when running from /workspace/project
_MEGA_PATH = os.environ.get("MEGATRON_LM_PATH", "/workspace/Megatron-LM")
if _MEGA_PATH not in sys.path:
    sys.path.insert(0, _MEGA_PATH)

import functools

from torch.distributed.checkpoint import filesystem as torch_fs
from torch.distributed.checkpoint.filesystem import SerializationFormat
import megatron.core.dist_checkpointing.strategies.filesystem_async as fs_async


def _wrap_write_item(func):
    """Wrap _write_item to inject SerializationFormat when PyTorch requires it."""
    if getattr(func, "_serialization_format_patched", False):
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as exc:
            if "serialization_format" not in str(exc):
                raise
            if "serialization_format" in kwargs:
                raise
            patched_kwargs = dict(kwargs)
            patched_kwargs["serialization_format"] = SerializationFormat.TORCH_SAVE
            return func(*args, **patched_kwargs)

    wrapper._serialization_format_patched = True
    return wrapper


# Patch both the base torch function and Megatron's module-level alias/partial.
torch_fs._write_item = _wrap_write_item(torch_fs._write_item)
fs_async._write_item = _wrap_write_item(fs_async._write_item)
