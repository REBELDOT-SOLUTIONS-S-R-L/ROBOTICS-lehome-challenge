"""Compatibility wrapper for the direct HDF5 recording service."""

from .core.record_service import record_dataset
from .core.recording import DirectHDF5Recorder

__all__ = ["DirectHDF5Recorder", "record_dataset"]
