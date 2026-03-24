"""Compatibility wrapper for the integrated annotated Mimic HDF5 recording service."""

from .core.annotated_record_service import record_dataset
from .core.annotated_recording import AnnotatedMimicHDF5Recorder

__all__ = ["AnnotatedMimicHDF5Recorder", "record_dataset"]
