"""
tal.io
======

Functions for reading and writing files, and converting between different file formats.

For example, to read a capture data file:

>>> from tal.io import read_capture
>>> capture_data = read_capture('data/Z.hdf5')
"""

from tal.io.capture_data import NLOSCaptureData
from tal.enums import FileFormat


def read_capture(filename: str, file_format: FileFormat = FileFormat.AUTODETECT) -> NLOSCaptureData:
    """
    Read a NLOS capture data located at path described by filename.

    See tal.enums.FileFormat for supported file formats.

    See tal.io.NLOSCaptureData for information about the data structure.
    """
    from tal.io.capture_data import NLOSCaptureData
    return NLOSCaptureData(filename, file_format)


def write_capture(capture_data: NLOSCaptureData, filename: str, file_format: FileFormat = FileFormat.HDF5_TAL):
    """
    Write a NLOS capture data to a file located at path described by filename.

    By default it is written in the TAL HDF5 format, see tal.enums.FileFormat for supported file formats.
    """
    from tal.io.capture_data import write_hdf5
    from tal.io.format import convert_dict
    write_hdf5(convert_dict(capture_data.as_dict(),
               format_to=file_format), filename)
