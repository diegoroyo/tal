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


def read_capture(filename: str, file_format: FileFormat = FileFormat.AUTODETECT,
                 skip_H: bool = False) -> NLOSCaptureData:
    """
    Read a NLOS capture data located at path described by filename.

    skip_H: If True, the H matrix is not loaded. This can be useful if you only need the calibration data.

    See tal.enums.FileFormat for supported file formats.

    See tal.io.NLOSCaptureData for information about the data structure.
    """
    from tal.io.capture_data import NLOSCaptureData
    return NLOSCaptureData(filename, file_format=file_format, skip_H=skip_H)


def write_capture(filename: str, capture_data: NLOSCaptureData, file_format: FileFormat = FileFormat.HDF5_TAL,
                  compression_level: int = 0):
    """
    Write a NLOS capture data to a file located at path described by filename.

    By default it is written in the TAL HDF5 format, see tal.enums.FileFormat for supported file formats.

    Files are compressed using gzip. When setting compression_level=0, no compression is used.
    Compression levels 1-9 are supported, where 1 is the fastest and 9 is the slowest but most efficient.
    """
    assert 0 <= compression_level <= 9, 'Compression level must be an integer between 0 and 9'
    from tal.io.capture_data import write_hdf5
    from tal.io.format import convert_dict
    write_hdf5(filename, convert_dict(
        capture_data.as_dict(), format_to=file_format), compression_level=compression_level)
