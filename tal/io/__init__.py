from tal.io.capture_data import NLOSCaptureData, write_hdf5
from tal.io.format import FileFormat, convert_dict


def read_capture(filename: str, file_format: FileFormat = FileFormat.AUTODETECT) -> NLOSCaptureData:
    return NLOSCaptureData(filename, file_format)


def write_capture(capture_data: NLOSCaptureData, filename: str, file_format: FileFormat = FileFormat.HDF5_TAL):
    write_hdf5(convert_dict(capture_data.as_dict(),
               format_to=file_format), filename)
