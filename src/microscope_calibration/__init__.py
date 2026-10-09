from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("microscope-calibration")
except PackageNotFoundError:
    __version__ = "unknown"
