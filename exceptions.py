"""Custom exceptions for the ideal-function assignment."""


class AssignmentError(Exception):
    """Base class for all errors raised by this project."""


class DataLoadError(AssignmentError):
    """Raised when a data file is missing or cannot be read."""


class InvalidDataError(AssignmentError):
    """Raised when loaded data does not have the expected structure."""