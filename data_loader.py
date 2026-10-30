# """Loading and validating input data."""
# from abc import ABC, abstractmethod

# import pandas as pd

# from exceptions import DataLoadError, InvalidDataError


# class BaseDataLoader(ABC):
#     """Base class: defines how data is loaded and validated."""

#     def __init__(self, path, required_columns):
#         self.path = path
#         self.required_columns = required_columns

#     @abstractmethod
#     def _read(self):
#         """Read the raw data. Each subclass implements its own format."""

#     def load(self):
#         """Read the file, convert low-level errors, and validate the result."""
#         try:
#             df = self._read()
#         except FileNotFoundError as err:
#             raise DataLoadError(f"File not found: {self.path}") from err
#         except (pd.errors.EmptyDataError, pd.errors.ParserError) as err:
#             raise DataLoadError(f"Cannot parse file: {self.path}") from err
#         self._validate(df)
#         return df

#     def _validate(self, df):
#         """Check the data is not empty and has the required columns."""
#         if df.empty:
#             raise InvalidDataError(f"No rows found in {self.path}")
#         missing = [c for c in self.required_columns if c not in df.columns]
#         if missing:
#             raise InvalidDataError(f"Missing columns {missing} in {self.path}")


# class CsvLoader(BaseDataLoader):
#     """Loads data from a CSV file."""

#     def _read(self):
#         return pd.read_csv(self.path)


"""Loading and validating input data."""
from abc import ABC, abstractmethod

import pandas as pd

from exceptions import DataLoadError, InvalidDataError


class BaseDataLoader(ABC):
    """Base class: defines how data is loaded and validated."""

    def __init__(self, path, required_columns):
        """Store the file path and the columns that must be present."""
        self.path = path
        self.required_columns = required_columns

    @abstractmethod
    def _read(self):
        """Read the raw data. Each subclass implements its own format."""

    def load(self):
        """Read the file, convert low-level errors, and validate the result."""
        try:
            df = self._read()
        except FileNotFoundError as err:
            raise DataLoadError(f"File not found: {self.path}") from err
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as err:
            raise DataLoadError(f"Cannot parse file: {self.path}") from err
        self._validate(df)
        return df

    def _validate(self, df):
        """Check the data is not empty and has the required columns."""
        if df.empty:
            raise InvalidDataError(f"No rows found in {self.path}")
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise InvalidDataError(f"Missing columns {missing} in {self.path}")


class CsvLoader(BaseDataLoader):
    """Loads data from a CSV file."""

    def _read(self):
        """Read the CSV file into a pandas DataFrame."""
        return pd.read_csv(self.path)