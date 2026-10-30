# """Unit tests for the data loader."""
# import os
# import tempfile
# import unittest

# from data_loader import CsvLoader
# from exceptions import DataLoadError, InvalidDataError


# class TestCsvLoader(unittest.TestCase):
#     """Tests for CsvLoader and its error handling."""

#     def test_missing_file_raises_dataloaderror(self):
#         """A file that does not exist should raise DataLoadError."""
#         loader = CsvLoader("does_not_exist.csv", ["x", "y"])
#         with self.assertRaises(DataLoadError):
#             loader.load()


# if __name__ == "__main__":
#     unittest.main()

#     def test_wrong_columns_raises_invalidatadataerror(self):
#         """A CSV with the wrong column names should raise InvalidDataError."""
#         with tempfile.TemporaryDirectory() as folder:
#             path = os.path.join(folder, "bad.csv")
#             with open(path, "w") as f:
#                 f.write("a,b\n1,2\n")
#             loader = CsvLoader(path, ["x", "y"])
#             with self.assertRaises(InvalidDataError):
#                 loader.load()

"""Unit tests for the data loader."""
import os
import tempfile
import unittest

from data_loader import CsvLoader
from exceptions import DataLoadError, InvalidDataError


class TestCsvLoader(unittest.TestCase):
    """Tests for CsvLoader and its error handling."""

    def test_missing_file_raises_dataloaderror(self):
        """A file that does not exist should raise DataLoadError."""
        loader = CsvLoader("does_not_exist.csv", ["x", "y"])
        with self.assertRaises(DataLoadError):
            loader.load()

    def test_wrong_columns_raises_invaliddataerror(self):
        """A CSV with the wrong column names should raise InvalidDataError."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bad.csv")
            with open(path, "w") as f:
                f.write("a,b\n1,2\n")
            loader = CsvLoader(path, ["x", "y"])
            with self.assertRaises(InvalidDataError):
                loader.load()


if __name__ == "__main__":
    unittest.main()