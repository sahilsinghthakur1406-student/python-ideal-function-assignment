"""Unit tests for the database manager."""
import os
import tempfile
import unittest

import pandas as pd

from assignment_main import DatabaseManager


class TestDatabase(unittest.TestCase):
    """Tests for DatabaseManager."""

    def test_saved_data_can_be_loaded_back(self):
        """A DataFrame inserted into a table must be returned unchanged."""
        with tempfile.TemporaryDirectory() as folder:
            db = DatabaseManager(os.path.join(folder, "test.db"))
            db.create_tables()
            data = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
            db.insert_dataframe(data, "test_data")
            result = db.load_table_as_df("test_data")
            pd.testing.assert_frame_equal(result, data)
            db.engine.dispose()


if __name__ == "__main__":
    unittest.main()