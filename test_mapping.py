"""Unit tests for mapping test points to ideal functions."""
import unittest

import pandas as pd

from assignment_main import TestDataMapper


class FakeDatabase:
    """Stand-in for DatabaseManager so the test needs no real database."""

    def __init__(self, test_df):
        """Store the test table that load_table_as_df returns."""
        self.test_df = test_df

    def load_table_as_df(self, table_name):
        """Return the prepared test table."""
        return self.test_df

    def insert_dataframe(self, df, table_name):
        """Pretend to save; do nothing."""


class TestMapping(unittest.TestCase):
    """Tests for TestDataMapper and the sqrt(2) rule."""

    def setUp(self):
        """Prepare an ideal function y2 = 10 everywhere, with D_max = 1."""
        self.ideal_df = pd.DataFrame({
            "x": [1.0, 2.0],
            "y2": [10.0, 10.0],
        })

    def make_mapper(self, test_points):
        """Build a mapper for the given test table."""
        return TestDataMapper(
            FakeDatabase(test_points),
            self.ideal_df,
            ["y2"],
            {"y2": 1.0},
        )

    def test_point_inside_threshold_is_mapped(self):
        """A point 0.5 from the ideal line is within 1 * sqrt(2)."""
        points = pd.DataFrame({"x": [1.0], "y": [10.5]})
        result = self.make_mapper(points).map_test_data()
        self.assertEqual(result.loc[0, "matched_ideal_function"], "y2")

    def test_point_outside_threshold_is_unmapped(self):
        """A point 5 from the ideal line is beyond 1 * sqrt(2)."""
        points = pd.DataFrame({"x": [1.0], "y": [15.0]})
        result = self.make_mapper(points).map_test_data()
        self.assertTrue(pd.isna(result.loc[0, "matched_ideal_function"]))


if __name__ == "__main__":
    unittest.main()