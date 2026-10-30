"""Unit tests for ideal-function selection."""
import unittest

import pandas as pd

from assignment_main import TrainingDataHandler


class TestSelection(unittest.TestCase):
    """Tests for TrainingDataHandler.select_best_functions."""

    def test_picks_function_with_smallest_sse(self):
        """The ideal function that equals the training data must be chosen."""
        handler = TrainingDataHandler(db_manager=None)
        handler.train_df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y1": [2.0, 4.0, 6.0],
        })
        handler.ideal_df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y1": [3.0, 6.0, 9.0],
            "y2": [2.0, 4.0, 6.0],
            "y3": [-1.0, -2.0, -3.0],
        })

        result = handler.select_best_functions(["y1"])

        self.assertEqual(result["y1"], "y2")


    def test_max_deviation_is_largest_training_gap(self):
        """D_max must be the largest absolute gap between training and ideal."""
        handler = TrainingDataHandler(db_manager=None)
        handler.train_df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y1": [2.0, 4.0, 6.0],
        })
        handler.ideal_df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y2": [2.0, 4.0, 9.0],
        })

        handler.select_best_functions(["y1"])

        self.assertEqual(handler.max_deviations["y2"], 3.0)


if __name__ == "__main__":
    unittest.main()