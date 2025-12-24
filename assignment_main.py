# import pandas as pd
# import numpy as np
# from sqlalchemy import create_engine
# from bokeh.plotting import figure, show


# # Custom exception example
# class DataMismatchError(Exception):
#     """Raised when test data x-values do not match ideal function x-values."""
#     pass

# # Base class for handling CSV data
# class CSVDataHandler:
#     """Load and clean CSV data."""
    
#     def __init__(self, filepath):
#         self.filepath = filepath
#         self.df = None
    
#     def load_data(self):
#         self.df = pd.read_csv(self.filepath)
#         self.df.columns = self.df.columns.str.strip()
#         return self.df

# # Inherited class for training-specific methods
# class TrainingDataHandler(CSVDataHandler):
#     """Process training data and select best ideal functions."""
    
#     def __init__(self, train_file, ideal_file):
#         super().__init__(train_file)
#         self.ideal_file = ideal_file
#         self.train_df = None
#         self.ideal_df = None
#         self.best_functions = {}
    
#     def load_all_data(self):
#         self.train_df = self.load_data()
#         self.ideal_df = pd.read_csv(self.ideal_file)
#         self.ideal_df.columns = self.ideal_df.columns.str.strip()
    
#     def select_best_functions(self, train_y_cols):
#         """Select the best ideal function for each training y-column."""
#         for y_col in train_y_cols:
#             min_error = float('inf')
#             best_func = None
#             for ideal_col in self.ideal_df.columns[1:]:
#                 train_sub = self.train_df[['x', y_col]].rename(columns={y_col: 'train_y'})
#                 ideal_sub = self.ideal_df[['x', ideal_col]].rename(columns={ideal_col: 'ideal_y'})
#                 merged = pd.merge(train_sub, ideal_sub, on='x', how='inner')
#                 mse = ((merged['train_y'] - merged['ideal_y']) ** 2).mean()
#                 if mse < min_error:
#                     min_error = mse
#                     best_func = ideal_col
#             self.best_functions[y_col] = best_func
#         return self.best_functions

# # Class to map test data to chosen ideal functions
# class TestDataMapper:
#     """Map test data points to the best ideal functions with deviation."""
    
#     def __init__(self, test_file, ideal_df, chosen_functions):
#         self.test_file = test_file
#         self.ideal_df = ideal_df
#         self.chosen_functions = chosen_functions
#         self.mapped_df = None
    
#     def map_test_data(self):
#         test_df = pd.read_csv(self.test_file)
#         mapped_data = []
#         for _, row in test_df.iterrows():
#             x_test, y_test = row['x'], row['y']
#             ideal_row = self.ideal_df[self.ideal_df['x'] == x_test]
#             if ideal_row.empty:
#                 continue
#             deviations = {fun: abs(y_test - ideal_row[fun].values[0]) for fun in self.chosen_functions}
#             best_func = min(deviations, key=deviations.get)
#             mapped_data.append({'x': x_test, 'y_test': y_test, 'matched_function': best_func, 'deviation': deviations[best_func]})
#         self.mapped_df = pd.DataFrame(mapped_data)
#         return self.mapped_df

# # Optional: visualization class using Bokeh
# class DataVisualizer:
#     """Visualize training, ideal, and test data."""
    
#     def __init__(self, train_df, ideal_df, mapped_df, chosen_functions):
#         self.train_df = train_df
#         self.ideal_df = ideal_df
#         self.mapped_df = mapped_df
#         self.chosen_functions = chosen_functions
    
#     def plot_data(self):
#         p = figure(title="Training & Ideal functions with Test data", x_axis_label='x', y_axis_label='y')
#         for func in ['y1','y2','y3','y4']:
#             p.circle(self.train_df['x'], self.train_df[func], legend_label=f"Train {func}", size=5)
#         colors = ['red','blue','green','orange']
#         for func, color in zip(self.chosen_functions, colors):
#             p.line(self.ideal_df['x'], self.ideal_df[func], legend_label=f"Ideal {func}", line_width=2, color=color)
#             subset = self.mapped_df[self.mapped_df['matched_function']==func]
#             p.triangle(subset['x'], subset['y_test'], size=8, color=color, legend_label=f"Test→{func}")
#         show(p)

# # ========================= Main Program =========================
# if __name__ == "__main__":
#     train_file = "train.csv"
#     ideal_file = "ideal.csv"
#     test_file = "test.csv"

#     # Step 1: Load training and ideal data
#     train_handler = TrainingDataHandler(train_file, ideal_file)
#     train_handler.load_all_data()
#     best_funcs = train_handler.select_best_functions(['y1','y2','y3','y4'])
#     print("Best functions:", best_funcs)

#     # Step 2: Map test data
#     mapper = TestDataMapper(test_file, train_handler.ideal_df, list(best_funcs.values()))
#     mapped_df = mapper.map_test_data()
#     mapped_df.to_csv("output_temp.csv", index=False)
#     print("✅ Mapping complete! File saved as 'output_temp.csv'")

#     # Step 3: Visualize
#     visualizer = DataVisualizer(train_handler.train_df, train_handler.ideal_df, mapped_df, list(best_funcs.values()))
#     visualizer.plot_data()


import os
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show

# =================== Helper function ===================
def safe_load_csv(path):
    """Load CSV safely, check file exists, and strip column names."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"✅ Loaded '{path}' with columns: {df.columns.tolist()}")
    print(df.head())
    return df

# =================== Custom Exception ===================
class DataMismatchError(Exception):
    """Raised when test data x-values do not match ideal function x-values."""
    pass

# =================== Training Data Handler ===================
class TrainingDataHandler:
    """Load training and ideal data, select best ideal functions."""
    def __init__(self, train_file, ideal_file):
        self.train_file = train_file
        self.ideal_file = ideal_file
        self.train_df = None
        self.ideal_df = None
        self.best_functions = {}

    def load_all_data(self):
        self.train_df = safe_load_csv(self.train_file)
        self.ideal_df = safe_load_csv(self.ideal_file)

    def select_best_functions(self, train_y_cols):
        """Select best ideal function for each training y-column."""
        for y_col in train_y_cols:
            min_error = float('inf')
            best_func = None
            for ideal_col in self.ideal_df.columns[1:]:
                train_sub = self.train_df[['x', y_col]].rename(columns={y_col: 'train_y'})
                ideal_sub = self.ideal_df[['x', ideal_col]].rename(columns={ideal_col: 'ideal_y'})
                merged = pd.merge(train_sub, ideal_sub, on='x', how='inner')
                mse = ((merged['train_y'] - merged['ideal_y']) ** 2).mean()
                if mse < min_error:
                    min_error = mse
                    best_func = ideal_col
            self.best_functions[y_col] = best_func
        return self.best_functions

# =================== Test Data Mapper ===================
class TestDataMapper:
    """Map test data to chosen ideal functions with deviation."""
    def __init__(self, test_file, ideal_df, chosen_functions):
        self.test_file = test_file
        self.ideal_df = ideal_df
        self.chosen_functions = chosen_functions
        self.mapped_df = None

    def map_test_data(self):
        test_df = safe_load_csv(self.test_file)

        # Determine the y-column dynamically if it's not named 'y'
        y_cols = [col for col in test_df.columns if col != 'x']
        if not y_cols:
            raise ValueError("No y-column found in test data")
        y_col = y_cols[0]

        mapped_data = []
        for _, row in test_df.iterrows():
            x_test, y_test = row['x'], row[y_col]
            ideal_row = self.ideal_df[self.ideal_df['x'] == x_test]
            if ideal_row.empty:
                continue
            deviations = {fun: abs(y_test - ideal_row[fun].values[0]) for fun in self.chosen_functions}
            best_func = min(deviations, key=deviations.get)
            mapped_data.append({
                'x': x_test,
                'y_test': y_test,
                'matched_function': best_func,
                'deviation': deviations[best_func]
            })
        self.mapped_df = pd.DataFrame(mapped_data)
        return self.mapped_df

# =================== Data Visualizer ===================
class DataVisualizer:
    """Visualize training, ideal, and test data."""
    def __init__(self, train_df, ideal_df, mapped_df, chosen_functions):
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.mapped_df = mapped_df
        self.chosen_functions = chosen_functions

    def plot_data(self):
        p = figure(title="Training & Ideal functions with Test data",
                   x_axis_label='x', y_axis_label='y')
        
        # Plot training data
        for func in self.train_df.columns[1:5]:  # first 4 y-columns
            p.circle(self.train_df['x'], self.train_df[func],
                     legend_label=f"Train {func}", size=5)
        
        colors = ['red', 'blue', 'green', 'orange']
        # Plot ideal functions
        for func, color in zip(self.chosen_functions, colors):
            p.line(self.ideal_df['x'], self.ideal_df[func],
                   legend_label=f"Ideal {func}", line_width=2, color=color)
            # Plot mapped test points
            subset = self.mapped_df[self.mapped_df['matched_function'] == func]
            p.triangle(subset['x'], subset['y_test'], size=8, color=color,
                       legend_label=f"Test → {func}")
        show(p)

# =================== Main Program ===================
if __name__ == "__main__":
    # Full paths for CSV files
    # train_file = r"C:\\Users\\sahil\\OneDrive\\code\\Dataset2\\Dataset2\\train.csv"
    # ideal_file = r"C:\\Users\\sahil\\OneDrive\\code\\Dataset2\\Dataset2\\ideal.csv"
    # test_file = r"C:\\Users\\sahil\\OneDrive\\code\\Dataset2\\Dataset2\\test.csv"

    train_file = "data/train.csv"
    ideal_file = "data/ideal.csv"
    test_file  = "data/test.csv"


    # Step 1: Load training and ideal data
    train_handler = TrainingDataHandler(train_file, ideal_file)
    train_handler.load_all_data()

    # Step 2: Select best ideal functions
    best_funcs = train_handler.select_best_functions(['y1', 'y2', 'y3', 'y4'])
    print("Best functions:", best_funcs)

    # Step 3: Map test data
    mapper = TestDataMapper(test_file, train_handler.ideal_df, list(best_funcs.values()))
    mapped_df = mapper.map_test_data()
    mapped_df.to_csv("output_temp.csv", index=False)
    print("✅ Mapping complete! File saved as 'output_temp.csv'")

    # Step 4: Visualize
    visualizer = DataVisualizer(train_handler.train_df, train_handler.ideal_df, mapped_df, list(best_funcs.values()))
    visualizer.plot_data()
