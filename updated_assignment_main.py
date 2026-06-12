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


# import os
# import pandas as pd
# import numpy as np
# from bokeh.plotting import figure, show

# # =================== Helper function ===================
# def safe_load_csv(path):
#     """Load CSV safely, check file exists, and strip column names."""
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"File not found: {path}")
#     df = pd.read_csv(path)
#     df.columns = df.columns.str.strip()
#     print(f"✅ Loaded '{path}' with columns: {df.columns.tolist()}")
#     print(df.head())
#     return df

# # =================== Custom Exception ===================
# class DataMismatchError(Exception):
#     """Raised when test data x-values do not match ideal function x-values."""
#     pass

# # =================== Training Data Handler ===================
# class TrainingDataHandler:
#     """Load training and ideal data, select best ideal functions."""
#     def __init__(self, train_file, ideal_file):
#         self.train_file = train_file
#         self.ideal_file = ideal_file
#         self.train_df = None
#         self.ideal_df = None
#         self.best_functions = {}

#     def load_all_data(self):
#         self.train_df = safe_load_csv(self.train_file)
#         self.ideal_df = safe_load_csv(self.ideal_file)

#     def select_best_functions(self, train_y_cols):
#         """Select best ideal function for each training y-column."""
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

# # =================== Test Data Mapper ===================
# class TestDataMapper:
#     """Map test data to chosen ideal functions with deviation."""
#     def __init__(self, test_file, ideal_df, chosen_functions):
#         self.test_file = test_file
#         self.ideal_df = ideal_df
#         self.chosen_functions = chosen_functions
#         self.mapped_df = None

#     def map_test_data(self):
#         test_df = safe_load_csv(self.test_file)

#         # Determine the y-column dynamically if it's not named 'y'
#         y_cols = [col for col in test_df.columns if col != 'x']
#         if not y_cols:
#             raise ValueError("No y-column found in test data")
#         y_col = y_cols[0]

#         mapped_data = []
#         for _, row in test_df.iterrows():
#             x_test, y_test = row['x'], row[y_col]
#             ideal_row = self.ideal_df[self.ideal_df['x'] == x_test]
#             if ideal_row.empty:
#                 continue
#             deviations = {fun: abs(y_test - ideal_row[fun].values[0]) for fun in self.chosen_functions}
#             best_func = min(deviations, key=deviations.get)
#             mapped_data.append({
#                 'x': x_test,
#                 'y_test': y_test,
#                 'matched_function': best_func,
#                 'deviation': deviations[best_func]
#             })
#         self.mapped_df = pd.DataFrame(mapped_data)
#         return self.mapped_df

# # =================== Data Visualizer ===================
# class DataVisualizer:
#     """Visualize training, ideal, and test data."""
#     def __init__(self, train_df, ideal_df, mapped_df, chosen_functions):
#         self.train_df = train_df
#         self.ideal_df = ideal_df
#         self.mapped_df = mapped_df
#         self.chosen_functions = chosen_functions

#     def plot_data(self):
#         p = figure(title="Training & Ideal functions with Test data",
#                    x_axis_label='x', y_axis_label='y')
        
#         # Plot training data
#         for func in self.train_df.columns[1:5]:  # first 4 y-columns
#             p.circle(self.train_df['x'], self.train_df[func],
#                      legend_label=f"Train {func}", size=5)
        
#         colors = ['red', 'blue', 'green', 'orange']
#         # Plot ideal functions
#         for func, color in zip(self.chosen_functions, colors):
#             p.line(self.ideal_df['x'], self.ideal_df[func],
#                    legend_label=f"Ideal {func}", line_width=2, color=color)
#             # Plot mapped test points
#             subset = self.mapped_df[self.mapped_df['matched_function'] == func]
#             p.triangle(subset['x'], subset['y_test'], size=8, color=color,
#                        legend_label=f"Test → {func}")
#         show(p)

# # =================== Main Program ===================
# if __name__ == "__main__":
#     # Full paths for CSV files
#     # train_file = r"C:\\Users\\sahil\\OneDrive\\code\\Dataset2\\Dataset2\\train.csv"
#     # ideal_file = r"C:\\Users\\sahil\\OneDrive\\code\\Dataset2\\Dataset2\\ideal.csv"
#     # test_file = r"C:\\Users\\sahil\\OneDrive\\code\\Dataset2\\Dataset2\\test.csv"

#     train_file = "data/train.csv"
#     ideal_file = "data/ideal.csv"
#     test_file  = "data/test.csv"


#     # Step 1: Load training and ideal data
#     train_handler = TrainingDataHandler(train_file, ideal_file)
#     train_handler.load_all_data()

#     # Step 2: Select best ideal functions
#     best_funcs = train_handler.select_best_functions(['y1', 'y2', 'y3', 'y4'])
#     print("Best functions:", best_funcs)

#     # Step 3: Map test data
#     mapper = TestDataMapper(test_file, train_handler.ideal_df, list(best_funcs.values()))
#     mapped_df = mapper.map_test_data()
#     mapped_df.to_csv("output_temp.csv", index=False)
#     print("✅ Mapping complete! File saved as 'output_temp.csv'")

#     # Step 4: Visualize
#     visualizer = DataVisualizer(train_handler.train_df, train_handler.ideal_df, mapped_df, list(best_funcs.values()))
#     visualizer.plot_data()


import os
import pandas as pd
import numpy as np
def safe_load_csv(path):
    return pd.read_csv(path)
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base, sessionmaker
from bokeh.plotting import figure, show

Base = declarative_base()

class TrainingData(Base):
    __tablename__ = 'training_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    y3 = Column(Float, nullable=False)
    y4 = Column(Float, nullable=False)

class IdealFunctions(Base):
    __tablename__ = 'ideal_functions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False)
    # y1 to y50 will be dynamically added below

# Dynamically add y1 to y50 columns to IdealFunctions table
for i in range(1, 51):
    setattr(IdealFunctions, f'y{i}', Column(Float, nullable=False))

class TestData(Base):
    __tablename__ = 'test_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)


class MappedTestData(Base):
    __tablename__ = 'mapped_test_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    matched_ideal_function = Column(String, nullable=True)
    deviation = Column(Float, nullable=True)


class DatabaseManager:
    """Handles SQLite database operations using SQLAlchemy."""

    def __init__(self, db_path):
        self.db_url = f"sqlite:///{db_path}"
        self.db_url = f"sqlite:///{db_path}"
        self.engine = create_engine(self.db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        """Creates tables in the database (drops old ones if they exist)."""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    def insert_dataframe(self, df, table_name):
        """Inserts a pandas DataFrame into a specific SQL table."""
        df.to_sql(table_name, con=self.engine, if_exists='append', index=False)

    def load_table_as_df(self, table_name):
        """Loads an SQL table into a pandas DataFrame."""
        df = pd.read_sql_table(table_name, con=self.engine)
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        return df
    


class TrainingDataHandler:
    """Load training and ideal data, select best ideal functions, and compute max training deviations."""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.train_df = None
        self.ideal_df = None
        self.best_functions = {}  # { train_y_col: ideal_y_col }
        self.max_deviations = {}   # { ideal_y_col: max_deviation }

    def load_data_from_db(self):
        """Loads data directly from the SQLite database tables."""
        self.train_df = self.db_manager.load_table_as_df('training_data')
        self.ideal_df = self.db_manager.load_table_as_df('ideal_functions')

    def select_best_functions(self, train_y_cols):
        """Selects ideal functions using MSE and calculates their max deviation (D_max)."""
        for y_col in train_y_cols:
            min_mse = float('inf')
            best_func = None
            
            for ideal_col in self.ideal_df.columns[1:]:  # skip 'x'
                train_sub = self.train_df[['x', y_col]].rename(columns={y_col: 'train_y'})
                ideal_sub = self.ideal_df[['x', ideal_col]].rename(columns={ideal_col: 'ideal_y'})
                merged = pd.merge(train_sub, ideal_sub, on='x', how='inner')
                
                mse = ((merged['train_y'] - merged['ideal_y']) ** 2).mean()
                if mse < min_mse:
                    min_mse = mse
                    best_func = ideal_col
            
            self.best_functions[y_col] = best_func
            
            # Compute and store maximum deviation (D_max) for the chosen function
            train_sub = self.train_df[['x', y_col]].rename(columns={y_col: 'train_y'})
            ideal_sub = self.ideal_df[['x', best_func]].rename(columns={best_func: 'ideal_y'})
            merged = pd.merge(train_sub, ideal_sub, on='x', how='inner')
            max_dev = (merged['train_y'] - merged['ideal_y']).abs().max()
            self.max_deviations[best_func] = max_dev
            
        return self.best_functions
    
    

class TestDataMapper:
    """Map test data to chosen ideal functions using the sqrt(2) deviation threshold constraint."""

    def __init__(self, db_manager, ideal_df, chosen_functions, max_deviations):
        self.db_manager = db_manager
        self.ideal_df = ideal_df
        self.chosen_functions = chosen_functions
        self.max_deviations = max_deviations
        self.mapped_df = None
        
    def map_test_data(self):
        test_df = self.db_manager.load_table_as_df('test_data')
        y_cols = [col for col in test_df.columns if col != 'x']
        y_col = y_cols[0]
        
        mapped_data = []
        for _, row in test_df.iterrows():
            x_test, y_test = row['x'], row[y_col]
            ideal_row = self.ideal_df[self.ideal_df['x'] == x_test]
            
            if ideal_row.empty:
                mapped_data.append({'x': x_test, 'y': y_test, 'matched_ideal_function': None, 'deviation': None})
                continue
            
            best_match = None
            min_deviation = float('inf')
            
            # Check deviation against each selected ideal function
            for func in self.chosen_functions:
                ideal_y = ideal_row[func].values[0]
                dev = abs(y_test - ideal_y)
                
                # Check threshold constraint: deviation <= D_max * sqrt(2)
                threshold = self.max_deviations[func] * np.sqrt(2)
                if dev <= threshold:
                    if dev < min_deviation:
                        min_deviation = dev
                        best_match = func
            
            mapped_data.append({
                'x': x_test,
                'y': y_test,
                'matched_ideal_function': best_match,
                'deviation': min_deviation if best_match else None
            })
        
        self.mapped_df = pd.DataFrame(mapped_data)
        # Store results directly in the database
        self.db_manager.insert_dataframe(self.mapped_df, 'mapped_test_data')
        return self.mapped_df
    

class DataVisualizer:
    def __init__(self, train_df, ideal_df, test_df, mapped_df, best_funcs):
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.test_df = test_df
        self.mapped_df = mapped_df
        self.best_funcs = best_funcs

    def plot(self):
        p = figure(title="Training vs Ideal vs Test Mapping",
                   x_axis_label='X',
                   y_axis_label='Y')

        # -------------------
        # Training data
        # -------------------
        for y_col in ['y1', 'y2', 'y3', 'y4']:
            p.circle(self.train_df['x'], self.train_df[y_col],
                     legend_label=f"Train {y_col}",
                     size=4, alpha=0.6)

        # -------------------
        # Ideal functions (selected only)
        # -------------------
        for train_y, ideal_y in self.best_funcs.items():
            p.line(self.ideal_df['x'], self.ideal_df[ideal_y],
                   legend_label=f"Ideal {ideal_y}",
                   line_width=2)

        # -------------------
        # Test data (mapped points)
        # -------------------
        mapped_valid = self.mapped_df[self.mapped_df['matched_ideal_function'].notna()]

        p.triangle(mapped_valid['x'], mapped_valid['y'],
                   size=8, color="red",
                   legend_label="Mapped Test Points")

        p.legend.click_policy = "hide"
        show(p)

    

if __name__ == "__main__":
    train_csv_path = r"C:\Users\sahil\OneDrive\code\Dataset2\Dataset2\train.csv"
    ideal_csv_path = r"C:\Users\sahil\OneDrive\code\Dataset2\Dataset2\ideal.csv"
    test_csv_path = r"C:\Users\sahil\OneDrive\code\Dataset2\Dataset2\test.csv"
    db_file_path = r"C:\Users\sahil\.gemini\antigravity\scratch\assignment.db"

    db_mgr = DatabaseManager(db_file_path)
    db_mgr.create_tables()

    train_df = safe_load_csv(train_csv_path)
    ideal_df = safe_load_csv(ideal_csv_path)
    test_df = safe_load_csv(test_csv_path)

    db_mgr.insert_dataframe(train_df, 'training_data')
    db_mgr.insert_dataframe(ideal_df, 'ideal_functions')
    db_mgr.insert_dataframe(test_df, 'test_data')

    train_handler = TrainingDataHandler(db_mgr)
    train_handler.load_data_from_db()

    best_funcs = train_handler.select_best_functions(['y1', 'y2', 'y3', 'y4'])
    print("Best functions selected:", best_funcs)

    mapper = TestDataMapper(
        db_mgr,
        train_handler.ideal_df,
        list(best_funcs.values()),
        train_handler.max_deviations
    )

    mapped_df = mapper.map_test_data()
    print("Mapping completed! All data written to SQLite database.")
    
    # 5. Visualize
    # (Optional: call your DataVisualizer plotting logic here)

    visualizer = DataVisualizer(
        train_handler.train_df,
        train_handler.ideal_df,
        test_df,
        mapped_df,
        best_funcs
    )

    visualizer.plot()