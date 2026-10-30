"""Ideal-function selection and test-data mapping (main program)."""
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from sqlalchemy import Column, Float, String, Table, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from data_loader import CsvLoader

Base = declarative_base()


class TrainingData(Base):
    """Training table: x plus the four training functions (5 columns)."""
    __tablename__ = 'training_data'
    x = Column(Float, primary_key=True)
    y1 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    y3 = Column(Float, nullable=False)
    y4 = Column(Float, nullable=False)


class IdealFunctions(Base):
    """Ideal-function table: x plus y1 to y50 (51 columns)."""
    __tablename__ = 'ideal_functions'
    x = Column(Float, primary_key=True)
    # y1 to y50 are added dynamically below


for i in range(1, 51):
    setattr(IdealFunctions, f'y{i}', Column(Float, nullable=False))

# Test-data table: x and y only, no primary key needed
test_data_table = Table(
    'test_data', Base.metadata,
    Column('x', Float, nullable=False),
    Column('y', Float, nullable=False),
)

# Results table: the four columns of Table 3 in the brief
mapped_test_data_table = Table(
    'mapped_test_data', Base.metadata,
    Column('x', Float, nullable=False),
    Column('y', Float, nullable=False),
    Column('delta_y', Float, nullable=False),
    Column('ideal_func_no', String, nullable=False),
)


class DatabaseManager:
    """Handles SQLite database operations using SQLAlchemy."""

    def __init__(self, db_path):
        """Create the engine and session factory for the given database file."""
        self.db_url = f"sqlite:///{db_path}"
        self.engine = create_engine(self.db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        """Create all tables (drops old ones if they exist)."""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    def insert_dataframe(self, df, table_name):
        """Insert a pandas DataFrame into a specific SQL table."""
        df.to_sql(table_name, con=self.engine, if_exists='append', index=False)

    def load_table_as_df(self, table_name):
        """Load an SQL table into a pandas DataFrame."""
        df = pd.read_sql_table(table_name, con=self.engine)
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        return df


class TrainingDataHandler:
    """Select the best ideal functions and compute their max training deviations."""

    def __init__(self, db_manager):
        """Store the database manager and prepare empty results."""
        self.db_manager = db_manager
        self.train_df = None
        self.ideal_df = None
        self.best_functions = {}  # { train_y_col: ideal_y_col }
        self.max_deviations = {}  # { ideal_y_col: max_deviation }

    def load_data_from_db(self):
        """Load training and ideal data from the SQLite database."""
        self.train_df = self.db_manager.load_table_as_df('training_data')
        self.ideal_df = self.db_manager.load_table_as_df('ideal_functions')

    def select_best_functions(self, train_y_cols):
        """Pick the ideal function with the smallest SSE for each training column."""
        for y_col in train_y_cols:
            min_sse = float('inf')
            best_func = None

            for ideal_col in self.ideal_df.columns[1:]:  # skip 'x'
                train_sub = self.train_df[['x', y_col]].rename(columns={y_col: 'train_y'})
                ideal_sub = self.ideal_df[['x', ideal_col]].rename(columns={ideal_col: 'ideal_y'})
                merged = pd.merge(train_sub, ideal_sub, on='x', how='inner')

                sse = ((merged['train_y'] - merged['ideal_y']) ** 2).sum()
                if sse < min_sse:
                    min_sse = sse
                    best_func = ideal_col

            self.best_functions[y_col] = best_func

            # Maximum deviation (D_max) between training data and the chosen function
            train_sub = self.train_df[['x', y_col]].rename(columns={y_col: 'train_y'})
            ideal_sub = self.ideal_df[['x', best_func]].rename(columns={best_func: 'ideal_y'})
            merged = pd.merge(train_sub, ideal_sub, on='x', how='inner')
            max_dev = (merged['train_y'] - merged['ideal_y']).abs().max()
            self.max_deviations[best_func] = max_dev

        return self.best_functions


class TestDataMapper:
    """Map test data to chosen ideal functions using the sqrt(2) deviation threshold."""

    def __init__(self, db_manager, ideal_df, chosen_functions, max_deviations):
        """Store the database manager, ideal data, chosen functions and D_max values."""
        self.db_manager = db_manager
        self.ideal_df = ideal_df
        self.chosen_functions = chosen_functions
        self.max_deviations = max_deviations
        self.mapped_df = None

    def map_test_data(self):
        """Map every test point to an ideal function if it is within D_max * sqrt(2)."""
        test_df = self.db_manager.load_table_as_df('test_data')
        y_cols = [col for col in test_df.columns if col != 'x']
        y_col = y_cols[0]

        mapped_data = []
        for _, row in test_df.iterrows():
            x_test, y_test = row['x'], row[y_col]
            ideal_row = self.ideal_df[self.ideal_df['x'] == x_test]

            if ideal_row.empty:
                mapped_data.append({'x': x_test, 'y': y_test,
                                    'matched_ideal_function': None, 'deviation': None})
                continue

            best_match = None
            min_deviation = float('inf')

            for func in self.chosen_functions:
                ideal_y = ideal_row[func].values[0]
                dev = abs(y_test - ideal_y)

                # Threshold constraint: deviation <= D_max * sqrt(2)
                threshold = self.max_deviations[func] * np.sqrt(2)
                if dev <= threshold and dev < min_deviation:
                    min_deviation = dev
                    best_match = func

            mapped_data.append({
                'x': x_test,
                'y': y_test,
                'matched_ideal_function': best_match,
                'deviation': min_deviation if best_match else None
            })

        self.mapped_df = pd.DataFrame(mapped_data)

        # Save only the mapped points, using the column names of Table 3
        to_save = self.mapped_df.dropna(subset=['matched_ideal_function'])
        to_save = to_save.rename(columns={
            'matched_ideal_function': 'ideal_func_no',
            'deviation': 'delta_y',
        })
        to_save = to_save[['x', 'y', 'delta_y', 'ideal_func_no']]
        self.db_manager.insert_dataframe(to_save, 'mapped_test_data')
        return self.mapped_df


class DataVisualizer:
    """Draws training data, matched ideal functions and mapped test points."""

    def __init__(self, train_df, ideal_df, test_df, mapped_df, best_funcs):
        """Store the data frames and the chosen function pairs."""
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.test_df = test_df
        self.mapped_df = mapped_df
        self.best_funcs = best_funcs

    def plot(self):
        """Draw training data, matched ideal functions and mapped test points."""
        colors = {'y1': 'blue', 'y2': 'green', 'y3': 'orange', 'y4': 'purple'}
        p = figure(title="Training data, ideal functions and mapped test points",
                   x_axis_label='x', y_axis_label='y',
                   width=900, height=600)

        # Training data: dots, one color per training function
        for y_col in ['y1', 'y2', 'y3', 'y4']:
            p.scatter(self.train_df['x'], self.train_df[y_col],
                      color=colors[y_col], size=3, alpha=0.4,
                      legend_label=f"Training {y_col}")

        # Matching ideal function: line in the same color as its training data
        for train_y, ideal_y in self.best_funcs.items():
            p.line(self.ideal_df['x'], self.ideal_df[ideal_y],
                   color=colors[train_y], line_width=2,
                   legend_label=f"Ideal {ideal_y} (fits {train_y})")

        # Mapped test points: triangles in the color of their ideal function
        mapped = self.mapped_df[self.mapped_df['matched_ideal_function'].notna()]
        for train_y, ideal_y in self.best_funcs.items():
            pts = mapped[mapped['matched_ideal_function'] == ideal_y]
            p.scatter(pts['x'], pts['y'], marker='triangle', size=10,
                      color=colors[train_y], line_color='black',
                      legend_label=f"Test points mapped to {ideal_y}")

        p.legend.click_policy = "hide"
        p.legend.location = "top_left"
        show(p)


if __name__ == "__main__":
    train_csv_path = r"C:\Users\sahil\OneDrive\code\Dataset2\Dataset2\train.csv"
    ideal_csv_path = r"C:\Users\sahil\OneDrive\code\Dataset2\Dataset2\ideal.csv"
    test_csv_path = r"C:\Users\sahil\OneDrive\code\Dataset2\Dataset2\test.csv"
    db_file_path = "assignment.db"

    db_mgr = DatabaseManager(db_file_path)
    db_mgr.create_tables()

    train_df = CsvLoader(train_csv_path, ["x", "y1", "y2", "y3", "y4"]).load()
    ideal_df = CsvLoader(ideal_csv_path, ["x"] + [f"y{i}" for i in range(1, 51)]).load()
    test_df = CsvLoader(test_csv_path, ["x", "y"]).load()

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
    print("Mapped points:", mapped_df['matched_ideal_function'].notna().sum(),
          "of", len(mapped_df))
    print("Mapping completed! All data written to SQLite database.")

    visualizer = DataVisualizer(
        train_handler.train_df,
        train_handler.ideal_df,
        test_df,
        mapped_df,
        best_funcs
    )
    visualizer.plot()

    print("Max deviations (D_max):", train_handler.max_deviations)
    print(mapped_df['matched_ideal_function'].value_counts())
    print("Rows - train:", len(train_df), "ideal:", len(ideal_df), "test:", len(test_df))
    print("All test x in ideal table:", test_df['x'].isin(ideal_df['x']).all())

    unmapped = mapped_df[mapped_df['matched_ideal_function'].isna()]
    ideal_idx = train_handler.ideal_df.set_index('x')
    gaps = []
    for _, r in unmapped.iterrows():
        if r['x'] in ideal_idx.index:
            gaps.append(min(abs(r['y'] - ideal_idx.loc[r['x'], f])
                            for f in best_funcs.values()))
    print("Unmapped:", len(unmapped), "with a matching x:", len(gaps))
    print(pd.Series(gaps).describe())


    