import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print out the DB path to ensure it's loaded correctly
db_path = os.getenv("DB_PATH")
print("DB_PATH loaded from .env:", db_path)

# Function to load data from SQLite database into a pandas DataFrame
def load_data(query, db_path):
    """
    Loads data from the SQLite database using a SQL query into a pandas DataFrame.

    :param query: SQL query to execute
    :param db_path: Path to the SQLite database file
    :return: pandas DataFrame containing the query result
    """
    # Establish connection to the database
    conn = sqlite3.connect(db_path)
    try:
        # Execute the query and load the data into a pandas DataFrame
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
    finally:
        # Close the connection
        conn.close()

# Example SQL query to fetch data from a table named 'your_table'
query = "SELECT * FROM your_table"

# Load data using the data loader function
data = load_data(query, db_path)

# Display the loaded data
print(data.head())
