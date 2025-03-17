import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

def load_data(db_path=None):
    """
    Loads data from the SQLite database and returns a Pandas DataFrame.
    If no db_path is provided, it will use the path from the environment variable DB_PATH.
    """
    if db_path is None:
        db_path = os.getenv("DB_PATH")  # Use the DB_PATH from .env file if not provided

    # Print to check if DB_PATH is correctly loaded
    print(f"Database path: {db_path}")

    if db_path is None:
        raise ValueError("Database path is not provided and no DB_PATH is set in the environment variables.")

    conn = sqlite3.connect(db_path)
    
    # Get table names
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:", tables)

    # Load data if table exists
    if tables:
        table_name = tables[0][0]  # Select the first table in the database
        df = pd.read_sql(f"SELECT * FROM {table_name};", conn)  
    else:
        raise ValueError("No table found in the database.")
    
    conn.close()
    
    return df

if __name__ == "__main__":
    df = load_data()  # The function will now use the DB_PATH from the .env file
    print("Data successfully loaded!")
    print(df.head())  # Display the first few rows of the loaded data
