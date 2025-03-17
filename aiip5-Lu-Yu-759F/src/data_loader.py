import sqlite3
import pandas as pd

def load_data(db_path="/Users/luyufish/Downloads/data/agri.db"):
    """
    Loads data from the SQLite database and returns a Pandas DataFrame.
    """
    conn = sqlite3.connect(db_path)
    
    # Get table names
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:", tables)

    # Load data if table exists
    if tables:
        table_name = tables[0][0]  
        df = pd.read_sql(f"SELECT * FROM {table_name};", conn)  
    else:
        raise ValueError("No table found in the database.")
    
    conn.close()
    
    return df

if __name__ == "__main__":
    df = load_data()
    print("Data successfully loaded!")
    print(df.head())  # Display first few rows
