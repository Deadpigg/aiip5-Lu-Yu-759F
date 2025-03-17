import pandas as pd
import numpy as np
from data_loader import load_data  # Import function from data_loader.py

def preprocess_data(df):
    """
    Cleans and preprocesses the dataset:
    - Standardizes text columns (lowercase, no extra spaces)
    - Ensures numerical columns are correctly formatted
    - Handles missing values
    - Maps categorical variables
    - Removes duplicates & outliers
    """
    
    # 1️⃣ Standardize text columns
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].str.lower().str.strip()

    # 2️⃣ Convert numeric columns to float
    numeric_cols = [
        "Temperature Sensor (°C)", "Light Intensity Sensor (lux)", 
        "CO2 Sensor (ppm)", "EC Sensor (dS/m)", "O2 Sensor (ppm)", 
        "pH Sensor", "Water Level Sensor (mm)", "Nutrient N Sensor (ppm)", 
        "Nutrient P Sensor (ppm)", "Nutrient K Sensor (ppm)"
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[numeric_cols] = df[numeric_cols].astype(float)

    # 3️⃣ Handle missing target variable
    missing_target_values = df['Plant Stage'].isnull().sum()
    if missing_target_values > 0:
        df = df.dropna(subset=['Plant Stage'])

    # 4️⃣ Ensure sensor values are non-negative
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: max(0, x) if x is not None else None)

    # 5️⃣ Handle missing values
    df.replace("", np.nan, inplace=True)  # Convert empty strings to NaN
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Fill numeric NaNs with median
    df[text_columns] = df[text_columns].fillna("unknown")  # Fill text NaNs with 'unknown'

    # 6️⃣ Map categorical variables
    mapping_dict = {
        "Previous Cycle Plant Type": {"herbs": 1, "leafy greens": 2, "vine crops": 3, "fruiting vegetables": 4},
        "Plant Type": {"fruiting vegetables": 1, "herbs": 2, "leafy greens": 3, "vine crops": 4},
        "Plant Stage": {"seedling": 1, "vegetative": 2, "maturity": 3}
    }
    
    for col in mapping_dict:
        if col in df.columns:
            df[col] = df[col].map(mapping_dict[col]).fillna(-1)  # Assign -1 for unmapped values

    # 7️⃣ Remove duplicates
    df = df.drop_duplicates()

    # 8️⃣ Remove outliers using the IQR method
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)]
    
    return df

if __name__ == "__main__":
    df = load_data()
    df_cleaned = preprocess_data(df)
    print("Data preprocessing complete!")
    print(df_cleaned.head())  # Display first few rows
