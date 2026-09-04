import pandas as pd
import os
from pathlib import Path

class TelemetryETL:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        
    def load_data(self):
        """Loads raw telemetry CSV."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Raw data not found at {self.input_path}")
        return pd.read_csv(self.input_path)

    def clean_data(self, df: pd.DataFrame):
        """Standardizes columns and handles missing values."""
        # 1. Handle missing values
        df = df.dropna()
        
        # 2. Time-series sorting (ensure chronological order)
        if 'time' in df.columns:
            df = df.sort_values(by='time')
            
        # 3. Basic filtering (Example: Remove rows with 0 voltage if car is moving)
        # According to your CSV: 'Ubat_ev' is battery voltage
        df = df[df['Ubat_ev'] > 0]
        
        return df

    def run(self):
        print(" Starting ETL Process...")
        df_raw = self.load_data()
        df_clean = self.clean_data(df_raw)
        
        # Save to processed folder
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df_clean.to_csv(self.output_path, index=False)
        print(f" Cleaned data saved to: {self.output_path}")
        return df_clean

if __name__ == "__main__":
    # Update paths based on your folder structure
    INPUT_CSV = "data/raw/MEGANE_E_TECH_EV60_220_driv_data.csv"
    OUTPUT_CSV = "data/processed/telemetry_cleaned.csv"
    
    etl = TelemetryETL(INPUT_CSV, OUTPUT_CSV)
    etl.run()