import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from pathlib import Path

class AnomalyDetector:
    def __init__(self, input_path: str, model_save_path: str):
        self.input_path = input_path
        self.model_save_path = model_save_path
        # We select the most indicative features for EV health
        self.feature_cols = [
            'I_bat_ev', 'Ubat_ev', 'Power_DC_ev', 
            'efficiency_ratio', 'v_i_ratio', 
            'rolling_power_std', 'power_per_rpm'
        ]
        # contamination=0.05 means we expect roughly 5% of data might be anomalous
        self.model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

    def load_features(self):
        return pd.read_csv(self.input_path)

    def train_and_predict(self):
        print("Training Isolation Forest Model...")
        df = self.load_features()
        
        # Extract features for the model
        X = df[self.feature_cols]
        
        # Fit the model and predict
        # -1 = Anomaly, 1 = Normal
        df['anomaly_score'] = self.model.fit_predict(X)
        
        # Convert to boolean for easier UI handling later
        df['is_anomaly'] = df['anomaly_score'].apply(lambda x: True if x == -1 else False)
        
        # Calculate decision function (lower scores are more anomalous)
        df['anomaly_magnitude'] = self.model.decision_function(X)
        
        return df

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump(self.model, self.model_save_path)
        print(f" Model saved to: {self.model_save_path}")

    def run(self):
        df_results = self.train_and_predict()
        self.save_model()
        
        # Save the results back to the features file or a dedicated results file
        results_path = self.input_path.replace("features.csv", "results.csv")
        df_results.to_csv(results_path, index=False)
        
        anomalies_found = df_results['is_anomaly'].sum()
        print(f" Detection Complete. Found {anomalies_found} potential anomalies.")
        return df_results

if __name__ == "__main__":
    FEATURES_CSV = "data/features/telemetry_features.csv"
    MODEL_PATH = "models/anomaly_detector.joblib"
    
    detector = AnomalyDetector(FEATURES_CSV, MODEL_PATH)
    detector.run()