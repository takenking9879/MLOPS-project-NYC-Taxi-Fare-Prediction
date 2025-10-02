from src.CT.data.data_preprocessing_ct import DataPreprocessingCT
from src.utils import create_logger
import os
from time import sleep
import pandas as pd
import gc
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
import numpy as np

logger = create_logger('data_preprocessing_cm', 'logs/preprocessing_cm_errors.log')

class DataPreprocessingCM(DataPreprocessingCT):
    def __init__(self, params_path: str, raw_data_dir: str, preprocessed_data_dir: str = None, artifacts_dir: str = None):
        super().__init__(params_path= params_path, raw_data_dir=raw_data_dir, preprocessed_data_dir=preprocessed_data_dir, artifacts_dir=artifacts_dir)
        self.logger = logger
        self.preprocessed_data_dir = preprocessed_data_dir or os.path.join("new_data", "processed")
        all_params = self.load_params()["continuous_monitoring"]
        self.retrain = True
        self.params = all_params["data_preprocessing_cm"]
        self.artifacts_dir = artifacts_dir
        self.columns_imputer = self.columns_imputer_aux
        self.columns_scaler = self.columns_scaler_aux

    def stratified_sample_skewed(self, df, target_col, n_samples, n_bins=50, random_state=42) -> pd.DataFrame:
        df = df.copy()
        df["_log_target"] = np.log1p(df[target_col])
        df["_bin"] = pd.qcut(df["_log_target"], q=n_bins, duplicates="drop")

        # Calcula n por bin proporcional al tamaño del bin
        bin_counts = df["_bin"].value_counts()
        total = len(df)
        n_per_bin = (bin_counts / total * n_samples).round().astype(int)

        df_sampled = df.groupby("_bin", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), n_per_bin[x.name]), random_state=random_state)
        )

        return df_sampled.drop(columns=["_log_target", "_bin"])

    def preprocess(self) -> None:
        try:
            os.makedirs(self.preprocessed_data_dir, exist_ok=True)
            if not self.retrain:
                return None
            all_files = os.listdir(self.raw_data_dir)

            for f in all_files:
                file_path_load = os.path.join(self.raw_data_dir, f)
                if f.endswith('.parquet'):
                    name = f.split(".")[0]
                    fit = False
                    df = pd.read_parquet(file_path_load)
                    df = self.filtering_extreme_outliers(df)
                    df = self.stratified_sample_skewed(df, target_col="Fare_Amt", n_samples=600000, n_bins=50)
                    df = self.fit_imputer(df, fit)
                    df = self.time_features(df)
                    pois = {
                        "nyc": (40.724944, -74.001541),
                        "jfk": (40.645494, -73.785937),
                        "lga": (40.774071, -73.872067),
                        "nla": (40.690764, -74.177721),
                        "chp": (41.366138, -73.137393),
                        "exp": (40.736000, -74.037500)
                    }
            
                    norm_columns = [
                        'haversine_distance_m',
                        'pickup_distance_to_nyc_m', 'dropoff_distance_to_nyc_m',
                        'pickup_distance_to_jfk_m', 'dropoff_distance_to_jfk_m',
                        'pickup_distance_to_lga_m', 'dropoff_distance_to_lga_m',
                        'pickup_distance_to_nla_m', 'dropoff_distance_to_nla_m',
                        'pickup_distance_to_chp_m', 'dropoff_distance_to_chp_m',
                        'pickup_distance_to_exp_m', 'dropoff_distance_to_exp_m'
                    ]

                    sleep(1)
                    df = self.add_haversine_poi_distances(df, pois)
                    df = self.normalize(df, fit=fit)
                    df['Passenger_Count'] = df['Passenger_Count'] * 0.1 #Llevamos Passenger_Count al rango [0,1]. Es rápido saber como se normaliza
                    file_path = os.path.join(self.preprocessed_data_dir, f"{name}_processed.parquet")
                    gc.collect()
                    df.to_parquet(file_path, index=False)
                    del df
                    self.logger.debug(f"Saved {name} to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving datasets: {e}")
            raise


def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        raw_data_dir = os.path.join(root_dir, 'CM_data/raw')
        processed_data_dir = os.path.join(root_dir, 'CM_data/processed')
        artifacts_dir = os.path.join(root_dir, 'artifacts')

        sleep(1)

        preprocessing = DataPreprocessingCM(params_path, raw_data_dir, processed_data_dir, artifacts_dir)
        preprocessing.preprocess()

    except Exception as e:
        preprocessing.logger.error(f"Failed to complete the data preprocessing pipeline: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()