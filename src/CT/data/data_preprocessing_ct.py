from src.data.data_preprocessing import DataPreprocessing
from src.utils import create_logger
import os
from time import sleep
import pandas as pd
import gc
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer

logger = create_logger('data_preprocessing_ct', 'logs/preprocessing_ct_errors.log')

class DataPreprocessingCT(DataPreprocessing):
    def __init__(self, params_path: str, raw_data_dir: str, preprocessed_data_dir: str = None, artifacts_dir: str = None):
        self.artifacts_dir = artifacts_dir
        super().__init__(params_path= params_path, raw_data_dir=raw_data_dir, preprocessed_data_dir=preprocessed_data_dir)
        self.logger = logger
        self.preprocessed_data_dir = preprocessed_data_dir or os.path.join("new_data", "processed")
        all_params = self.load_params()["continuous_training"]
        self.retrain = all_params["retrain"]
        self.params = all_params["data_preprocessing_ct"]
        self.columns_imputer = self.columns_imputer_aux
        self.columns_scaler = self.columns_scaler_aux

    def load_scaler(self):
        scaler_path = os.path.join(self.artifacts_dir, "scaler_preprocessing.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        data = joblib.load(scaler_path)
        return data['scaler'], data['columns']
    
    def load_imputer(self):
        imputer_path = os.path.join(self.artifacts_dir, "imputer.pkl")
        if not os.path.exists(imputer_path):
            raise FileNotFoundError(f"Scaler not found at {imputer_path}")
        data = joblib.load(imputer_path)
        return data['imputer'], data['columns']

    def _choose_scaler(self):
        scaler, columns = self.load_scaler()
        self.columns_scaler_aux = columns
        return scaler
    
    def _choose_imputer(self):
        imputer, columns = self.load_imputer()
        self.columns_imputer_aux = columns
        return imputer
    
    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        try:
            self.logger.info("Normalizando columnas...")
            cols_to_scale = self.columns_scaler
            if fit:
                df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
                self.logger.debug("Data normalized with fit_transform")
            else:
                df[self.columns_scaler] = self.scaler.transform(df[self.columns_scaler])
                self.logger.debug("Data normalized with transform")
            self.logger.info("Columnas normalizadas")
            gc.collect()
            return df
        except Exception as e:
            self.logger.error(f"Error normalizing data: {e}")
            raise

    def preprocess(self) -> None:
        try:
            os.makedirs(self.preprocessed_data_dir, exist_ok=True)
            if not self.retrain:
                return None
            all_files = os.listdir(self.raw_data_dir)
            
            # Mover 'Train.parquet' al inicio si existe
            if 'train.parquet' in all_files:
                all_files.remove('train.parquet')
                all_files = ['train.parquet'] + all_files

            for f in all_files:
                file_path_load = os.path.join(self.raw_data_dir, f)
                if f.endswith('.parquet'):
                    name = f.split(".")[0]
                    fit = False
                    df = pd.read_parquet(file_path_load)
                    df = self.filtering_extreme_outliers(df)
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
        raw_data_dir = os.path.join(root_dir, 'new_data/raw')
        processed_data_dir = os.path.join(root_dir, 'new_data/processed')
        artifacts_dir = os.path.join(root_dir, 'artifacts')

        sleep(1)

        preprocessing = DataPreprocessingCT(params_path, raw_data_dir, processed_data_dir, artifacts_dir)
        preprocessing.preprocess()

    except Exception as e:
        preprocessing.logger.error(f"Failed to complete the data preprocessing pipeline: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()