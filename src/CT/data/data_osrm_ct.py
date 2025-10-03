from src.data.data_osrm import DataOSRM
from src.utils import create_logger
logger = create_logger('data_osrm_ct', 'logs/osrm_ct_errors.log')
import os
import joblib
import pandas as pd
import gc 

class DataOSRMCT(DataOSRM):
    def __init__(self, params_path: str, preprocessed_data_dir: str = None, preprocessed_osrm_data_dir: str = None, artifacts_dir: str = None):
        super().__init__(params_path=params_path, preprocessed_data_dir=preprocessed_osrm_data_dir, preprocessed_osrm_data_dir=preprocessed_osrm_data_dir)
        self.preprocessed_osrm_data_dir = preprocessed_osrm_data_dir or os.path.join("new_data", "processed_osrm")
        self.preprocessed_data_dir = preprocessed_data_dir or os.path.join("new_data", "processed")
        all_params = self.load_params()["continuous_training"]
        self.retrain = all_params["retrain"]
        self.params = all_params["data_osrm_ct"]
        self.artifacts_dir = artifacts_dir
        self.scaler, self.columns_scaler = self.load_scaler()

    def load_scaler(self):
        scaler_path = os.path.join(self.artifacts_dir, "scaler_osrm_features.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        data = joblib.load(scaler_path)
        return data['scaler'], data['columns']
    
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

    def add_features(self) -> None:
        try:
            os.makedirs(self.preprocessed_osrm_data_dir, exist_ok=True)
            if not self.retrain:
                return None

            all_files = os.listdir(self.preprocessed_data_dir)
            
            # Mover 'Train.parquet' al inicio si existe
            if 'train_processed.parquet' in all_files:
                all_files.remove('train_processed.parquet')
                all_files = ['train_processed.parquet'] + all_files

            for f in all_files:
                file_path_load = os.path.join(self.preprocessed_data_dir, f)
                if f.endswith('.parquet'):
                    name = f.split(".")[0]
                    fit = False
                    df = pd.read_parquet(file_path_load)
                    df = self.osrm_feature(df, use_csv_path=os.path.join("new_data/osrm", "osrm_routes.csv"),
                                            force_recalc= (name == "val_processed" or name == "test_processed"))  
                    # por ejemplo, solo recalcular para val y test
                                        
                    norm_columns = []
                    if self.params["osrm"]["add_osrm"]==True:
                        self.logger.info(f"Número de filas al inicio de OSRM {df.shape[0]}")
                        df = df[df['route_distance_m']>0]
                        df = df[df['route_duration_s']>0]
                        df = df.dropna()
                        self.logger.info(f"Número de filas con OSRM calculado {df.shape[0]}")
                        norm_columns += ['route_distance_m', 'route_duration_s', 'average_speed_m_s', 'ratio_haversine_osrm']
                        df['average_speed_m_s'] = (df['route_distance_m'] / df['route_duration_s']).astype("float32")
                        df['ratio_haversine_osrm'] = df['route_distance_m'] / df['haversine_distance_m'].astype("float32")
                        df = self.normalize(df, fit)
                    file_path = os.path.join(self.preprocessed_osrm_data_dir, f"{name}_osrm.parquet")
                    gc.collect()
                    df.to_parquet(file_path, index=False)
                    del df
                    self.logger.debug(f"Saved {name} to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving datasets: {e}")
            raise

def main():
    try:
        gc.collect()
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        processed_osrm_data_dir = os.path.join(root_dir, 'new_data/processed_osrm')
        processed_data_dir = os.path.join(root_dir, 'new_data/processed')
        artifacts_dir = os.path.join(root_dir, 'artifacts')

        preprocessing = DataOSRMCT(params_path,processed_data_dir,processed_osrm_data_dir, artifacts_dir)
        preprocessing.add_features()

        remove_temp_csv = '/home/jorge/DocumentsWLS/Data_Science_Projects/MLOPS-project-NYC-Taxi-Fare-Prediction/new_data/osrm/osrm_routes.csv'
        if os.path.exists(remove_temp_csv):
            os.remove()
            print(f"{remove_temp_csv} eliminado correctamente")
        else:
            print(f"{remove_temp_csv} no existe")
        gc.collect()
    except Exception as e:
        preprocessing.logger.error(f"Failed to complete the data preprocessing pipeline: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()