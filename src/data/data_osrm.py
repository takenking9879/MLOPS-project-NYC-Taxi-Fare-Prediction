from src.utils import create_logger, BaseUtils
from src.osrm_console import run_osrm_in_chunks_single_csv
import pandas as pd
import os 
import joblib
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler

logger = create_logger('data_osrm', 'osrm_errors.log')

class DataOSRM(BaseUtils):
    def __init__(self, params_path: str, preprocessed_data_dir: str = None, preprocessed_osrm_data_dir: str = None):
        super().__init__(logger=logger,params_path=params_path)
        self.preprocessed_osrm_data_dir = preprocessed_osrm_data_dir or os.path.join("data", "processed_osrm")
        self.preprocessed_data_dir = preprocessed_data_dir or os.path.join("data", "processed")
        self.params = self.load_params()["data_osrm"]
        self.scaler = StandardScaler()

    def osrm_feature(self, df: pd.DataFrame, use_csv_path: str = None, force_recalc: bool = False) -> pd.DataFrame:
            """
            Calcula o carga los features de OSRM para un DataFrame.
            
            Parámetros:
            - df: DataFrame con columnas ['Start_Lon','Start_Lat','End_Lon','End_Lat'].
            - use_csv_path: si se provee, usa este CSV existente para los resultados de OSRM.
            - force_recalc: si True, fuerza el recálculo incluso si el CSV existe.
            """
            try:
                params = self.params["osrm"]
                if not params.get("add_osrm", False):
                    return df

                # Carpeta base para OSRM
                osrm_dir = "data/osrm"
                os.makedirs(osrm_dir, exist_ok=True)

                # Determinar CSV final
                if use_csv_path is None:
                    csv_path = os.path.join(osrm_dir, "osrm_routes.csv")
                else:
                    csv_path = use_csv_path

                # Si existe CSV y no forzamos recálculo, lo usamos
                if os.path.exists(csv_path) and not force_recalc:
                    self.logger.info(f"[OSRM] Usando CSV existente {csv_path}")
                    dtypes = {
                        "orig_index": np.int64,
                        "route_distance_m": np.float32,
                        "route_duration_s": np.float32
                    }
                    routes_df = pd.read_csv(csv_path, dtype=dtypes).set_index("orig_index").sort_index()
                    df_out = df.copy()
                    df_out[["route_distance_m", "route_duration_s"]] = routes_df[["route_distance_m", "route_duration_s"]]
                    return df_out

                # 🚀 Si no existe CSV o forzamos recálculo, ejecutar OSRM
                df, csv_path = run_osrm_in_chunks_single_csv(
                    df,
                    osrm_base=params["osrm_base"],
                    concurrency=params["concurrency"],
                    chunk_size=params["chunk_size"],
                    csv_path=csv_path,
                    chunk_progress=True,
                    cleanup=True
                )

                df[["route_distance_m", "route_duration_s"]] = df[["route_distance_m", "route_duration_s"]].astype("float32")
                gc.collect()
                self.logger.info("Las distancias OSRM han sido calculadas")
                return df

            except Exception as e:
                self.logger.error("Hubo un error calculando OSRM", e)
                raise



    def normalize(self, df: pd.DataFrame, columns :list, fit: bool = True) -> pd.DataFrame:
        try:
            self.logger.info("Normalizando columnas...")
            cols_to_scale = columns
            self.columns_scaler = df[cols_to_scale].columns     # <-- guardamos las columnas originales
            if fit:
                df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
                self.logger.debug("Data normalized with fit_transform")
            else:
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
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
            all_files = os.listdir(self.preprocessed_data_dir)
            
            # Mover 'Train.parquet' al inicio si existe
            if 'train_processed.parquet' in all_files:
                all_files.remove('train_processed.parquet')
                all_files = ['train_processed.parquet'] + all_files

            for f in all_files:
                file_path_load = os.path.join(self.preprocessed_data_dir, f)
                if f.endswith('.parquet'):
                    name = f.split(".")[0]
                    fit = ('train_processed' == name)
                    df = pd.read_parquet(file_path_load)
                    df = self.osrm_feature(df, use_csv_path=os.path.join("data/osrm", "osrm_routes.csv"),
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
                        df = self.normalize(df, norm_columns, fit)
                    file_path = os.path.join(self.preprocessed_osrm_data_dir, f"{name}_osrm.parquet")
                    gc.collect()
                    df.to_parquet(file_path, index=False)
                    del df
                    self.logger.debug(f"Saved {name} to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving datasets: {e}")
            raise

    def save_scaler(self, filename: str = "scaler_osrm_features.pkl") -> None:
        try:
            artifacts_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')), "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            save_path = os.path.join(artifacts_dir, filename)
            
            # Guardar scaler + columnas originales
            joblib.dump({
                'scaler': self.scaler,
                'columns': self.columns_scaler
            }, save_path)
            
            self.logger.debug(f"Scaler guardado en {save_path}")
        except Exception as e:
            self.logger.error(f"No se pudo guardar el scaler: {e}")
            raise

def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        processed_osrm_data_dir = os.path.join(root_dir, 'data/processed_osrm')
        processed_data_dir = os.path.join(root_dir, 'data/processed')

        preprocessing = DataOSRM(params_path,processed_data_dir,processed_osrm_data_dir)
        preprocessing.add_features()
        preprocessing.save_scaler()

        remove_temp_csv = '../../data/osrm/osrm_routes.csv'
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