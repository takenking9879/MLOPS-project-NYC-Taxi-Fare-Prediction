from src.utils import create_logger, BaseUtils
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
import pandas as pd
import os 
import joblib
import numpy as np
from src.conversions import km_to_m, kmh_to_ms, days_to_s, min_to_s
import gc
from time import sleep


logger = create_logger('data_preprocessing', 'preprocessing_errors.log')

class DataPreprocessing(BaseUtils):
    def __init__(self, params_path: str, raw_data_dir: str, preprocessed_data_dir: str = None):
        super().__init__(logger=logger,params_path=params_path)
        self.raw_data_dir = raw_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir or os.path.join("data", "processed")
        self.params = self.load_params()["data_preprocessing"]
        self.scaler = self._choose_scaler()
        self.imputer = self._choose_imputer()
        self.columns_imputer = None
        self.columns_scaler = None

    def haversine_distance(self,df: str) -> pd.DataFrame:
        # Radio de la Tierra en km
        R = 6371.0
        # Convertir a radianes
        lon1 = np.radians(df['Start_Lon'])
        lat1 = np.radians(df['Start_Lat'])
        lon2 = np.radians(df['End_Lon'])
        lat2 = np.radians(df['End_Lat'])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        df['haversine_distance_m'] = R * c * 1000
        cols_to_float32 = ['Start_Lon','Start_Lat','End_Lon','End_Lat','haversine_distance_m']
        df[cols_to_float32] = df[cols_to_float32].astype("float32")
        return df
    
    def filtering_extreme_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            params = self.params["filters"]
            before_len = df.shape[0]
            self.logger.info(f"Tamaño inicial de datos: {before_len:,}")
            df = df.dropna(subset=['Trip_Pickup_DateTime', 'Trip_Dropoff_DateTime', 'Trip_Distance', 'Fare_Amt', 'Total_Amt'])
            df['Real_time'] = (df['Trip_Dropoff_DateTime'] - df['Trip_Pickup_DateTime']).dt.total_seconds()
            df = df[(df['Passenger_Count'] > 0) & (df['Passenger_Count'] <= 10)]
            
            # -----------------------------
            # 1️⃣ Filtros triviales rápidos
            # -----------------------------
            if params.get('real_time', True):
                df = df[df['Real_time'] > 0.0]
            if params.get('trip_distance', True):
                df = df[df['Trip_Distance'] > 0.0]

            # -----------------------------
            # 2️⃣ Columnas derivadas
            # -----------------------------
            df['Real_distance'] = df['Trip_Distance'] * 1609.344
            del df['Trip_Distance']
            # Convertimos float64 a float32 para maximizar eficiencia sin perder precision significativa

            df['Real_velocity'] = (df['Real_distance'] / df['Real_time']).astype("float32")
            cols_to_float32 = ['Passenger_Count','Real_distance', 'Real_time', 'Fare_Amt', 'Total_Amt']
            df[cols_to_float32] = df[cols_to_float32].astype("float32")

            # -----------------------------
            # 3️⃣ Filtros rápidos sobre derivadas
            # -----------------------------
            quick_filters = [
                ('Real_velocity', '<=', kmh_to_ms(params["real_velocity_limit"])),
                ('Real_time', '<=', days_to_s(params["real_time_days"])),
                ('Real_time', '<', min_to_s(params["constant_vel_limit"][0])),
                ('Real_velocity', '<=', kmh_to_ms(params["constant_vel_limit"][1])),
                ('Real_distance', '<=', km_to_m(params["constant_vel_limit"][2]))
            ]
            
            for col, op, val in quick_filters:
                if op == '<=':
                    df = df[df[col] <= val]
                elif op == '<':
                    df = df[df[col] < val]

            del df['Real_velocity']

            # -----------------------------
            # 4️⃣ Filtros costosos
            # -----------------------------
            df = self.haversine_distance(df)
            df = df[df['haversine_distance_m'] <= km_to_m(params["distance_limit"])]

            df['ratio_haversine_real'] = df['haversine_distance_m'] / df['Real_distance']
            df = df[df['ratio_haversine_real'] < params["haversine_ratio_limit"]]

            # Filtro de pares (ratio, distancia)
            if "ratio_distance_list" in params:
                ratios = np.array([float(x[0]) for x in params["ratio_distance_list"]])
                dists = np.array([km_to_m(float(x[1])) for x in params["ratio_distance_list"]])
                mask = np.any([(df['ratio_haversine_real'] > r) & (df['Real_distance'] > d) for r,d in zip(ratios,dists)], axis=0)
                df = df[~mask]
                del df['ratio_haversine_real']

            gc.collect()


            self.logger.info(f"Tamaño final de datos: {df.shape[0]:,} (eliminados {before_len - df.shape[0]:,})")
            return df
        except Exception as e:
            self.logger.error("Hubo un error al filtrar los datos", e)
            raise

    def time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Extraer hora y día de la semana
        hour = df['Trip_Pickup_DateTime'].dt.hour
        weekday = df['Trip_Pickup_DateTime'].dt.weekday  # 0=Monday, 6=Sunday
        
        del df['Trip_Pickup_DateTime']
        del df['Trip_Dropoff_DateTime']

        # Codificación cíclica
        df['pickup_hour_sin'] = np.sin(2 * np.pi * hour/24)
        df['pickup_hour_cos'] = np.cos(2 * np.pi * hour/24)
        df['pickup_weekday_sin'] = np.sin(2 * np.pi * weekday/7)
        df['pickup_weekday_cos'] = np.cos(2 * np.pi * weekday/7)
        
        del hour
        del weekday

        cyc_cols = ['pickup_hour_sin', 'pickup_hour_cos', 'pickup_weekday_sin', 'pickup_weekday_cos']
        df[cyc_cols] = df[cyc_cols].astype("float32")
        return df

    def log1_features(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            try:
                df[f'{col}_log1p'] = np.log1p(df[col])
            except Exception as e:
                self.logger.warning(f"No se encontró {col} en el dataframe", e)  # ignora columnas que no existen
            gc.collect()
        self.logger.info("Se calcularon las columnas con log")
        return df

    def _choose_scaler(self):
            try:
                scalers = {
                    "standardscaler": StandardScaler,
                    "minmaxscaler": MinMaxScaler,
                    "robustscaler": RobustScaler
                }
                scaler_type = self.params['scaler_method'].lower()
                if scaler_type not in scalers:
                    raise ValueError(f"Unknown scaler_method '{scaler_type}' in params.yaml")
                
                scaler = scalers[scaler_type]()
                self.logger.debug(f"Scaler chosen: {scaler.__class__.__name__}")
                return scaler
            except Exception as e:
                self.logger.error(f"Error choosing scaler: {e}")
                raise

    def _choose_imputer(self):
        try:
            imputers = {
                "knnimputer": KNNImputer,
                "simpleimputer": lambda: SimpleImputer(strategy='mean')
            }
            imputer_type = self.params['imputer_method'].lower()
            if imputer_type not in imputers:
                raise ValueError(f"Unknown imputer_method '{imputer_type}' in params.yaml")
            
            imputer = imputers[imputer_type]()
            self.logger.debug(f"Imputer chosen: {imputer.__class__.__name__}")
            return imputer
        except Exception as e:
            self.logger.error(f"Error choosing imputer: {e}")
            raise

    def fit_imputer(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Fit the imputer using the provided DataFrame (without the target columns)
        and transform the dataframe if there are missing values.
        """
        try:
            # Columnas que nunca se imputan
            drop_cols = [
                'Real_distance', 'Real_time', 'Fare_Amt','Total_Amt',
                'haversine_distance_m', 'route_distance_m', 'route_duration_s',
                'haversine_distance_m_log1p', 'route_distance_m_log1p', 'route_duration_s_log1p',
                'Trip_Pickup_DateTime', 'Trip_Dropoff_DateTime'
            ]

            # Columnas usadas para imputar
            data = df.drop(columns=drop_cols, errors='ignore')

            if fit:
                self.imputer.fit(data)
                self.columns_imputer = data.columns
                self.logger.debug("Imputer fitted with training data")
                gc.collect()


            # Transformar solo si hay Nans
            if df.isnull().sum().sum() != 0:
                df[self.columns_imputer] = self.imputer.transform(df[self.columns_imputer])
                gc.collect()
                self.logger.debug("Missing values imputed")
            return df
        except Exception as e:
            self.logger.error(f"Error fitting imputer: {e}")
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

    def add_haversine_poi_distances(self, df: pd.DataFrame, pois: dict) -> pd.DataFrame:
        """
        Calcula distancias Haversine desde Start (pickup) y End (dropoff) a puntos de interés.

        """
        try:
            R = 6371.0  # radio de la Tierra en km
            
            # Convertir coordenadas a radianes
            lat_start = np.radians(df['Start_Lat'])
            lon_start = np.radians(df['Start_Lon'])
            lat_end = np.radians(df['End_Lat'])
            lon_end = np.radians(df['End_Lon'])
            
            for name, (lat_poi, lon_poi) in pois.items():
                lat_poi_rad = np.radians(lat_poi)
                lon_poi_rad = np.radians(lon_poi)
                
                # Start -> POI
                dlat = lat_poi_rad - lat_start
                dlon = lon_poi_rad - lon_start
                a = np.sin(dlat/2)**2 + np.cos(lat_start) * np.cos(lat_poi_rad) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                df[f'pickup_distance_to_{name}_m'] = R * c * 1000
                df[f'pickup_distance_to_{name}_m'] = df[f'pickup_distance_to_{name}_m'].astype("float32")
                
                # End -> POI
                dlat = lat_poi_rad - lat_end
                dlon = lon_poi_rad - lon_end
                a = np.sin(dlat/2)**2 + np.cos(lat_end) * np.cos(lat_poi_rad) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                df[f'dropoff_distance_to_{name}_m'] = R * c * 1000  # en metros
                df[f'dropoff_distance_to_{name}_m'] = df[f'dropoff_distance_to_{name}_m'].astype("float32")
                gc.collect()
            return df
        except Exception as e:
            self.logger.error("Hubo un error al calcular las distancias a los POIs", e)
            raise
            


    def preprocess(self) -> None:
        try:
            os.makedirs(self.preprocessed_data_dir, exist_ok=True)
            all_files = os.listdir(self.raw_data_dir)
            
            # Mover 'Train.parquet' al inicio si existe
            if 'train.parquet' in all_files:
                all_files.remove('train.parquet')
                all_files = ['train.parquet'] + all_files

            for f in all_files:
                file_path_load = os.path.join(self.raw_data_dir, f)
                if f.endswith('.parquet'):
                    name = f.split(".")[0]
                    fit = ('train' == name)
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
                    df = self.normalize(df,norm_columns, fit=fit)
                    df['Passenger_Count'] = df['Passenger_Count'] * 0.1 #Llevamos Passenger_Count al rango [0,1]. Es rápido saber como se normaliza
                    file_path = os.path.join(self.preprocessed_data_dir, f"{name}_processed.parquet")
                    gc.collect()
                    df.to_parquet(file_path, index=False)
                    del df
                    self.logger.debug(f"Saved {name} to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving datasets: {e}")
            raise

    def save_scaler(self, filename: str = "scaler_preprocessing.pkl") -> None:
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

    def save_imputer(self, filename: str = "imputer.pkl") -> None:
        try:
            artifacts_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')), "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            save_path = os.path.join(artifacts_dir, filename)
            
            # Guardar imputer + columnas originales
            joblib.dump({
                'imputer': self.imputer,
                'columns': self.columns_imputer  # Usar las mismas columnas del fit
            }, save_path)
            
            self.logger.debug(f"imputer guardado en {save_path}")
        except Exception as e:
            self.logger.error(f"No se pudo guardar el imputer: {e}")
            raise

def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        raw_data_dir = os.path.join(root_dir, 'data/raw')
        processed_data_dir = os.path.join(root_dir, 'data/processed')

        sleep(2)

        preprocessing = DataPreprocessing(params_path, raw_data_dir, processed_data_dir)
        preprocessing.preprocess()
        preprocessing.save_imputer()
        preprocessing.save_scaler()
    except Exception as e:
        preprocessing.logger.error(f"Failed to complete the data preprocessing pipeline: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()