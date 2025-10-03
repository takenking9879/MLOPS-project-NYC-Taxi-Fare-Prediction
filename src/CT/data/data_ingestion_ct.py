
from src.data.data_ingestion import DataIngestion
from src.utils import create_logger
from pathlib import Path
import os
import pandas as pd
import gc

logger = create_logger('data_ingestion_ct', 'logs/ingestion_ct_errors.log')

class DataIngestionCT(DataIngestion):
    def __init__(self, params_path: str, raw_data_dir: str, original_data_dir: str):
        super().__init__(params_path=params_path,raw_data_dir=raw_data_dir, original_data_dir=original_data_dir)
        self.logger = logger
        self.params = self.load_params()["continuous_training"]
        self.retrain = self.params["retrain"]
        self.params = self.params["data_ingestion_ct"]

    def load_parquet(self, path: str, columns: list) -> pd.DataFrame:
        try:
            df = pd.read_parquet(path, columns=columns)
            column_mappings = {
                "Trip_Pickup_DateTime": [
                    "pickup_datetime", "tpep_pickup_datetime", "lpep_pickup_datetime",
                    "Trip_Pickup_DateTime", "Pickup_DateTime"
                ],
                "Trip_Dropoff_DateTime": [
                    "dropoff_datetime", "tpep_dropoff_datetime", "lpep_dropoff_datetime",
                    "Trip_Dropoff_DateTime", "Dropoff_DateTime"
                ],
                "Passenger_Count": [
                    "passenger_count", "Passenger_count", "Passenger_Count",
                    "passengers", "num_passengers"
                ],
                "Trip_Distance": [
                    "trip_distance", "Trip_distance", "Trip_Distance",
                    "distance", "distance_miles", "trip_miles"
                ],
                "Start_Lon": [
                    "pickup_longitude", "Start_Lon", "pickup_long", "pickup_lon",
                    "start_longitude"
                ],
                "Start_Lat": [
                    "pickup_latitude", "Start_Lat", "pickup_lat", "start_latitude"
                ],
                "End_Lon": [
                    "dropoff_longitude", "End_Lon", "dropoff_long", "dropoff_lon",
                    "end_longitude"
                ],
                "End_Lat": [
                    "dropoff_latitude", "End_Lat", "dropoff_lat", "end_latitude"
                ],
                "Fare_Amt": [
                    "fare_amount", "Fare_Amt", "fare", "fare_amt", "base_fare"
                ],
                "Total_Amt": [
                    "total_amount", "Total_Amt", "total", "total_fare", "payment_total"
                ]
            }
            for standard_name, variants in column_mappings.items():
                for var in variants:
                    if var in df.columns:
                        df = df.rename(columns={var: standard_name})
                        break
            self.logger.debug('Parquet file retrived from %s', path)
            return df
        except FileNotFoundError:
            self.logger.error('File not found: %s', path)
            raise
        except Exception as e:
            self.logger.error('Unexpected error: %s', e)
            raise

    def split_data(self):
        try:
            os.makedirs(self.raw_data_dir, exist_ok=True)
            if not self.retrain:
                return None
            columns_required = self.params["columns_required"]
            folder_path = Path(self.original_data_dir)

            # Guardar rutas completas en lugar de solo nombres
            files = [str(f) for f in folder_path.glob("*.parquet")]

            # Extraer años y meses en enteros
            dates = [(f, int(f.split('_')[-1][:4]), int(f.split('-')[-1][:2])) for f in files]

            # Revisar si todos los años son iguales
            if len(set(y for _, y, m in dates)) != 1:
                self.logger.warning("Los archivos no son del mismo año")
                raise

            # Ordenar por mes
            dates_sorted = sorted(dates, key=lambda x: x[2])  # x[2] es el mes

            # Extraer meses y normalizar
            months_sorted = [m for _, _, m in dates_sorted]
            normalized = [m - months_sorted[0] for m in months_sorted]

            # Checar consecutividad
            if normalized != [0, 1, 2]:
                self.logger.warning("Los archivos no son consecutivos, la ventana debe de ser 3 meses consecutivos")

            # Finalmente, files ordenados por mes
            files_sorted = [f for f, _, _ in dates_sorted]

            dfs = []
            self.logger.info("Dividiendo en entrenamiento...")
            for parquet_file in files_sorted[:2]:
                df = self.load_parquet(parquet_file, columns_required)
                dfs.append(df)

            df_train = pd.concat(dfs, axis=0, ignore_index=True)
            del dfs
            df_train = self.filter_valid_coordinates(df_train)
            df_train = self.columns_to_datetime(df_train)
            df_train = self.sort_bydate(df_train)
            self.save_data_parquet(df_train, 'train')
            self.logger.info("train parquet finalizado")
            del df_train
            gc.collect()

            df_val_test = self.load_parquet(files_sorted[2], columns_required)
            df_val_test = self.filter_valid_coordinates(df_val_test)
            df_val_test = self.columns_to_datetime(df_val_test)
            df_val_test = self.sort_bydate(df_val_test)

            mid = len(df_val_test) // 2
            df_val = df_val_test.iloc[:mid]
            df_test = df_val_test.iloc[mid:]
            del df_val_test

            self.save_data_parquet(df_val, 'val')
            self.save_data_parquet(df_test, 'test')
            del df_val
            del df_test
            gc.collect()  # limpia memoria de val y test

        except Exception as e:
            self.logger.error("No se pudo dividir en train-val-test: %s", e)


def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        raw_data_dir = os.path.join(root_dir, 'new_data/raw')
        original_data_dir = os.path.join(root_dir, 'new_data/original')

        ingestion = DataIngestionCT(params_path, raw_data_dir, original_data_dir)
        ingestion.split_data()

    except Exception as e:
        ingestion.logger.error(f"Failed to complete the data ingestion pipeline: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()