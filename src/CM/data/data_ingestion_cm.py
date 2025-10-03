from src.CT.data.data_ingestion_ct import DataIngestionCT
from src.utils import create_logger
from pathlib import Path
import os
import pandas as pd
import gc

logger = create_logger('data_ingestion_cm', 'logs/ingestion_cm_errors.log')

class DataIngestionCM(DataIngestionCT):
    def __init__(self, params_path: str, raw_data_dir: str, original_data_dir: str):
        super().__init__(params_path=params_path, raw_data_dir=raw_data_dir, original_data_dir=original_data_dir)
        self.logger = logger
        self.params = self.load_params()["continuous_monitoring"]
        self.retrain = True
        self.params = self.params["data_ingestion_cm"]

    def load_parquet(self, path: str, columns: list) -> pd.DataFrame:
        try:
            df = pd.read_parquet(path)
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
            return df[columns]
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

            files = list(folder_path.glob("*.parquet"))
            if len(files) != 1:
                self.logger.error(f"Se esperaba un solo archivo parquet, pero se encontraron {len(files)}")
                return None

            parquet_file = str(files[0])
            self.logger.info(f"Usando archivo parquet: {parquet_file}")

            # Cargar y procesar
            df = self.load_parquet(parquet_file, columns_required)
            df = self.filter_valid_coordinates(df)
            df = self.columns_to_datetime(df)
            df = self.sort_bydate(df)

            # Guardar todo junto como monitoring.parquet
            self.save_data_parquet(df, 'monitoring')

            self.logger.info("Parquet de monitoring guardado")

            del df
            gc.collect()

        except Exception as e:
            self.logger.error("No se pudo procesar el parquet de CM: %s", e)


def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        raw_data_dir = os.path.join(root_dir, 'CM_data/raw')
        original_data_dir = os.path.join(root_dir, 'CM_data/original')

        ingestion = DataIngestionCM(params_path, raw_data_dir, original_data_dir)
        ingestion.split_data()

    except Exception as e:
        ingestion.logger.error(f"Failed to complete the data ingestion pipeline: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
