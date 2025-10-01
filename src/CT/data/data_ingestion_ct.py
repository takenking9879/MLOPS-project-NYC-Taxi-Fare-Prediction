
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