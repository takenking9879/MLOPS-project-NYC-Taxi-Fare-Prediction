from src.utils import create_logger, BaseUtils
import os
from pathlib import Path
import pandas as pd
import gc
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import gc
from shapely import contains_xy

# Logging configuration
logger = create_logger('data_ingestion', 'ingestion_errors.log')

class DataIngestion(BaseUtils):
    def __init__(self, params_path: str, raw_data_dir: str, original_data_dir: str):
        super().__init__(logger=logger, params_path=params_path)
        self.raw_data_dir = raw_data_dir
        self.original_data_dir = original_data_dir
        self.params = self.load_params()["data_ingestion"]

    def columns_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:    
        columns_datetime = self.params["columns_datetime"]
        for col in columns_datetime:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    self.logger.info("Columna '%s' convertida a datetime.", col)
                except Exception as e:
                    self.logger.warning("No se pudo convertir la columna '%s' a datetime: %s", col, e)
            else:
                self.logger.error("Columna '%s' no existe en el DataFrame.", col)
                raise
        return df
    
    def sort_bydate(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            return df.sort_values(by='Trip_Pickup_DateTime')
        except Exception as e:
            self.logger.error("No se pudo reordenar: %s", e)
            raise

    def filter_valid_coordinates(self, df: pd.DataFrame, chunk_size: int = 500_000) -> pd.DataFrame:
        """
        Filtra filas cuyo Start y End estén dentro del polígono de USA,
        procesando en chunks y con mínima memoria adicional.
        """
        try:
            # 1️⃣ Filtro trivial rápido (validez de rango)
            good = (
                df['Start_Lat'].between(-90, 90) &
                df['End_Lat'].between(-90, 90) &
                df['Start_Lon'].between(-180, 180) &
                df['End_Lon'].between(-180, 180)
            )
            df = df.loc[good].copy()
            n = len(df)
            if n == 0:
                return df

            # 2️⃣ Ruta robusta al shapefile
            root_dir = Path(__file__).resolve().parents[2]
            shp_path = root_dir / "USA_polygon" / "cb_2018_us_nation_5m.shp"

            if not shp_path.exists():
                self.logger.warning(f"Shapefile no encontrado: {shp_path}. Se omite filtro de USA.")
                return df

            gdf_usa = gpd.read_file(shp_path)
            gdf_usa = gdf_usa.to_crs(epsg=4326)
            usa_union = gdf_usa.geometry.union_all()

            # 3️⃣ Preparar máscara booleana
            keep_mask = np.zeros(n, dtype=bool)

            # 5️⃣ Preparar arrays numpy para evitar overhead pandas
            start_lons = df['Start_Lon'].to_numpy(dtype=np.float64)
            start_lats = df['Start_Lat'].to_numpy(dtype=np.float64)
            end_lons = df['End_Lon'].to_numpy(dtype=np.float64)
            end_lats = df['End_Lat'].to_numpy(dtype=np.float64)

            # 6️⃣ Procesar por chunks
            for i in range(0, n, chunk_size):
                j = min(i + chunk_size, n)
                xs_s = start_lons[i:j]
                ys_s = start_lats[i:j]
                xs_e = end_lons[i:j]
                ys_e = end_lats[i:j]

                # vectorized.contains es muy rápido
                start_in = contains_xy(usa_union, xs_s, ys_s)
                end_in = contains_xy(usa_union, xs_e, ys_e)
                keep_mask[i:j] = start_in & end_in
                
                # liberar arrays del chunk
                del xs_s, ys_s, xs_e, ys_e
                gc.collect()

            # 7️⃣ Aplicar máscara y devolver
            filtered = df.iloc[keep_mask].copy()
            return filtered

        except Exception as e:
            self.logger.error("No se pudo filtrar las coordenadas: %s", e)
            raise


    def split_data(self):
        try:
            os.makedirs(self.raw_data_dir, exist_ok=True)
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

    def save_data_parquet(self, df: pd.DataFrame, filename: str):
        try:
            os.makedirs(self.raw_data_dir, exist_ok=True)
            file_path = os.path.join(self.raw_data_dir, f"{filename}.parquet")
            df.to_parquet(file_path, index=False)
            self.logger.debug(f"{filename} data saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving {filename} as parquet file: {e}")
            raise

def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        raw_data_dir = os.path.join(root_dir, 'data/raw')
        original_data_dir = os.path.join(root_dir, 'data/original')

        ingestion = DataIngestion(params_path, raw_data_dir, original_data_dir)
        ingestion.split_data()

    except Exception as e:
        ingestion.logger.error(f"Failed to complete the data ingestion pipeline: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
