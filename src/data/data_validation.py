
import pandas as pd
from pathlib import Path
import glob
import os
from src.utils import BaseUtils, create_logger

logger = create_logger('data_validation', 'validation_errors.log')

class DataValidation(BaseUtils):
    def __init__(self, params_path: str, original_data_dir: str):
        super().__init__(logger=logger,params_path=params_path)
        self.original_data_dir = original_data_dir
        self.params = self.load_params()["data_validation"]

    def check_formats(self, df: pd.DataFrame, columns_num: list = None, columns_str: list = None):
        """Checa que ciertas columnas sean numéricas o string.
        columns_num: lista de columnas que deberían ser numéricas
        columns_str: lista de columnas que deberían ser string
        """
        try:
            for col in columns_num:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        self.logger.warning("Columna '%s' debería ser numérica pero no lo es.", col)
                else:
                    self.logger.warning("Columna '%s' no existe en el DataFrame.", col)

            for col in columns_str:
                if col in df.columns:
                    if not pd.api.types.is_string_dtype(df[col]):
                        self.logger.warning("Columna '%s' debería ser string pero no lo es.", col)
                else:
                    self.logger.warning("Columna '%s' no existe en el DataFrame.", col)
        except Exception as e:
            self.logger.error('Unexpected error: %s', e)
            raise

    def check_parquets(self):
        columns_required = self.params["columns_required"]
        columns_num = self.params["columns_num"]
        columns_str = self.params["columns_str"]
        try:
            folder_path = Path(self.original_data_dir)
            for parquet_file in folder_path.rglob("*.parquet"):  # o "*.parquet"
                df = self.load_parquet(parquet_file, columns_required)
                self.logger.info("Validating file: %s", parquet_file.name)
                self.check_formats(df, columns_num, columns_str)
                del df
            self.logger.info("All files have been examined")
        except Exception as e:
            self.logger.error('Unexpected error: %s', e)
            raise
                
#Ver si checo tambien los nuevos de entrenamiento y prueba. O ver como esta eso luego
    
def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params_path =  os.path.join(root_dir, 'params.yaml')
        original_data_dir = os.path.join(root_dir, 'data/original')

        validation = DataValidation(params_path, original_data_dir)
        validation.check_parquets()

    except Exception as e:
        validation.logger.error(f"Failed to complete the data validation pipeline: {e}")
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()