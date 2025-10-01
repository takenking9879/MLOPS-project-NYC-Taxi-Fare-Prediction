
from src.data.data_validation import DataValidation
from src.utils import create_logger
from pathlib import Path
import os
import json

logger = create_logger('data_validation_ct', 'logs/validation_ct_errors.log')

class DataValidationCT(DataValidation):
    def __init__(self, params_path: str, original_data_dir: str):
        super().__init__(params_path=params_path, original_data_dir=original_data_dir)
        self.logger = logger
        self.params = self.load_params()["continuous_training"]
        self.retrain = self.params["retrain"]
        self.params = self.params["data_validation_ct"]

    def check_parquets(self):
        if not self.retrain:
            return None
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

def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        params_path =  os.path.join(root_dir, 'params.yaml')
        original_data_dir = os.path.join(root_dir, 'new_data/original')

        validation = DataValidationCT(params_path, original_data_dir)
        validation.check_parquets()

        output_path = os.path.join(root_dir,'outputs/dummy_ValidationCT_complete.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"status": "ok"}, f)

    except Exception as e:
        validation.logger.error(f"Failed to complete the data validation pipeline: {e}")
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()