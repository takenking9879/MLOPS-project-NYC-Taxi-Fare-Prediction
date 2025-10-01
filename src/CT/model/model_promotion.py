from src.utils import BaseUtils, create_logger, guardar_version_parquets
import joblib
import pandas as pd
import os
import json
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import xgboost as xgb
import re
import gc
import numpy as np
import shutil

logger = create_logger("model_promotion", "logs/model_promotion.log")

class ModelPromotionCT(BaseUtils):
    def __init__(self, params_path: str,processed_osrm_dir: str,  new_model_dir: str, models_dir: str = None):
        super().__init__(logger=logger, params_path=params_path)
        all_params = self.load_params()["continuous_training"]
        self.retrain_option = all_params["retrain"]
        self.target_col = all_params["model_promotion"].get("target", "Fare_Amt")
        self.metric_to_compare = all_params["model_promotion"].get("metric_to_compare", "rmse").lower()
        self.processed_osrm_dir = processed_osrm_dir
        self.new_model_dir = new_model_dir
        self.old_model_path = self.get_latest_version(models_dir)
        self.new_version = False

    def get_latest_version(self,base_dir: str) -> str:
        try:
            if not self.retrain_option:
                    return ""
            dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            versions = [(d, int(re.sub(r"\D", "", d))) for d in dirs if re.match(r"v\d+", d)]
            if not versions:
                return None

            # Carpeta más grande
            latest_name, latest_num = max(versions, key=lambda x: x[1])
            
            # Nueva carpeta (siguiente versión)
            self.new_folder = os.path.join(base_dir, f"v{latest_num + 1}")
            
            return os.path.join(base_dir, latest_name, "final_model.pkl")
        except Exception as e:
            self.logger.error(f"Hubo un error obteniendo la última versión {e}")
            raise

    def load_val_metrics(self) -> None:
        try:
            if not self.retrain_option:
                return None
            # Ruta donde se guarda tu json
            new_metrics_path = os.path.join(self.new_model_dir, "val_metrics.json")
            with open(new_metrics_path, "r") as f:
                metrics = json.load(f)
            self.new_model_metrics = metrics
        except Exception as e:
            self.logger.error(f"Hubo un error cargando las métricas del modelo reentrenado {e}")

    def evaluate(self, split: str = "val") -> None:
        """
        Evalúa el modelo en el split indicado ('val' o 'test') y guarda las métricas correspondientes.
        """
        try:
            if not self.retrain_option:
                return None
            if not self.new_version and split == "test":
                return None
            # =================== 1. Cargar modelo ===================
            split = split.lower()

            if split == "val":
                 model: xgb.XGBRegressor = joblib.load(self.old_model_path)
                 self.logger.info(f"Cargando modelo desde {self.old_model_path}")
            if split == "test":
                new_model_path = os.path.join(self.new_model_dir, "final_model.pkl")
                model: xgb.XGBRegressor = joblib.load(new_model_path)
                self.logger.info(f"Cargando modelo desde {new_model_path}")

            # =================== 2. Cargar datos ===================
            if split == "val":
                data_path = os.path.join(self.processed_osrm_dir, "val_processed_osrm.parquet")
            elif split == "test":
                data_path = os.path.join(self.processed_osrm_dir, "test_processed_osrm.parquet")
            else:
                raise ValueError(f"Split '{split}' no reconocido. Usa 'val' o 'test'.")

            df = pd.read_parquet(data_path)

            # =================== 3. Limpiar columnas ===================
            for c in ['Real_time', 'Real_distance']:
                if c in df.columns:
                    del df[c]

            if self.target_col == 'Fare_Amt' and 'Total_Amt' in df.columns:
                del df['Total_Amt']
            elif self.target_col == 'Total_Amt' and 'Fare_Amt' in df.columns:
                del df['Fare_Amt']

            X, y = df.drop(columns=[self.target_col]), df[self.target_col]

            # =================== 4. Predicciones y métricas ===================
            preds = model.predict(X)
            del X

            metrics = {
                "rmse": float(root_mean_squared_error(y, preds)),
                "r2": float(r2_score(y, preds)),
                "mae": float(mean_absolute_error(y, preds)),
                "mape": float(np.mean(np.abs((y - preds) / y)) * 100),
                "medae": float(median_absolute_error(y, preds))
            }

            # =================== 5. Guardar métricas ===================
            if split == "val":
                self.old_model_metrics = metrics
                self.logger.info(f"Métricas del split '{split}' calculadas correctamente")
            elif split == "test":
                self.test_metrics = metrics
                os.makedirs(self.new_folder, exist_ok=True)
                test_path = os.path.join(self.new_folder, "test_metrics.json")
                with open(test_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                self.logger.info(f"Métricas del split '{split}' guardadas en {test_path}")

        except Exception as e:
            self.logger.error(f"Error evaluando split '{split}': {e}")
            if hasattr(self, "new_folder") and os.path.exists(self.new_folder) and split == "test":
                shutil.rmtree(self.new_folder)
                self.logger.info(f"Se eliminó la carpeta incompleta {self.new_folder}")
            raise


    def compare_performances(self) -> None:
        """
        Compara RMSE (o la métrica seleccionada) entre el nuevo modelo y el viejo.
        """
        try:
            if not self.retrain_option:
                return None
            if self.metric_to_compare not in ["rmse", "mae", "mape", "medae"]:
                raise ValueError(
                    f"La métrica {self.metric_to_compare} no es válida, usa {['rmse', 'mae', 'mape', 'medae']}"
                )

            new_val = self.new_model_metrics[self.metric_to_compare]
            old_val = self.old_model_metrics[self.metric_to_compare]

            improved = "improved ✅" if new_val < old_val else "worse ❌"

            # Solo copia si el nuevo modelo es mejor
            if new_val < old_val:
                shutil.copytree(self.new_model_dir, self.new_folder, dirs_exist_ok=True)
                self.new_version = True


            self.logger.info(
                f"Retrained model's {self.metric_to_compare.upper()}: {new_val:.3f} \n"
                f"Previous model's {self.metric_to_compare.upper()}: {old_val:.3f} \n"
                f"Retrained model's Performance: {improved}"
            )

        except Exception as e:
            self.logger.error(f"Error comparando {self.metric_to_compare.upper()}: {e}")
            if hasattr(self, "new_folder") and os.path.exists(self.new_folder):
                try:
                    shutil.rmtree(self.new_folder)
                    self.logger.info(f"Se eliminó la carpeta incompleta {self.new_folder}")
                except Exception as e:
                    self.logger.error(f"No se pudo eliminar {self.new_folder}: {e}")
            raise
    
    def update_logs(self, root: str):

        if not self.retrain_option:
                return None
        json_path = os.path.join(root, "flask_app/models/parquet_logs.json")
        original_path = os.path.join(root, "new_data/original")
        version = os.path.basename(self.new_folder)
        if self.new_version:
            guardar_version_parquets(None,original_path ,json_path, version)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    params = os.path.join(root, "params.yaml")
    processed_dir = os.path.join(root, "new_data/processed_osrm")
    new_model_dir = os.path.join(root, "new_model/")
    models_dir = os.path.join(root, "flask_app/models/")

    promotion = ModelPromotionCT(params, processed_dir, new_model_dir, models_dir)
    promotion.load_val_metrics()
    promotion.evaluate("val")
    promotion.compare_performances()
    promotion.evaluate("test")

    promotion.update_logs(root)


    gc.collect()

if __name__ == "__main__":
    main()