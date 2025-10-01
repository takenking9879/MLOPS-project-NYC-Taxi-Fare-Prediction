# src/model/train_xgboost.py
import os, json, gc
from typing import List
import joblib
import pandas as pd
import numpy as np
from copy import deepcopy
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import re
from src.utils import create_logger, BaseUtils

logger = create_logger("model_retraining", "logs/retraining.log")


class RetrainingCT(BaseUtils):
    def __init__(self, params_path: str, processed_osrm_dir: str, models_dir: str, output_dir: str = None):
        super().__init__(logger=logger, params_path=params_path)
        self.processed_osrm_dir = processed_osrm_dir
        all_params = self.load_params()["continuous_training"]
        self.retrain_option = all_params["retrain"]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = None
        self.target_col = all_params["model_retraining"]["target"]
        self.model_path = self.get_latest_version(models_dir)

    def get_latest_version(self,base_dir: str) -> str:
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        versions = [(d, int(re.sub(r"\D", "", d))) for d in dirs if re.match(r"v\d+", d)]
        if not versions:
            return None

        # Carpeta más grande
        latest_name, latest_num = max(versions, key=lambda x: x[1])
        
        return os.path.join(base_dir, latest_name, "final_model.pkl")

    def retrain(self) -> None:
        try:
            if not self.retrain_option:
                return None
            # =================== 1. Load model ===================
            self.logger.info(f"Cargando modelo desde {self.model_path}")
            model: xgb.XGBRegressor = joblib.load(self.model_path)

            # =================== 2. Load data ===================
            train_path = os.path.join(self.processed_osrm_dir, "train_processed_osrm.parquet")
            train_df = pd.read_parquet(train_path)

            for c in ['Real_time', 'Real_distance']:
                if c in train_df.columns:
                    del train_df[c]

            if self.target_col == 'Fare_Amt' and 'Total_Amt' in train_df.columns:
                del train_df['Total_Amt']
            elif self.target_col == 'Total_Amt' and 'Fare_Amt' in train_df.columns:
                del train_df['Fare_Amt']

            X_train, y_train = train_df.drop(columns=[self.target_col]), train_df[self.target_col]
            del train_df
            # =================== 3. Retrain ===================
            self.logger.info("Reentrenando modelo con train...")
            feature_names = X_train.columns
            model.fit(X_train, y_train) #Aqui iba el xgb_model=model.get_booster()
            del y_train
            del X_train

            fi = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_
            }).sort_values(by="importance", ascending=False)

            fi_path = os.path.join(self.output_dir, "feature_importances.csv")
            fi.to_csv(fi_path, index=False)
            del fi

            self.logger.info(f"Importancias guardadas en {fi_path}")
            self.model = model

            # =================== 6. Save retrained model ===================
            retrained_path = os.path.join(self.output_dir, "final_model.pkl")
            joblib.dump(model, retrained_path)
            gc.collect()
            self.logger.info(f"Modelo reentrenado guardado en {retrained_path}")
        except Exception as e:
            self.logger.error(f"Error en retrain: {e}")
            raise

    def evaluate(self) -> None:
        try:
            if not self.retrain_option:
                return None
            val_path = os.path.join(self.processed_osrm_dir, "val_processed_osrm.parquet")

            val_df = pd.read_parquet(val_path)
            
            for c in ['Real_time', 'Real_distance']:
                if c in val_df.columns:
                    del val_df[c]

            if self.target_col == 'Fare_Amt' and 'Total_Amt' in val_df.columns:
                del val_df['Total_Amt']
            elif self.target_col == 'Total_Amt' and 'Fare_Amt' in val_df.columns:
                del val_df['Fare_Amt']

            X_val, y_val = val_df.drop(columns=[self.target_col]), val_df[self.target_col]

            # =================== 5. Evaluate ===================
            preds = self.model.predict(X_val)
            del X_val

            metrics = {
                "rmse": float(root_mean_squared_error(y_val, preds)),
                "r2": float(r2_score(y_val, preds)),
                "mae": float(mean_absolute_error(y_val, preds)),
                "mape": float(np.mean(np.abs((y_val - preds) / y_val)) * 100),
                "medae": float(median_absolute_error(y_val, preds))
            }

            metrics_path = os.path.join(self.output_dir, "val_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            gc.collect()
            self.logger.info(f"Métricas guardadas en {metrics_path}")
        except Exception as e:
            self.logger.error(f"Error en retrain_and_evaluate: {e}")
            raise

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    params = os.path.join(root, "params.yaml")
    processed_dir = os.path.join(root, "new_data/processed_osrm")
    output_dir = os.path.join(root, "new_model/")
    models_dir = os.path.join(root, "flask_app/models")

    Trainer = RetrainingCT(params, processed_dir,models_dir, output_dir)
    Trainer.retrain()
    Trainer.evaluate()
    gc.collect()

if __name__ == "__main__":
    main()
