# src/model/train_xgboost.py
import os, json, gc
from typing import List
import joblib
import pandas as pd
import numpy as np
from copy import deepcopy
import xgboost as xgb

from src.utils import create_logger, BaseUtils

logger = create_logger("train_xgboost", "logs/train_xgboost.log")

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

class TrainXGBoost(BaseUtils):
    def __init__(self, params_path: str, processed_dir: str, output_dir: str = None):
        super().__init__(logger=logger, params_path=params_path)
        self.processed_dir = processed_dir
        self.params = self.load_params()
        self.output_dir = output_dir or self.params.get("model_building", {}).get("output_models_dir", "models/saved_models")
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_fi(self, feature_names: List[str], fi_array, key: str, ts: str):
        import pandas as pd
        fi_df = pd.DataFrame({"feature": feature_names, "importance": fi_array})
        fi_df = fi_df.sort_values("importance", ascending=False)
        p = os.path.join(self.output_dir, f"{key}_{ts}_feature_importances.csv")
        fi_df.to_csv(p, index=False)
        return p

    def run(self):
        if XGBRegressor is None:
            raise ImportError("xgboost no disponible")
        cfg = self.params.get("model_building", {})
        train_path = cfg.get("train_path")
        target = cfg.get("target", "Fare_Amt")
        model_cfg = cfg.get("models", {}).get("xgboost", {})
        ts = "model"

        logger.info("Cargando parquet para XGBoost...")
        df = pd.read_parquet(train_path)

        # drops
        for c in ['Real_time', 'Real_distance']:
            if c in df.columns:
                del df[c]
        if target == 'Fare_Amt' and 'Total_Amt' in df.columns:
            del df['Total_Amt']
        elif target == 'Total_Amt' and 'Fare_Amt' in df.columns:
            del df['Fare_Amt']

        y = df[target]
        X = df.drop(columns=[target])
        feature_names = X.columns.tolist()
        del df

        # crear modelo
        xgb_params = deepcopy(model_cfg.get("params", {}))
        model = XGBRegressor(**xgb_params)

        logger.info("Entrenando XGBRegressor con params: %s", xgb_params)
        model.fit(X, y, verbose = True)

        model_path = os.path.join(self.output_dir, f"xgboost_{ts}.pkl")
        joblib.dump(model, model_path)

        # feature importances
        fi_path = self._save_fi(feature_names, model.feature_importances_, "xgboost", ts)

        meta = {
            "model_key": "xgboost",
            "model_class": model_cfg.get("class"),
            "params": model_cfg.get("params"),
            "feature_importances_file": fi_path,
            "model_path": model_path
        }
        meta_path = os.path.splitext(model_path)[0] + "_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("XGBoost guardado en %s", model_path)

        # liberar
        try:
            del X, y, model
        except:
            pass
        gc.collect()

if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    params = os.path.join(root, "params.yaml")
    processed_dir = os.path.join(root, "data/processed_osrm")
    out = os.path.join(root, "models/saved_models")
    Trainer = TrainXGBoost(params, processed_dir, out)
    Trainer.run()
    gc.collect()
