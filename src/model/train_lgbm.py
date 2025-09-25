# src/model/train_lgbm.py
import os, time, json, gc
from typing import Dict, Any, List
import importlib
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

from src.utils import create_logger, BaseUtils

logger = create_logger("train_lgbm", "logs/train_lgbm.log")

class TrainLGBM(BaseUtils):
    def __init__(self, params_path: str, processed_dir: str, output_dir: str = None):
        super().__init__(logger=logger, params_path=params_path)
        self.processed_dir = processed_dir
        self.params = self.load_params()
        self.output_dir = output_dir or self.params.get("model_building", {}).get("output_models_dir", "models/saved_models")
        os.makedirs(self.output_dir, exist_ok=True)

    def _model_factory(self, class_path: str, params: Dict[str, Any]):
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**params)

    def _save_fi(self, feature_names: List[str], vals, key: str, ts: str):
        fi_df = pd.DataFrame({"feature": feature_names, "importance": vals})
        fi_df = fi_df.sort_values("importance", ascending=False)
        p = os.path.join(self.output_dir, f"{key}_{ts}_feature_importances.csv")
        fi_df.to_csv(p, index=False)
        return p

    def run(self):
        cfg = self.params.get("model_building", {})
        train_path = cfg.get("train_path")
        target = cfg.get("target", "Fare_Amt")
        model_cfg = cfg.get("models", {}).get("lgbm", {})
        ts = "model"

        logger.info("Cargando parquet para LGBM...")
        df = self.load_parquet(train_path, columns=None)

        # drops
        for c in ['Real_time', 'Real_distance']:
            if c in df.columns:
                del df[c]
        if target == 'Fare_Amt' and 'Total_Amt' in df.columns:
            del df['Total_Amt']
        elif target == 'Total_Amt' and 'Fare_Amt' in df.columns:
            del df['Fare_Amt']

        if target not in df.columns:
            raise KeyError(f"Target {target} no encontrado")

        y = df[target]
        X = df.drop(columns=[target])
        feature_names = X.columns.tolist()
        del df

        # init and train
        class_path = model_cfg.get("class", "lightgbm.LGBMRegressor")
        params = model_cfg.get("params", {})
        model = self._model_factory(class_path, params)
        logger.info("Entrenando LGBM...")
        model.fit(X, y, callbacks=[lgb.log_evaluation(25)])

        model_path = os.path.join(self.output_dir, f"lgbm_{ts}.pkl")
        joblib.dump(model, model_path)

        # feature importances
        if hasattr(model, "feature_importances_"):
            fi_path = self._save_fi(feature_names, model.feature_importances_, "lgbm", ts)
        else:
            fi_path = None

        meta = {"model_key": "lgbm", "model_class": class_path, "params": params, "feature_importances_file": fi_path, "model_path": model_path}
        meta_path = os.path.splitext(model_path)[0] + "_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("LGBM guardado en %s", model_path)

        # liberar
        try:
            del df, X, y, model
        except:
            pass
        gc.collect()

if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    params = os.path.join(root, "params.yaml")
    processed_dir = os.path.join(root, "data/processed_osrm")
    out = os.path.join(root, "models/saved_models")
    Trainer = TrainLGBM(params, processed_dir, out)
    Trainer.run()
    gc.collect()
