# src/model/train_xgboost.py
import os
import json
import gc
import joblib
import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from typing import Optional, Tuple
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
    median_absolute_error,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.utils import create_logger, BaseUtils

logger = create_logger("model_retraining", "logs/retraining.log")

def compute_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict:
    # safe division for MAPE
    safe_y = np.where(y_true == 0, 1e-8, y_true)
    return {
        "rmse": float(root_mean_squared_error(y_true, preds)),
        "r2": float(r2_score(y_true, preds)),
        "mae": float(mean_absolute_error(y_true, preds)),
        "mape": float(np.mean(np.abs((y_true - preds) / safe_y)) * 100),
        "medae": float(median_absolute_error(y_true, preds)),
    }

class RetrainingCT(BaseUtils):
    def __init__(
        self,
        params_path: str,
        processed_osrm_dir: str,
        models_dir: str,
        output_dir: str = None,
    ):
        super().__init__(logger=logger, params_path=params_path)
        self.processed_osrm_dir = processed_osrm_dir
        all_params = self.load_params()["continuous_training"]
        self.retrain_option = all_params["retrain"]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.target_col = all_params["model_retraining"]["target"]
        self.model_path = self.get_latest_version(models_dir)

        # OOF / stacking params (from params.yaml or defaults)
        mr_params = all_params["model_retraining"]
        self.oof_folds = mr_params.get("oof_folds", 5)
        self.oof_shuffle = mr_params.get("oof_shuffle", True)
        self.oof_random_state = mr_params.get("oof_random_state", 42)
        self.meta_alpha = mr_params.get("meta_alpha", 0.5)

    def get_latest_version(self, base_dir: str) -> Optional[str]:
        if not os.path.isdir(base_dir):
            return None
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        versions = [(d, int(re.sub(r"\D", "", d))) for d in dirs if re.match(r"v\d+", d)]
        if not versions:
            return None
        latest_name, latest_num = max(versions, key=lambda x: x[1])
        return os.path.join(base_dir, latest_name, "final_model.pkl")

    def _load_train(self) -> Tuple[pd.DataFrame, pd.Series]:
        train_path = os.path.join(self.processed_osrm_dir, "train_processed_osrm.parquet")
        train_df = pd.read_parquet(train_path)

        for c in ["Real_time", "Real_distance"]:
            if c in train_df.columns:
                del train_df[c]

        if self.target_col == "Fare_Amt" and "Total_Amt" in train_df.columns:
            del train_df["Total_Amt"]
        elif self.target_col == "Total_Amt" and "Fare_Amt" in train_df.columns:
            del train_df["Fare_Amt"]

        X_train = train_df.drop(columns=[self.target_col])
        y_train = train_df[self.target_col]
        del train_df
        return X_train, y_train

    def _load_val(self) -> Tuple[pd.DataFrame, pd.Series]:
        val_path = os.path.join(self.processed_osrm_dir, "val_processed_osrm.parquet")
        val_df = pd.read_parquet(val_path)

        for c in ["Real_time", "Real_distance"]:
            if c in val_df.columns:
                del val_df[c]

        if self.target_col == "Fare_Amt" and "Total_Amt" in val_df.columns:
            del val_df["Total_Amt"]
        elif self.target_col == "Total_Amt" and "Fare_Amt" in val_df.columns:
            del val_df["Fare_Amt"]

        X_val = val_df.drop(columns=[self.target_col])
        y_val = val_df[self.target_col]
        del val_df
        return X_val, y_val

    # -------------------- Cambios clave --------------------
    # 1️⃣ Guardamos el stacking siempre con nombre consistente
    # 2️⃣ Evaluamos correctamente el old model solo si es distinto
    # 3️⃣ Evitamos copiar old y new pred si son idénticos (solo si hay modelo viejo real)

    def retrain(self) -> None:
        try:
            if not self.retrain_option:
                return None

            # ----------------- load old model if exists -----------------
            xgb_old: Optional[xgb.XGBRegressor] = None
            if self.model_path and os.path.exists(self.model_path):
                try:
                    xgb_old = joblib.load(self.model_path)
                    self.logger.info(f"Cargando modelo viejo desde {self.model_path}")
                except Exception as e:
                    self.logger.warning(f"No se pudo cargar xgb_old: {e}")
                    xgb_old = None
            else:
                self.logger.info("No se encontró modelo viejo; procediendo sin él.")

            # ----------------- load train data -----------------
            X_train, y_train = self._load_train()
            n_samples = X_train.shape[0]

            # ----------------- prepare OOF arrays -----------------
            oof_new = np.zeros(n_samples, dtype=float)
            oof_old = np.zeros(n_samples, dtype=float) if xgb_old is not None else None

            kf = KFold(n_splits=self.oof_folds, shuffle=self.oof_shuffle, random_state=self.oof_random_state)
            params = xgb_old.get_params() if xgb_old is not None else {}

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
                X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
                X_val_oof = X_train.iloc[val_idx]
                self.logger.info(
                                f"Fold {fold_idx}/{self.oof_folds} - "
                                f"train size: {len(train_idx)}, val size: {len(val_idx)}"
                            )

                if xgb_old is not None:
                    try:
                        oof_old[val_idx] = xgb_old.predict(X_val_oof)
                    except Exception:
                        oof_old[val_idx] = 0.0

                # entrenar nuevo fold con semilla diferente para diversificar
                xgb_fold = xgb.XGBRegressor(**params)
                xgb_fold.fit(X_tr, y_tr)
                oof_new[val_idx] = xgb_fold.predict(X_val_oof)

                del xgb_fold
                gc.collect()

            # ----------------- train meta-model -----------------
            meta_model = None
            if xgb_old is not None:
                self.logger.info("Entrenando meta-modelo (Ridge)...")
                stack_X = np.column_stack([oof_old, oof_new])
                stack_y = y_train.values
                meta_model = Ridge(alpha=self.meta_alpha)
                meta_model.fit(stack_X, stack_y)

            # ----------------- train final XGB -----------------
            self.logger.info("Entrenando XGB final sobre todo el train...")
            xgb_final = xgb.XGBRegressor(**params)
            xgb_final.fit(X_train, y_train)

            # ----------------- save models -----------------
            final_path = os.path.join(self.output_dir, "final_model.pkl")
            joblib.dump(xgb_final, final_path)

            if meta_model is not None:
                stacking_path = os.path.join(self.output_dir, "stacking.pkl")
                joblib.dump(meta_model, stacking_path)

            self.model = xgb_final
            gc.collect()

        except Exception as e:
            self.logger.error(f"Error en retrain: {e}")
            raise


    def evaluate(self) -> None:
        try:
            if not self.retrain_option:
                return None

            X_val, y_val = self._load_val()
            xgb_new = self.model

            # cargamos old solo si existe
            xgb_old = None
            if self.model_path and os.path.exists(self.model_path):
                try:
                    xgb_old = joblib.load(self.model_path)
                except Exception:
                    self.logger.warning("No se pudo cargar xgb_old en evaluate.")

            # cargamos stacking
            stacking_path = os.path.join(self.output_dir, "stacking.pkl")
            meta_model = joblib.load(stacking_path) if os.path.exists(stacking_path) else None

            preds_new = xgb_new.predict(X_val)
            preds_old = xgb_old.predict(X_val) if xgb_old is not None else None

            preds_stack = None
            if preds_old is not None and meta_model is not None:
                stack_input = np.column_stack([preds_old, preds_new])
                preds_stack = meta_model.predict(stack_input)

            results = {
                "new": compute_metrics(y_val.values, preds_new),
                "old": compute_metrics(y_val.values, preds_old) if preds_old is not None else None,
                "stacked": compute_metrics(y_val.values, preds_stack) if preds_stack is not None else None
            }

            metrics_path = os.path.join(self.output_dir, "val_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(results, f, indent=4)

            self.logger.info(f"Métricas guardadas en {metrics_path}")

        except Exception as e:
            self.logger.error(f"Error en evaluate: {e}")
            raise

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    params = os.path.join(root, "params.yaml")
    processed_dir = os.path.join(root, "new_data/processed_osrm")
    output_dir = os.path.join(root, "new_model/")
    models_dir = os.path.join(root, "flask_app/models")

    Trainer = RetrainingCT(params, processed_dir, models_dir, output_dir)
    Trainer.retrain()
    Trainer.evaluate()
    gc.collect()


if __name__ == "__main__":
    main()
