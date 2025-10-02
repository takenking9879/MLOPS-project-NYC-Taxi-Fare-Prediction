# model_promotion.py (ajustado: promueve y evalúa en test SOLO al mejor en val; guarda final_model.pkl + stacking.pkl)
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
    def __init__(self, params_path: str, processed_osrm_dir: str, new_model_dir: str, models_dir: str = None):
        super().__init__(logger=logger, params_path=params_path)
        all_params = self.load_params()["continuous_training"]
        self.evaluate_all_in_test = all_params["model_promotion"].get("evaluate_all_in_test", False)
        self.retrain_option = all_params["retrain"]
        self.target_col = all_params["model_promotion"].get("target", "Fare_Amt")
        self.metric_to_compare = all_params["model_promotion"].get("metric_to_compare", "rmse").lower()
        self.processed_osrm_dir = processed_osrm_dir
        self.new_model_dir = new_model_dir
        self.new_folder = None
        self.old_model_path = self.get_latest_version(models_dir)
        self.new_version = False

        # placeholders
        self.new_model_metrics_new = None
        self.new_model_metrics_stacked = None
        self.old_model_metrics = None
        self.best_on_val = None  # "new" or "stacked" if one wins
        self.test_metrics = {}

    def get_latest_version(self, base_dir: str) -> str:
        try:
            if not self.retrain_option:
                return ""
            if not os.path.isdir(base_dir):
                return None
            dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            versions = [(d, int(re.sub(r"\D", "", d))) for d in dirs if re.match(r"v\d+", d)]
            if not versions:
                return None

            latest_name, latest_num = max(versions, key=lambda x: x[1])

            # prepare the new folder path for the next version
            self.new_folder = os.path.join(base_dir, f"v{latest_num + 1}")

            return os.path.join(base_dir, latest_name, "final_model.pkl")
        except Exception as e:
            self.logger.error(f"Hubo un error obteniendo la última versión {e}")
            raise

    def load_val_metrics(self) -> None:
        """
        Lee val_metrics_new.json y val_metrics_stacked.json (si existen) dentro de new_model_dir.
        """
        try:
            if not self.retrain_option:
                return None

            new_metrics_path = os.path.join(self.new_model_dir, "val_metrics_new.json")
            stacked_metrics_path = os.path.join(self.new_model_dir, "val_metrics_stacked.json")

            self.new_model_metrics_new = None
            self.new_model_metrics_stacked = None

            if os.path.exists(new_metrics_path):
                with open(new_metrics_path, "r") as f:
                    self.new_model_metrics_new = json.load(f)
                self.logger.info(f"Cargadas métricas 'new' desde {new_metrics_path}")
            else:
                # fallback a viejo val_metrics.json si existe
                fallback = os.path.join(self.new_model_dir, "val_metrics.json")
                if os.path.exists(fallback):
                    with open(fallback, "r") as f:
                        fb = json.load(f)
                    # intenta mapear a "new" si archivo combinado
                    if isinstance(fb, dict) and "new" in fb and isinstance(fb["new"], dict):
                        self.new_model_metrics_new = fb["new"]
                        self.logger.info(f"Fallback: tomé 'new' de {fallback}")
                    else:
                        # si fb es directamente métricas del nuevo
                        self.new_model_metrics_new = fb
                        self.logger.info(f"Fallback: tomé métricas directas de {fallback}")

            if os.path.exists(stacked_metrics_path):
                with open(stacked_metrics_path, "r") as f:
                    self.new_model_metrics_stacked = json.load(f)
                self.logger.info(f"Cargadas métricas 'stacked' desde {stacked_metrics_path}")
            else:
                # fallback a combined file
                fallback = os.path.join(self.new_model_dir, "val_metrics.json")
                if os.path.exists(fallback):
                    with open(fallback, "r") as f:
                        fb = json.load(f)
                    if isinstance(fb, dict) and "stacked" in fb and isinstance(fb["stacked"], dict):
                        self.new_model_metrics_stacked = fb["stacked"]
                        self.logger.info(f"Fallback: tomé 'stacked' de {fallback}")

        except Exception as e:
            self.logger.error(f"Hubo un error cargando las métricas del modelo reentrenado: {e}")
            raise

    def evaluate(self, split: str = "val") -> None:
        """
        - split='val': evalúa el modelo viejo en val y guarda en self.old_model_metrics
        - split='test': si se promovió, evalúa al ganador en val (new o stacked)
                        y opcionalmente al otro modelo si evaluate_all_in_test=True.
                        Guarda test_metrics_{candidate}.json en la nueva carpeta.
        """
        try:
            if not self.retrain_option:
                return None

            split = split.lower()

            # dataset path
            if split == "val":
                data_path = os.path.join(self.processed_osrm_dir, "val_processed_osrm.parquet")
            elif split == "test":
                data_path = os.path.join(self.processed_osrm_dir, "test_processed_osrm.parquet")
            else:
                raise ValueError("split debe ser 'val' o 'test'")

            df = pd.read_parquet(data_path)
            for c in ['Real_time', 'Real_distance']:
                if c in df.columns:
                    del df[c]

            if self.target_col == 'Fare_Amt' and 'Total_Amt' in df.columns:
                del df['Total_Amt']
            elif self.target_col == 'Total_Amt' and 'Fare_Amt' in df.columns:
                del df['Fare_Amt']

            X, y = df.drop(columns=[self.target_col]), df[self.target_col]

            if split == "val":
                # evaluate only old model to obtain baseline for comparison
                if not self.old_model_path or not os.path.exists(self.old_model_path):
                    raise FileNotFoundError("No se encontró modelo viejo para evaluar en 'val'.")
                xgb_old = joblib.load(self.old_model_path)
                self.logger.info(f"Cargando modelo viejo desde {self.old_model_path} para 'val'")

                preds_old = xgb_old.predict(X)
                metrics_old = {
                    "rmse": float(root_mean_squared_error(y, preds_old)),
                    "r2": float(r2_score(y, preds_old)),
                    "mae": float(mean_absolute_error(y, preds_old)),
                    "mape": float(np.mean(np.abs((y - preds_old) / np.where(y == 0, 1e-8, y))) * 100),
                    "medae": float(median_absolute_error(y, preds_old))
                }
                self.old_model_metrics = metrics_old
                self.logger.info("Métricas del split 'val' (old model) calculadas correctamente")

            elif split == "test":
                if not self.new_version:
                    self.logger.info("No se promovió; se omite evaluación en 'test'.")
                    return None

                if self.best_on_val is None:
                    self.logger.warning("No se definió best_on_val al promover; omito evaluación en test.")
                    return None

                # definir candidatos a evaluar
                candidates = [self.best_on_val]
                if self.evaluate_all_in_test:
                    candidates = []
                    if self.new_model_metrics_new is not None:
                        candidates.append("new")
                    if self.new_model_metrics_stacked is not None:
                        candidates.append("stacked")

                for chosen in candidates:
                    if chosen == "new":
                        final_model_path = os.path.join(self.new_folder, "final_model.pkl")
                        if not os.path.exists(final_model_path):
                            final_model_path = os.path.join(self.new_model_dir, "final_model.pkl")
                        model_to_eval = joblib.load(final_model_path)
                        preds = model_to_eval.predict(X)

                    elif chosen == "stacked":
                        xgb_old = joblib.load(self.old_model_path)
                        xgb_new_path = os.path.join(self.new_folder, "final_model.pkl")
                        if not os.path.exists(xgb_new_path):
                            xgb_new_path = os.path.join(self.new_model_dir, "final_model.pkl")
                        xgb_new = joblib.load(xgb_new_path)

                        stacking_path = os.path.join(self.new_folder, "stacking.pkl")
                        if not os.path.exists(stacking_path):
                            stacking_path = os.path.join(self.new_folder, "meta_model.pkl")
                        if not os.path.exists(stacking_path):
                            stacking_path = os.path.join(self.new_model_dir, "stacking.pkl")
                        if not os.path.exists(stacking_path):
                            stacking_path = os.path.join(self.new_model_dir, "meta_model.pkl")
                        stacking_model = joblib.load(stacking_path)

                        preds_old = xgb_old.predict(X)
                        preds_new = xgb_new.predict(X)
                        stack_input = np.column_stack([preds_old, preds_new])
                        preds = stacking_model.predict(stack_input)

                    else:
                        raise RuntimeError(f"Candidato desconocido '{chosen}' para evaluación en test.")

                    metrics = {
                        "rmse": float(root_mean_squared_error(y, preds)),
                        "r2": float(r2_score(y, preds)),
                        "mae": float(mean_absolute_error(y, preds)),
                        "mape": float(np.mean(np.abs((y - preds) / np.where(y == 0, 1e-8, y))) * 100),
                        "medae": float(median_absolute_error(y, preds))
                    }

                    # guardar métricas por candidato
                    os.makedirs(self.new_folder, exist_ok=True)
                    test_path = os.path.join(self.new_folder, f"test_metrics_{chosen}.json")
                    with open(test_path, "w") as f:
                        json.dump(metrics, f, indent=4)
                    self.test_metrics[chosen] = metrics
                    self.logger.info(f"Métricas del split 'test' (candidate={chosen}) guardadas en {test_path}")

        except Exception as e:
            self.logger.error(f"Error evaluando split '{split}': {e}")
            # cleanup parcial
            if split == "test" and hasattr(self, "new_folder") and self.new_folder and os.path.exists(self.new_folder):
                try:
                    shutil.rmtree(self.new_folder)
                    self.logger.info(f"Se eliminó la carpeta incompleta {self.new_folder}")
                except Exception as ee:
                    self.logger.error(f"No se pudo eliminar {self.new_folder}: {ee}")
            raise


    def compare_performances(self) -> None:
        """
        Compara la métrica seleccionada entre old y new/stacked.
        - Si new o stacked mejora, promueve (copia new_model_dir -> new_folder).
        - Selecciona el ganador en validación (best_on_val).
        - Guarda promotion_choice.json en la nueva versión con RMSEs y la elección.
        """
        try:
            if not self.retrain_option:
                return None
            if self.metric_to_compare not in ["rmse", "mae", "mape", "medae"]:
                raise ValueError("Métrica inválida")

            if not hasattr(self, "old_model_metrics") or self.old_model_metrics is None:
                raise RuntimeError("old_model_metrics no está cargado. Ejecuta evaluate('val') primero.")

            old_val = self.old_model_metrics.get(self.metric_to_compare)
            rmse_old = self.old_model_metrics.get("rmse")

            # ensure val metrics were loaded
            if self.new_model_metrics_new is None or self.new_model_metrics_stacked is None:
                # try to load from files
                self.load_val_metrics()

            new_val = self.new_model_metrics_new.get(self.metric_to_compare) if self.new_model_metrics_new else None
            stacked_val = self.new_model_metrics_stacked.get(self.metric_to_compare) if self.new_model_metrics_stacked else None

            # also extract RMSE for logging/tracing
            rmse_new = self.new_model_metrics_new.get("rmse") if self.new_model_metrics_new else None
            rmse_stacked = self.new_model_metrics_stacked.get("rmse") if self.new_model_metrics_stacked else None

            # log RMSEs explicitly as requested
            def fmt(x):
                return f"{x:.6f}" if (x is not None) else "n/a"


            # decide improvements
            improved_new = (new_val is not None) and (old_val is not None) and (new_val < old_val)
            improved_stacked = (stacked_val is not None) and (old_val is not None) and (stacked_val < old_val)

            # choose best_on_val if any improved (prefer lower metric)
            chosen = None
            if improved_new and improved_stacked:
                chosen = "new" if (new_val <= stacked_val) else "stacked"
            elif improved_new:
                chosen = "new"
            elif improved_stacked:
                chosen = "stacked"
            else:
                chosen = None

            # log message including the metric_to_compare values and RMSEs
            log_msg = (
            f"Retrained model's {self.metric_to_compare.upper()}: {new_val:.4f}\n"
            f"Previous model's {self.metric_to_compare.upper()}: {old_val:.4f}\n"
            f"Stacked {self.metric_to_compare.upper()}: {stacked_val:.4f}\n"
            f"Selected based on val: {chosen if chosen else 'none'}"
            )

            self.logger.info("\n\n" + log_msg + "\n")

            if chosen is not None:
                # promote: copy new_model_dir -> new_folder
                if not self.new_folder:
                    raise RuntimeError("new_folder no está definido (no se encontró base para versionar).")
                shutil.copytree(self.new_model_dir, self.new_folder, dirs_exist_ok=True)
                self.new_version = True
                self.best_on_val = chosen

                # write promotion_choice.json with trace info (which won and RMSEs)
                promotion_choice = {
                    "chosen_on_val": chosen,
                    "metric": self.metric_to_compare,
                    "old_val": old_val,
                    "new_val": new_val,
                    "stacked_val": stacked_val,
                    "rmse_old": rmse_old,
                    "rmse_new": rmse_new,
                    "rmse_stacked": rmse_stacked
                }
                try:
                    with open(os.path.join(self.new_folder, "promotion_choice.json"), "w") as pf:
                        json.dump(promotion_choice, pf, indent=2)
                except Exception as e:
                    self.logger.warning(f"No se pudo escribir promotion_choice.json: {e}")

                self.logger.info(f"Se creó nueva versión y se eligió '{chosen}' como mejor en validación.")
            else:
                self.logger.info("Ninguna métrica mejoró respecto al modelo anterior. No se crea nueva versión.")

        except Exception as e:
            self.logger.error(f"Error comparando {self.metric_to_compare.upper()}: {e}")
            # cleanup incomplete folder
            if hasattr(self, "new_folder") and self.new_folder and os.path.exists(self.new_folder):
                try:
                    shutil.rmtree(self.new_folder)
                    self.logger.info(f"Se eliminó la carpeta incompleta {self.new_folder}")
                except Exception as ee:
                    self.logger.error(f"No se pudo eliminar {self.new_folder}: {ee}")
            raise

    def update_logs(self, root: str):
        if not self.retrain_option:
            return None
        json_path = os.path.join(root, "flask_app/models/parquet_logs.json")
        original_path = os.path.join(root, "new_data/original")
        version = os.path.basename(self.new_folder) if self.new_folder else None
        if self.new_version:
            guardar_version_parquets(None, original_path, json_path, version)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    params = os.path.join(root, "params.yaml")
    processed_dir = os.path.join(root, "new_data/processed_osrm")
    new_model_dir = os.path.join(root, "new_model/")
    models_dir = os.path.join(root, "flask_app/models/")

    promotion = ModelPromotionCT(params, processed_dir, new_model_dir, models_dir)
    promotion.load_val_metrics()
    promotion.evaluate("val")       # calcula old_model_metrics
    promotion.compare_performances()
    promotion.evaluate("test")      # si se promovió, evalúa SOLO el mejor y guarda test_metrics.json en la nueva carpeta
    promotion.update_logs(root)

    gc.collect()

if __name__ == "__main__":
    main()
