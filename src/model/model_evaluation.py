# src/model/model_evaluation_regression.py
import os
import json
import time
import glob
import shutil
import logging
import gc
from typing import Dict, Any, Tuple, Optional, List


import pandas as pd
import numpy as np
import joblib
import yaml

# Dagshub + MLflow
import dagshub
import mlflow
import mlflow.sklearn

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

# CatBoost / XGBoost imports (opcionalmente disponibles)
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

# tus utilidades
from src.utils import create_logger, BaseUtils, guardar_version_parquets

# Logger conforme pediste
logger = create_logger("model_evaluation", "logs/model_evaluation.log")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # root_mean_squared_error devuelve ya la raíz (sklearn>=1.0), si no existe puedes reemplazar por np.sqrt(mse).
    try:
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        # fallback seguro
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def _medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(median_absolute_error(y_true, y_pred))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Evitar división por 0: reemplazamos denominadores pequeños por eps
    eps = 1e-8
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0


class ModelEvaluation(BaseUtils):
    """
    Evaluación de modelos para regresión usando MCDM (normalización + ponderación).
    """

    def __init__(self, params_path: str):
        super().__init__(logger=logger, params_path=params_path)
        self.params = self.load_params().get("model_evaluation", {})
        try:
            # inicializa dagshub como backend de mlflow (usa tu repo)
            dagshub.init(repo_owner='takenking9879', repo_name='MLOPS-project-NYC-Taxi-Fare-Prediction', mlflow=True)
            self._using_dagshub = True
            logger.info("Dagshub inicializado como backend MLflow")
        except Exception as e:
            self._using_dagshub = False
            logger.warning("No se pudo inicializar dagshub: %s", e)

        # si no usamos dagshub, intentamos setear tracking_uri si está en params
        self.tracking_uri = self.params.get("tracking_uri")
        self.experiment_name = self.params.get("experiment_name", "default_experiment")
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)


        # Paths
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        self.root_dir = root
        self.models_dir = os.path.abspath(self.params.get("models_dir", os.path.join(root, "models/saved_models")))
        self.val_path = os.path.abspath(self.params.get("val_path", os.path.join(root, "data/processed_osrm/val_processed_osrm.parquet")))
        self.test_path = os.path.abspath(self.params.get("test_path", os.path.join(root, "data/processed_osrm/test_processed_osrm.parquet")))

        # final model outputs
        self.final_dir = os.path.abspath(self.params.get("output_final_dir", os.path.join(root, "models/final_model")))
        self.metrics_dir = os.path.abspath(self.params.get("metrics_dir", os.path.join(root, "models/metrics")))
        os.makedirs(self.final_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # selection weights (se normalizan internamente)
        sel = self.params.get("selection", {})
        self.selection_mode = sel.get("mode", "overall")
        self.weights = sel.get("weights", {"rmse": 0.5, "mae": 0.25, "r2": 0.05, "mape": 0.1, "medae": 0.1})

        # TARGET configurable desde params (si no existe en params, se detecta del dataframe)
        self.target = self.params.get("target", None)

        # columnas a dropear (configurable)
        self.drop_columns = self.params.get("drop_columns", ["Real_time", "Real_distance"])

        logger.info(
            "ModelEvaluation initialized. models_dir=%s, val=%s, test=%s, mlflow=%s, experiment=%s, target=%s, drop_columns=%s",
            self.models_dir, self.val_path, self.test_path, ("dagshub" if self._using_dagshub else self.tracking_uri), self.experiment_name, self.target, self.drop_columns,
        )

    def _find_latest_model_for_key(self, key: str) -> Optional[str]:
        patterns = []
        if key == "xgboost":
            patterns = ["*xgboost*.pkl", "*xgboost*.model"]
        elif key in ("lightgbm", "lgbm"):
            patterns = ["*lgbm*.pkl", "*lightgbm*.pkl"]
        elif key == "catboost":
            patterns = ["*catboost*.cbm", "*catboost*.pkl"]
        elif key in ("stacking", "stacking_meta", "stacking_meta_model"):
            patterns = ["*stacking*.pkl", "*stacking_meta*.pkl", "*stacking_meta_model*.pkl"]
        else:
            patterns = [f"*{key}*"]

        candidates = []
        for pat in patterns:
            for c in glob.glob(os.path.join(self.models_dir, pat)):
                if not c.endswith("_meta.json"):  # <-- evitar archivos meta
                    candidates.append(c)

        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def _load_model_for_prediction(self, path: str):
        if path.lower().endswith(".cbm") and CatBoostRegressor is not None:
            m = CatBoostRegressor()
            m.load_model(path)
            return m
        try:
            model = joblib.load(path)
            return model
        except Exception:
            if xgb is not None:
                try:
                    booster = xgb.Booster()
                    booster.load_model(path)
                    return booster
                except Exception:
                    pass
            raise

    def _predict_with_model(self, model_obj, X: pd.DataFrame) -> np.ndarray:
        if xgb is not None and isinstance(model_obj, xgb.Booster):
            dmat = xgb.DMatrix(X.values, feature_names=X.columns.tolist())
            preds = model_obj.predict(dmat)
            del dmat
            return preds
        if CatBoostRegressor is not None and isinstance(model_obj, CatBoostRegressor):
            preds = model_obj.predict(X)
            return np.array(preds).reshape(-1,)
        if hasattr(model_obj, "predict"):
            preds = model_obj.predict(X)
            return np.array(preds).reshape(-1,)
        raise ValueError("Modelo no soportado para predict")

    def _compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {}
        metrics["rmse"] = _rmse(y_true, y_pred)
        metrics["mae"] = _mae(y_true, y_pred)
        metrics["r2"] = _r2(y_true, y_pred)
        metrics["mape"] = _mape(y_true, y_pred)
        metrics["medae"] = _medae(y_true, y_pred)
        return metrics

    def _log_metrics_and_save(self, model_key: str, model_path: str, metrics: Dict[str, float], meta_path: Optional[str]):
        model_basename = os.path.splitext(os.path.basename(model_path))[0]
        metrics_file = os.path.join(self.metrics_dir, f"{model_basename}_metrics.json")
        try:
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.warning("No se pudo guardar metrics localmente para %s: %s", model_key, e)

        model_params = {}
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as mf:
                    model_params = json.load(mf).get("params", {})
            except Exception:
                model_params = {}

        # Log a MLflow (Dagshub-backed si se inicializó)
        try:
            nested = mlflow.active_run() is not None
            with mlflow.start_run(run_name=f"eval_{model_key}_{int(time.time())}", nested=nested):
                try:
                    mlflow.log_param("model_key", model_key)
                    for k, v in model_params.items():
                        try:
                            mlflow.log_param(k, v)
                        except Exception:
                            pass
                except Exception:
                    pass
                for k, v in metrics.items():
                    try:
                        mlflow.log_metric(k, float(v))
                    except Exception:
                        pass
                mlflow.set_tag("model_key", model_key)
        except Exception as e:
            logger.warning("No se pudo loggear en MLflow para %s: %s", model_key, e)

    def evaluate_model(self, model_path: str, model_key: str, meta_path: Optional[str], X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        logger.info("Evaluando modelo %s desde %s", model_key, model_path)
        try:
            model_obj = self._load_model_for_prediction(model_path)
        except Exception as e:
            logger.error("No se pudo cargar modelo %s: %s", model_path, e)
            return {}

        try:
            preds = self._predict_with_model(model_obj, X_val)
        except Exception as e:
            logger.error("Error predict %s: %s", model_key, e)
            preds = np.zeros(len(X_val), dtype=float)

        try:
            metrics = self._compute_all_metrics(y_val.values, preds)
        except Exception as e:
            logger.error("Error calculando métricas para %s: %s", model_key, e)
            metrics = {}

        try:
            self._log_metrics_and_save(model_key, model_path, metrics, meta_path)
        except Exception as e:
            self.logger = logger
            logger.warning("Error guardando/loggeando metrics para %s: %s", model_key, e)

        try:
            del model_obj, preds
            gc.collect()
        except Exception:
            pass

        return metrics

    def _copy_feature_importances_for_final(self, selected_model_path: str, final_name: str = "final_model_feature_importances.csv") -> Optional[str]:
        try:
            meta_path = os.path.splitext(selected_model_path)[0] + "_meta.json"
            fi_src = None
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as mf:
                        meta = json.load(mf)
                    fi_file = meta.get("feature_importances_file")
                    if fi_file:
                        if os.path.isabs(fi_file) and os.path.exists(fi_file):
                            fi_src = fi_file
                        else:
                            cand = os.path.join(self.models_dir, os.path.basename(fi_file))
                            if os.path.exists(cand):
                                fi_src = cand
                            else:
                                cand2 = os.path.join(os.path.dirname(meta_path), os.path.basename(fi_file))
                                if os.path.exists(cand2):
                                    fi_src = cand2
                except Exception:
                    pass

            if fi_src is None:
                model_basename = os.path.splitext(os.path.basename(selected_model_path))[0]
                pattern = os.path.join(self.models_dir, f"{model_basename}*feature_importances.csv")
                matches = glob.glob(pattern)
                if matches:
                    matches = sorted(matches, key=lambda p: os.path.getmtime(p), reverse=True)
                    fi_src = matches[0]

            if fi_src and os.path.exists(fi_src):
                dst = os.path.join(self.final_dir, final_name)
                shutil.copyfile(fi_src, dst)
                logger.info("Feature importances copiado a %s", dst)
                return dst
            else:
                logger.warning("No se encontro CSV de feature importances para %s", selected_model_path)
                return None
        except Exception as e:
            logger.error("Error copiando feature importances: %s", e)
            return None

    def _promote_selected_model(self, selected_model_path: str, canonical_name: str, validation_metrics: Dict[str, Any]):
        try:
            if os.path.exists(self.final_dir):
                shutil.rmtree(self.final_dir)
            os.makedirs(self.final_dir, exist_ok=True)

            dest_model = os.path.join(self.final_dir, self.params.get("meta", {}).get("final_model_name", "final_model.pkl"))
            shutil.copyfile(selected_model_path, dest_model)
            logger.info("Modelo final copiado a %s", dest_model)

            fi_dst = self._copy_feature_importances_for_final(
                selected_model_path,
                final_name=self.params.get("meta", {}).get("final_feature_importances_name", "final_model_feature_importances.csv"),
            )

            try:
                val_file = os.path.join(self.final_dir, f"{canonical_name}_validation_metrics.json")
                with open(val_file, "w") as f:
                    json.dump(validation_metrics, f, indent=2)
                logger.info("Validation metrics guardadas en %s", val_file)
            except Exception as e:
                logger.warning("No se pudieron guardar validation metrics: %s", e)

            info = {
                "final_model_name": canonical_name,
                "selected_model_source": os.path.abspath(selected_model_path),
                "promoted_destination": os.path.abspath(dest_model),
                "feature_importances_copied": os.path.abspath(fi_dst) if fi_dst else None,
                "validation_metrics_saved": os.path.abspath(val_file) if os.path.exists(val_file) else None,
                "timestamp": int(time.time()),
            }
            info_path = os.path.join(self.final_dir, "final_model_info.json")
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
            logger.info("Final model info escrito en %s", info_path)
        except Exception as e:
            logger.error("Error promoviendo modelo final: %s", e)

    # -------------------- NUEVA FUNCIÓN: MCDM / Normalización + ponderación --------------------
    def _compute_mcdm_scores(self, results: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, float, Dict[str, float], Optional[str]]]:
        """
        Aplica el método: normalización (min-max por métrica), ponderación (self.weights),
        y devuelve lista de tuplas (key, model_path, score, metrics, meta_path).
        Score: menor = mejor.
        """
        metrics_order = ["rmse", "mae", "r2", "mape", "medae"]
        keys = list(results.keys())
        n_models = len(keys)
        M = np.full((n_models, len(metrics_order)), np.nan, dtype=float)

        # construir matriz M
        for i, k in enumerate(keys):
            mets = results[k]["metrics"]
            for j, mname in enumerate(metrics_order):
                M[i, j] = float(mets.get(mname, np.nan))

        # normalizar por columna (min-max) y convertir a escala donde 0 = mejor, 1 = peor
        norms = np.zeros_like(M)
        for j, mname in enumerate(metrics_order):
            col = M[:, j]
            finite = np.isfinite(col)
            if finite.sum() == 0:
                norms[:, j] = 0.5  # neutral si no hay datos
                continue
            mn = np.nanmin(col)
            mx = np.nanmax(col)
            if mx == mn:
                # todos iguales -> neutro
                norms[:, j] = 0.5
            else:
                if mname in ("rmse", "mae", "mape", "medae"):
                    # lower is better -> 0 best
                    norms[:, j] = (col - mn) / (mx - mn)
                else:
                    # higher is better (r2) -> invertimos para que 0 sea mejor
                    norms[:, j] = (mx - col) / (mx - mn)
                # clamp y reemplazar NaN por 0.5
                norms[:, j] = np.where(np.isfinite(norms[:, j]), np.clip(norms[:, j], 0.0, 1.0), 0.5)

        # preparar vector de pesos normalizados
        weights_vec = np.array([self.weights.get(m, 0.0) for m in metrics_order], dtype=float)
        if weights_vec.sum() <= 0:
            # pesos por defecto uniformes si no hay
            weights_vec = np.ones_like(weights_vec) / len(weights_vec)
        else:
            weights_vec = weights_vec / weights_vec.sum()

        # calcular scores (menor mejor)
        scores = (norms * weights_vec).sum(axis=1)

        # empaquetar
        out = []
        for i, k in enumerate(keys):
            out.append((k, results[k]["model_path"], float(scores[i]), results[k]["metrics"], results[k].get("meta_path")))
        # ordenar por score ascendente (menor = mejor)
        out = sorted(out, key=lambda x: x[2])
        # también loggear la tabla de normalización (útil para debug)
        try:
            norm_df = pd.DataFrame(norms, index=keys, columns=metrics_order)
            norm_df["score"] = scores
            logger.info("Tabla normalizada por métrica:\n%s", norm_df.to_string())
        except Exception:
            pass
        return out

    # -------------------- FIN MCDM --------------------

    def evaluate_all_and_select(self):
        logger.info("Cargando validation desde %s", self.val_path)
        try:
            df_val = pd.read_parquet(self.val_path)
        except Exception as e:
            logger.error("No se pudo cargar validation: %s", e)
            raise

        # Determinar target
        if self.target and self.target in df_val.columns:
            target_col = self.target
        else:
            if self.target and self.target not in df_val.columns:
                logger.warning("Target especificado en params (%s) no está en validation. Se usará fallback.", self.target)
            target_col = "Fare_Amt" if "Fare_Amt" in df_val.columns else df_val.columns[-1]

        # Drops importantes
        for c in self.drop_columns:
            if c in df_val.columns:
                del df_val[c]

        # Eliminar columna complementaria al target
        if target_col == 'Fare_Amt' and 'Total_Amt' in df_val.columns:
            del df_val['Total_Amt']
        elif target_col == 'Total_Amt' and 'Fare_Amt' in df_val.columns:
            del df_val['Fare_Amt']

        y_val = df_val[target_col]
        X_val = df_val.drop(columns=[target_col])
        logger.info("Validation shape X=%s, y=%s (target=%s)", X_val.shape, y_val.shape, target_col)

        try:
            del df_val
            gc.collect()
        except Exception:
            pass

        canonical_keys = ["xgboost", "lightgbm", "catboost", "stacking"]
        results: Dict[str, Dict[str, Any]] = {}
        base_preds = {}

        # ---- evaluar modelos base ----
        for key in ["lightgbm", "xgboost", "catboost"]:
            model_path = self._find_latest_model_for_key(key)
            if model_path is None:
                logger.warning("No se encontró modelo para key=%s", key)
                continue
            meta_path = os.path.splitext(model_path)[0] + "_meta.json"
            metrics = self.evaluate_model(model_path, key, meta_path if os.path.exists(meta_path) else None, X_val, y_val)
            if metrics:
                results[key] = {
                    "model_path": model_path,
                    "meta_path": meta_path if os.path.exists(meta_path) else None,
                    "metrics": metrics,
                }
                # guardar predicciones para stacking
                model_obj = self._load_model_for_prediction(model_path)
                base_preds[key + "_pred"] = self._predict_with_model(model_obj, X_val)
                del model_obj
                gc.collect()

        # ---- evaluar stacking ----
        if "stacking" in canonical_keys and base_preds:
            X_meta = pd.DataFrame(base_preds)
            stacking_model_path = self._find_latest_model_for_key("stacking")
            if stacking_model_path:
                stacking_metrics = self.evaluate_model(stacking_model_path, "stacking", None, X_meta, y_val)
                results["stacking"] = {
                    "model_path": stacking_model_path,
                    "meta_path": None,
                    "metrics": stacking_metrics,
                }

        if not results:
            logger.error("No se evaluaron modelos. Abortando.")
            return

        # ---- aplicar MCDM ----
        scored = self._compute_mcdm_scores(results)
        best_key, best_path, best_score, best_metrics, best_meta = scored[0]
        logger.info("Mejor modelo seleccionado por MCDM: %s (score=%.6f)", best_key, best_score)

        # promover final
        try:
            self._promote_selected_model(best_path, best_key, best_metrics)
        except Exception as e:
            logger.error("Error promoviendo modelo seleccionado: %s", e)

        # ---- evaluar en test ----
        if os.path.exists(self.test_path):
            logger.info("Evaluando modelo final en test: %s", self.test_path)
            df_test = pd.read_parquet(self.test_path)

            # target coherente
            if self.target and self.target in df_test.columns:
                tcol = self.target
            elif target_col in df_test.columns:
                tcol = target_col
            elif "Fare_Amt" in df_test.columns:
                tcol = "Fare_Amt"
            else:
                tcol = df_test.columns[-1]

            for c in self.drop_columns:
                if c in df_test.columns:
                    del df_test[c]

            if tcol == 'Fare_Amt' and 'Total_Amt' in df_test.columns:
                del df_test['Total_Amt']
            elif tcol == 'Total_Amt' and 'Fare_Amt' in df_test.columns:
                del df_test['Fare_Amt']

            y_test = df_test[tcol]
            X_test = df_test.drop(columns=[tcol])

            promoted_model_path = os.path.join(self.final_dir, self.params.get("meta", {}).get("final_model_name", "final_model.pkl"))
            if not os.path.exists(promoted_model_path):
                promoted_model_path = best_path

            try:
                model_obj = self._load_model_for_prediction(promoted_model_path)

                # stacking test
                if best_key == "stacking":
                    base_preds_test = {}
                    for base_key in ["xgboost", "lightgbm", "catboost"]:
                        model_path = self._find_latest_model_for_key(base_key)
                        if model_path:
                            base_model = self._load_model_for_prediction(model_path)
                            base_preds_test[base_key + "_pred"] = self._predict_with_model(base_model, X_test)
                            del base_model
                            gc.collect()
                    X_test_meta = pd.DataFrame(base_preds_test)
                    preds_test = self._predict_with_model(model_obj, X_test_meta)
                else:
                    preds_test = self._predict_with_model(model_obj, X_test)

                test_metrics = self._compute_all_metrics(y_test.values, preds_test)
                test_metrics_file = os.path.join(self.final_dir, f"{best_key}_test_metrics.json")
                with open(test_metrics_file, "w") as f:
                    json.dump(test_metrics, f, indent=2)
                logger.info("Test metrics guardadas en %s", test_metrics_file)

                try:
                    nested = mlflow.active_run() is not None
                    with mlflow.start_run(run_name=f"final_test_{best_key}_{int(time.time())}", nested=nested):
                        mlflow.log_param("selected_model", best_key)
                        for k, v in test_metrics.items():
                            try:
                                mlflow.log_metric(f"test_{k}", float(v))
                            except Exception:
                                pass
                        mlflow.set_tag("final_selected", best_key)
                except Exception as e:
                    logger.warning("No se pudieron loggear test metrics a MLflow: %s", e)

                del model_obj, preds_test, X_test, y_test
                gc.collect()
            except Exception as e:
                logger.error("No se pudo evaluar modelo final en test: %s", e)
        else:
            logger.info("No existe test_path, se omite evaluación en test.")

        # ---- guardar resumen local ----
        try:
            summary = {k: v["metrics"] for k, v in results.items()}
            summary_path = os.path.join(self.metrics_dir, "metrics_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Metrics summary guardado en %s", summary_path)
        except Exception as e:
            logger.warning("No se pudo guardar summary: %s", e)



def main():
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        params_path = os.path.join(root, "params.yaml")
        json_path = os.path.join(root, "flask_app/models/parquet_logs.json")
        original_path = os.path.join(root, "data/original")
        evaluator = ModelEvaluation(params_path)
        evaluator.evaluate_all_and_select()

        guardar_version_parquets(None, original_path ,json_path, "v1")
    except Exception as e:
        logger.error("Error en main model_evaluation: %s", e)
        raise


if __name__ == "__main__":
    main()
