# src/model/train_catboost.py
import os
import time
import json
import gc
import importlib
import logging
from typing import List, Dict, Any

import pandas as pd
import numpy as np

from src.utils import create_logger, BaseUtils

logger = create_logger("train_catboost", "logs/train_catboost.log")

try:
    from catboost import CatBoostRegressor, Pool
except Exception:
    CatBoostRegressor = None
    Pool = None


class TrainCatBoost(BaseUtils):
    def __init__(self, params_path: str, processed_dir: str, output_dir: str = None):
        super().__init__(logger=logger, params_path=params_path)
        self.processed_dir = processed_dir
        self.params = self.load_params()
        self.output_dir = (
            output_dir
            or self.params.get("model_building", {}).get("output_models_dir", "models/saved_models")
        )
        os.makedirs(self.output_dir, exist_ok=True)
        # aseguro folder logs (create_logger puede crear fichero, pero por si acaso)
        os.makedirs(os.path.dirname("logs/train_catboost.log"), exist_ok=True)

    def _save_fi(self, feature_names: List[str], vals, key: str, ts: str):
        fi_df = pd.DataFrame({"feature": feature_names, "importance": vals})
        fi_df = fi_df.sort_values("importance", ascending=False)
        p = os.path.join(self.output_dir, f"{key}_{ts}_feature_importances.csv")
        fi_df.to_csv(p, index=False)
        return p

    def run(self):
        start_time = time.time()
        if CatBoostRegressor is None or Pool is None:
            logger.error("CatBoost no disponible en el entorno. Instala catboost para continuar.")
            raise ImportError("catboost no disponible")

        cfg = self.params.get("model_building", {})
        train_path = cfg.get("train_path")
        target = cfg.get("target", "Fare_Amt")
        model_cfg = cfg.get("models", {}).get("catboost", {})
        ts = "model"  # nombre fijo para DVC

        try:
            logger.info("Cargando parquet para CatBoost desde: %s", train_path)
            df = pd.read_parquet(train_path)
            logger.info("Dataset cargado shape=%s", df.shape)

            # drops de columnas pesadas si existen
            for c in ["Real_time", "Real_distance"]:
                if c in df.columns:
                    del df[c]

            if target == "Fare_Amt" and "Total_Amt" in df.columns:
                del df["Total_Amt"]
            elif target == "Total_Amt" and "Fare_Amt" in df.columns:
                del df["Fare_Amt"]

            if target not in df.columns:
                raise KeyError(f"Target {target} no encontrado en {train_path}")

            # separar X, y
            y = df[target]
            X = df.drop(columns=[target])
            logger.info("Shapes -> X: %s, y: %s", X.shape, y.shape)
            feature_names = X.columns.tolist()
            del df

            # instantiate model (importlib para respetar params)
            class_path = model_cfg.get("class", "catboost.CatBoostRegressor")
            params = dict(model_cfg.get("params", {}))  # copia

            logger.info("CatBoost params: %s", params)

            try:
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                model = cls(**params)
            except Exception as e:
                logger.warning("Fallo al instanciar con importlib (%s). Intentando CatBoostRegressor directamente. Error: %s", class_path, e)
                model = CatBoostRegressor(**params)

            # crear Pool (esto puede consumir memoria)
            logger.info("Creando Pool para CatBoost (esto puede consumir memoria)...")
            pool = Pool(X, y)

            # intentar fit; si falla por GPU, reintentar en CPU quitando claves GPU
            try:
                logger.info("Entrenando CatBoost...")
                model.fit(pool, verbose=False)
            except Exception as e:
                logger.warning("Fit falló con error: %s", e)
                # fallback: quitar parámetros GPU si existen y reintentar
                removed_gpu = False
                for k in ["task_type", "devices", "devices_list", "devices_per_host"]:
                    if k in params:
                        params.pop(k, None)
                        removed_gpu = True
                if removed_gpu:
                    logger.info("Reintentando fit sin parámetros GPU...")
                    try:
                        model = CatBoostRegressor(**params)
                        pool = Pool(X, y)
                        model.fit(pool,  verbose=25)
                    except Exception as e2:
                        logger.error("Segundo intento (CPU) falló: %s", e2, exc_info=True)
                        raise
                else:
                    logger.error("Fit falló y no había parámetros GPU para quitar. Error: %s", e, exc_info=True)
                    raise

            # Guardar modelo con nombre fijo para DVC
            model_path = os.path.join(self.output_dir, f"catboost_{ts}.cbm")
            logger.info("Guardando modelo en %s ...", model_path)
            model.save_model(model_path)

            # confirmar guardado
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Después de save_model no se encontró {model_path}")

            # feature importances
            fi_path = None
            if hasattr(model, "feature_importances_"):
                fi_path = self._save_fi(feature_names, model.feature_importances_, "catboost", ts)

            meta = {
                "model_key": "catboost",
                "model_class": class_path,
                "params": params,
                "feature_importances_file": fi_path,
                "model_path": model_path,
            }
            meta_path = os.path.splitext(model_path)[0] + "_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            logger.info("CatBoost guardado correctamente en %s (meta: %s)", model_path, meta_path)

        except Exception as ex:
            logger.exception("Error en train_catboost: %s", ex)
            # re-raise para que DVC detecte fallo
            raise
        finally:
            # limpieza y hint al SO
            try:
                del X, y, pool, model
            except Exception:
                pass
            gc.collect()
            try:
                time.sleep(1)
            except Exception:
                pass
            elapsed = time.time() - start_time
            logger.info("Fin train_catboost (tiempo: %.1f s)", elapsed)


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    params = os.path.join(root, "params.yaml")
    processed_dir = os.path.join(root, "data/processed_osrm")
    out = os.path.join(root, "models/saved_models")
    trainer = TrainCatBoost(params_path=params, processed_dir=processed_dir, output_dir=out)
    trainer.run()
    gc.collect()
