# src/model/train_stacking_manual.py
import os, time, gc, json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.utils import create_logger, BaseUtils

logger = create_logger("train_stacking_manual", "logs/train_stacking_manual.log")

try:
    import xgboost as xgb
except Exception:
    xgb = None
    logger.warning("xgboost no disponible")

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None
    logger.warning("catboost no disponible")


class TrainStackingManual(BaseUtils):
    def __init__(self, params_path: str, processed_dir: str, output_dir: str = None):
        super().__init__(logger=logger, params_path=params_path)
        self.processed_dir = processed_dir
        self.params = self.load_params()
        self.output_dir = output_dir or self.params.get("model_building", {}).get("output_models_dir", "models/saved_models")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Output dir: %s", self.output_dir)

    def _save_coefs(self, coef_dict, ts="stacking"):

        # CSV
        coef_df = pd.DataFrame(coef_dict.items(), columns=["feature", "coef"])
        coef_path_csv = os.path.join(self.output_dir, f"{ts}_coefs.csv")
        coef_df.to_csv(coef_path_csv, index=False)
        logger.info("Coeficientes guardados en CSV: %s", coef_path_csv)

        return coef_path_csv

    def run(self):
        start_all = time.time()
        cfg = self.params.get("model_building", {})
        train_path = cfg.get("train_path")
        target = cfg.get("target", "Fare_Amt")
        meta_frac = float(cfg.get("meta_train_frac", 0.1))

        logger.info("Stacking manual: cargando datos (frac=%s) desde %s", meta_frac, train_path)
        t0 = time.time()
        try:
            df_meta = pd.read_parquet(train_path)
            if meta_frac < 1.0:
                df_meta = df_meta.sample(frac=meta_frac, random_state=42)
            logger.info("Datos cargados shape=%s", df_meta.shape)
        except Exception as e:
            logger.exception("Error cargando parquet")
            raise e
        logger.debug("Tiempo carga parquet: %.2f s", time.time() - t0)

        # eliminar columnas irrelevantes
        for c in ['Real_time', 'Real_distance']:
            if c in df_meta.columns:
                del df_meta[c]
        if target == 'Fare_Amt' and 'Total_Amt' in df_meta.columns:
            del df_meta['Total_Amt']
        elif target == 'Total_Amt' and 'Fare_Amt' in df_meta.columns:
            del df_meta['Fare_Amt']

        if target not in df_meta.columns:
            raise KeyError("target no encontrado")

        y_meta = df_meta[target]
        X_meta = df_meta.drop(columns=[target])
        del df_meta

        meta_features = pd.DataFrame(index=X_meta.index)

        # cargar modelos base
        base_dir = self.output_dir
        candidates = {
            "lightgbm": [p for p in os.listdir(base_dir) if p.startswith("lgbm_") and p.endswith(".pkl")],
            "xgboost": [p for p in os.listdir(base_dir) if p.startswith("xgboost_") and p.endswith(".pkl")],
            "catboost": [p for p in os.listdir(base_dir) if p.startswith("catboost_") and p.endswith(".cbm")]
        }
        logger.info("Modelos encontrados: %s", {k: len(v) for k, v in candidates.items()})

        for mk, files in candidates.items():
            if not files:
                logger.warning("No se encontró modelo base %s", mk)
                continue
            chosen = sorted(files)[-1]
            path = os.path.join(base_dir, chosen)
            logger.info("Cargando %s: %s", mk, path)
            t_start = time.time()

            preds = np.zeros(len(X_meta))
            try:
                if mk == "xgboost" and xgb is not None:
                    model = joblib.load(path)
                    preds = model.predict(X_meta)
                    del model

                elif mk == "lightgbm":
                    model = joblib.load(path)
                    preds = model.predict(X_meta)
                    del model

                elif mk == "catboost" and CatBoostRegressor is not None:
                    model = CatBoostRegressor()
                    model.load_model(path)
                    preds = model.predict(X_meta)
                    del model

            except Exception as e:
                logger.exception("Error cargando modelo base %s", mk)

            meta_features[f"{mk}_pred"] = preds
            logger.info("Predicciones de %s añadidas, tiempo %.2f s", mk, time.time() - t_start)
            gc.collect()

        if meta_features.shape[1] == 0:
            raise ValueError("No se generaron meta-features — verifica modelos base.")

        # meta-modelo lineal
        meta_model_cfg = cfg.get("meta_model", {})
        meta_class_str = meta_model_cfg.get("class", "sklearn.linear_model.LinearRegression")
        meta_params = meta_model_cfg.get("params", {})

        # crear instancia del meta-modelo dinámicamente
        components = meta_class_str.split(".")
        mod = __import__(".".join(components[:-1]), fromlist=[components[-1]])
        meta_model_class = getattr(mod, components[-1])
        meta_model = meta_model_class(**meta_params)

        logger.info("Entrenando meta-modelo %s sobre predicciones de base: %s", meta_class_str, meta_features.columns.tolist())
        t_meta = time.time()
        meta_model.fit(meta_features, y_meta)
        logger.info("Meta-modelo entrenado en %.2f s", time.time() - t_meta)

        # guardar coeficientes
        coef_dict = {"intercept": float(meta_model.intercept_)}
        for col, val in zip(meta_features.columns, meta_model.coef_):
            coef_dict[col] = float(val)
        self._save_coefs(coef_dict, ts="stacking_meta")

        # guardar modelo completo
        meta_model_path = os.path.join(self.output_dir, f"stacking_meta_model.pkl")
        joblib.dump(meta_model, meta_model_path)
        logger.info("Meta-modelo guardado en %s, total tiempo %.2f s", meta_model_path, time.time() - start_all)
        gc.collect()


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    params = os.path.join(root, "params.yaml")
    processed_dir = os.path.join(root, "data/processed_osrm")
    out = os.path.join(root, "models/saved_models")
    Trainer = TrainStackingManual(params, processed_dir, out)
    Trainer.run()
