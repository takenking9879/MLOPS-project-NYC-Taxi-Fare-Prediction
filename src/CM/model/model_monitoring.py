# src/CM/monitoring_cm.py
import os
import shutil
import joblib
import pandas as pd
import numpy as np
from evidently import Report
from evidently import DataDefinition
from evidently import Dataset
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset
from evidently import Regression
from src.utils import create_logger
import re
import gc
from evidently.ui.workspace import CloudWorkspace
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="evidently")

logger = create_logger("continuous_monitoring", "logs/continuous_monitoring.log")
def delete_data(delete = False) -> None:
    if delete == False:
        return None
    folder = "/home/jorge/DocumentsWLS/Data_Science_Projects/MLOPS-project-NYC-Taxi-Fare-Prediction/CM_data"

    # Recorre todo el contenido y lo elimina
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # borrar archivo o symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # borrar directorio recursivamente
        except Exception as e:
            print(f"Error al borrar {file_path}: {e}")

class ContinuousMonitoringCM:
    def __init__(self, processed_dir: str, models_dir: str, target_col: str = "Fare_Amt"):
        self.processed_dir = processed_dir
        self.target_col = target_col
        self.logger = logger
        self.monitoring_file = os.path.join(self.processed_dir, "monitoring_processed_osrm.parquet")
        self.model_path = self.get_latest_version(models_dir)
        self.model = self.load_model()
        self.df_new = self.load_monitoring_data()
        self.df_old = self.load_old_data()
        self.token = self.get_token()

    def get_latest_version(self, base_dir: str) -> str:
        try:
            if not os.path.isdir(base_dir):
                return None
            dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            versions = [(d, int(re.sub(r"\D", "", d))) for d in dirs if re.match(r"v\d+", d)]
            if not versions:
                return None

            latest_name, latest_num = max(versions, key=lambda x: x[1])

            return os.path.join(base_dir, latest_name, "final_model.pkl")
        except Exception as e:
            self.logger.error(f"Hubo un error obteniendo la última versión {e}")
            raise

    def load_model(self):
        logger.info(f"Cargando modelo desde {self.model_path}")
        model = joblib.load(self.model_path)
        logger.info("Modelo cargado correctamente")
        return model

    def load_monitoring_data(self):
        if not os.path.exists(self.monitoring_file):
            raise FileNotFoundError(f"No se encontró {self.monitoring_file}")
        logger.info(f"Cargando datos de monitoring desde {self.monitoring_file}")
        df = pd.read_parquet(self.monitoring_file)
        if self.target_col not in df.columns:
            raise ValueError(f"La columna target '{self.target_col}' no está en el dataset")
        for c in ['Real_time', 'Real_distance', 'Total_Amt']:
                if c in df.columns:
                    del df[c]
        return df

    def load_old_data(self):
        old_file = "new_data/processed_osrm/val_processed_osrm.parquet"
        if not os.path.exists(old_file):
            raise FileNotFoundError(f"No se encontró el archivo {old_file}")
        df_old = pd.read_parquet(old_file)
        if self.target_col not in df_old.columns:
            raise ValueError(f"La columna target '{self.target_col}' no está en el dataset de referencia")
        # Subsample estratificado
        for c in ['Real_time', 'Real_distance', 'Total_Amt']:
                if c in df_old.columns:
                    del df_old[c]
        df_old = self.stratified_sample_skewed(df_old, self.target_col, n_samples=500000)
        gc.collect()
        return df_old

    def predict(self):
        X = self.df_new.drop(columns=[self.target_col])
        logger.info("Generando predicciones sobre el dataset de monitoring")
        self.df_new["prediction"] = self.model.predict(X)
        logger.info("Predicciones generadas correctamente")
        del X
        X = self.df_old.drop(columns=[self.target_col])
        logger.info("Generando predicciones sobre el dataset anterior")
        self.df_old["prediction"] = self.model.predict(X)
        logger.info("Predicciones generadas correctamente")

    def stratified_sample_skewed(self, df, target_col, n_samples, n_bins=50, random_state=42):
        df = df.copy()
        df["_log_target"] = np.log1p(df[target_col])
        df["_bin"] = pd.qcut(df["_log_target"], q=n_bins, duplicates="drop")
        bin_counts = df["_bin"].value_counts()
        total = len(df)
        n_per_bin = (bin_counts / total * n_samples).round().astype(int)
        df_sampled = df.groupby("_bin", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), n_per_bin[x.name]), random_state=random_state)
        )
        return df_sampled.drop(columns=["_log_target", "_bin"])

    def get_token(self):
        token_file = "Evidentlytoken.txt"
        if not os.path.exists(token_file):
            raise FileNotFoundError(f"No se encontró {token_file}")
        with open(token_file, "r") as f:
            token = f.read().strip()
        return token

    def generate_evidently_report(self, output_dir="reports", html_name="monitoring_report.html"):
        # Subsample estratificado para df_old

        # Crear workspace de Evidently Cloud
        ws = CloudWorkspace(token=self.token, url="https://app.evidently.cloud")
        project = ws.get_project("0199a389-3a83-78d1-b971-9c255efdba61")

        # Definir esquema: todas las features numéricas excepto target/prediction
        feature_columns = [c for c in self.df_new.columns if c not in ["Fare_Amt", "prediction"]]

        schema = DataDefinition(
            numerical_columns=feature_columns,
            regression=[Regression(target="Fare_Amt", prediction="prediction")]
        )

        # Crear datasets de Evidently
        eval_data_new = Dataset.from_pandas(self.df_new, data_definition=schema)
        eval_data_old = Dataset.from_pandas(self.df_old, data_definition=schema)

        # Crear reporte
        report = Report([
            DataDriftPreset(), 
            RegressionPreset(),
        ], include_tests=True)
        self.logger.info("Subiendo el reporte a la nube...")
        my_eval = report.run(eval_data_new, eval_data_old)

        # Subir a Evidently Cloud
        ws.add_run(project.id, my_eval, include_data=False)
        self.logger.info("El reporte ha sido subido a la nube")


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    processed_dir = os.path.join(root_dir, "CM_data/processed_osrm")
    models_dir = os.path.join(root_dir, "flask_app/models")

    cm = ContinuousMonitoringCM(processed_dir, models_dir)
    cm.predict()
    cm.generate_evidently_report()
    delete_data(delete=True)

if __name__ == "__main__":
    main()