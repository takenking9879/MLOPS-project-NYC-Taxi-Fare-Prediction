import logging
import yaml
import pandas as pd
import os
import json
import re

def create_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Crea y configura un logger con consola y archivo opcional.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Evitar duplicar handlers si la función se llama varias veces
    if not logger.handlers:
        # Handler de consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler de archivo, solo si se pasa log_file
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.ERROR)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

def guardar_version_parquets(self, ruta_parquets, ruta_json, version):
    """
    Guarda en un JSON los .parquet de la carpeta indicada bajo una versión dada.
    ruta_parquets : str
        Carpeta donde están los .parquet (ej. "./data/original").
    ruta_json : str
        Ruta del JSON donde se guardará (ej. "./logs/used_parquets.json").
    version : str
        Nombre de la versión (ej. "v1").
    """
    # Obtener todos los archivos parquet en la carpeta
    archivos = [f for f in os.listdir(ruta_parquets) if f.endswith(".parquet")]

    # Extraer año-mes (YYYY-MM) de cada archivo
    archivos = [
        re.search(r"(\d{4})-(\d{2})", f).group(0)
        for f in archivos if re.search(r"(\d{4})-(\d{2})", f)
    ]

    # Ordenar cronológicamente (como strings funciona bien para YYYY-MM)
    archivos = sorted(archivos)

    # Si ya existe el JSON, lo cargamos
    if os.path.exists(ruta_json):
        with open(ruta_json, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Actualizar o crear la versión
    data[version] = archivos

    # Guardar el JSON actualizado
    with open(ruta_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Guardados {len(archivos)} archivos en {ruta_json} bajo '{version}'")

class BaseUtils:
    """
    Clase base con métodos utilitarios para cargar parámetros y usar logging.
    """
    def __init__(self, logger: logging.Logger, params_path: str, columns: list = []):
        self.logger = logger
        self.params_path = params_path

    def load_params(self) -> dict:
        """
        Carga un archivo YAML y retorna un diccionario con los parámetros.
        """
        try:
            with open(self.params_path, 'r') as file:
                params = yaml.safe_load(file)
            self.logger.debug('Parameters retrieved from %s', self.params_path)
            return params
        except FileNotFoundError:
            self.logger.error('File not found: %s', self.params_path)
            raise
        except yaml.YAMLError as e:
            self.logger.error('YAML error: %s', e)
            raise
        except Exception as e:
            self.logger.error('Unexpected error: %s', e)
            raise
    
    def load_parquet(self, path: str, columns: list) -> pd.DataFrame:
        try:
            df = pd.read_parquet(path, columns=columns)
            self.logger.debug('Parquet file retrived from %s', path)
            return df
        except FileNotFoundError:
            self.logger.error('File not found: %s', path)
            raise
        except Exception as e:
            self.logger.error('Unexpected error: %s', e)
            raise
        