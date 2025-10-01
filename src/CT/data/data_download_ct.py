from src.utils import create_logger, BaseUtils
import requests
import os

logger = create_logger('data_download_ct', 'logs/data_download_ct.log')

class DataDownloadCT(BaseUtils):
    def __init__(self, params_path: str, new_data_original_dir: str):
        super().__init__(logger=logger,params_path=params_path)
        self.new_data_original_dir = new_data_original_dir
        self.params = self.load_params()["continuous_training"]
        self.retrain = self.params["retrain"]
        self.params = self.params["data_download_ct"]
    
    def descargar_parquets(self, start_year: int, start_month: int) -> None:
        """
        Descarga 3 archivos parquet consecutivos de NYC taxi a partir de un año y mes dados.
        Si el año es 2009 y el mes es 01 o 02, muestra un warning y no descarga nada.
        """
        # Caso especial
        if start_year == 2009 and start_month in [1, 2]:
            self.logger.warning("Los datos de 01 y 02 ya son considerados por el modelo")
            raise ValueError("Datos de 2009-01 y 2009-02 ya considerados") #Cambiar esto dependiendo de que datos usaste en el pipeline principal

        os.makedirs(self.new_data_original_dir, exist_ok=True)
        if not self.retrain:
            return None
        
        base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"

        for i in range(3):  # mes inicial + 2 siguientes
            year = start_year
            month = start_month + i

            if month > 12:
                month -= 12
                year += 1

            url = base_url.format(year=year, month=month)
            filename = os.path.join(self.new_data_original_dir, f"yellow_tripdata_{year}-{month:02d}.parquet")

            try:
                self.logger.info(f"Descargando: {url}")
                response = requests.get(url, stream=True)

                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    self.logger.info(f"Guardado en: {filename}")
                else:
                    self.logger.warning(f"No se pudo descargar {url}, status={response.status_code}")

            except Exception as e:
                self.logger.error(f"Error inesperado al descargar {url}: {e}", exc_info=True)
                raise

    def download_window(self):
        try:
            year = self.params["year"]
            month = self.params["month"]
            self.descargar_parquets(year, month)
        except Exception as e:
            self.logger.error("No se pudo descargar la ventana de datos")
            raise

def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        params_path =  os.path.join(root_dir, 'params.yaml')
        new_data_original_dir = os.path.join(root_dir, 'new_data/original')

        download = DataDownloadCT(params_path, new_data_original_dir)
        download.download_window()

    except Exception as e:
        download.logger.error(f"Failed to complete the data validation pipeline: {e}")
        raise
        
if __name__ == "__main__":
    main()


