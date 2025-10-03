from setuptools import setup, find_packages

setup(
    name="mlops_taxi",  # Nombre de tu librería
    version="0.1.0",
    description="Package for the MLOps project for taxi fare prediction",
    author="Jorge Angel Manzanares Cortes",
    author_email="projorge.15@gmail.com",
    packages=find_packages(),           # Indica que los paquetes vienen de src/
    install_requires=[
        "MLflow==2.22.1",
        "Flask==3.1.2",
        "Flask_Cors==6.0.1",
        "dvc==3.63.0",
        "boto3==1.40.26",
        "PyYAML==6.0.2",
        "requests==2.32.5",
        "numpy==2.3.2",
        "pandas==2.3.2",
        "scikit-learn==1.7.2",
        "scipy==1.16.1",
        "dagshub==0.6.3",
        "xgboost==3.0.5",
        "lightgbm==4.6.0",
        "catboost==1.2.8",
        "matplotlib==3.10.6",
        "seaborn==0.13.2",
        "joblib==1.5.2",
        "ipykernel==6.30.1",
        "statsmodels==0.14.5",
        "watchdog==6.0.0",
        "gunicorn==23.0.0",
        "category-encoders==2.8.1",
        "osrm-py==0.5",
        "tqdm==4.67.1",
        "geopandas==1.1.1",
        "shapely==2.1.1",
        "evidently==0.7.14",

    ],
    python_requires=">=3.11",
)
