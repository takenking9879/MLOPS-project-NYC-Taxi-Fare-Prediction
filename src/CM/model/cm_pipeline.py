import subprocess

scripts = [
    "src/CM/data/data_download_cm.py",
    "src/CM/data/data_validation_cm.py",
    "src/CM/data/data_ingestion_cm.py",
    "src/CM/data/data_preprocessing_cm.py",
    "src/CM/data/data_osrm_cm.py",
    "src/CM/model/model_monitoring.py"
]

for s in scripts:
    subprocess.run(["python3", "-u", s])
