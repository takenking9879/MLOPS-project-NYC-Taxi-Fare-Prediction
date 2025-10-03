"""
Flask app para NYC Taxi Fare Prediction
- Ubica este archivo en flask_app/app.py
- Plantillas HTML se crearán automáticamente en flask_app/templates si no existen
- models/ debe contener v1/, v2/, ... cada uno con final_model.pkl y opcionalmente stacking.pkl

Notas:
- Requiere: flask, joblib, pandas, numpy, requests, python-dateutil
- Frontend usa Leaflet + Nominatim para geocoding (sin API key)
- Se convierte Trip_Pickup_DateTime a datetime antes del preprocesado.
- `time_features` protege la eliminación de Trip_Dropoff_DateTime si no existe.
- Se añadió un cache simple `_MODEL_CACHE` y `get_model(path)` para evitar cargas repetidas con joblib.load.
- `osrm_distance` ahora mergea los resultados de OSRM en el DataFrame.

"""
import os
osrm_base = os.environ.get("OSRM_BASE", "http://osrm:5000/route/v1/driving")
from .osrm_app import run_chunk_async
import asyncio
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
import gc
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "change_me_for_prod"

# ------------------ Helpers para templates auto-creación ------------------
TEMPLATES = {
    "index.html": """(HTML OMITIDO PARA BREVIDAD)""",
    "result.html": """(HTML OMITIDO PARA BREVIDAD)"""
}


def ensure_templates(app_root):
    tpl_dir = os.path.join(app_root, 'templates')
    os.makedirs(tpl_dir, exist_ok=True)
    for name, content in TEMPLATES.items():
        path = os.path.join(tpl_dir, name)
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

# Al iniciar, asegurar templates si se ejecuta desde flask_app/
ensure_templates(os.path.dirname(__file__))

# ------------------ Simple model cache ------------------
_MODEL_CACHE = {}

def get_model(path):
    """Carga y cachea modelos para evitar reloads repetidos en cada request."""
    if path is None:
        return None
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    if not os.path.exists(path):
        return None
    mdl = joblib.load(path)
    _MODEL_CACHE[path] = mdl
    return mdl

# ------------------ Lógica de modelos ------------------

def list_versions(models_root):
    if not os.path.isdir(models_root):
        return []
    dirs = [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d)) and d.startswith('v')]
    # ordenar por número
    def num(d):
        try:
            return int(''.join(filter(str.isdigit, d)))
        except:
            return 0
    dirs_sorted = sorted(dirs, key=num)
    return dirs_sorted


def load_models_for_version(models_root, version_name):
    """
    Retorna un dict con claves:
      - method: 'stacked'|'latest'|'old'
      - fare_predictor: func(X_df) -> np.array
      - details: información para debug
    """
    details = {}
    version_dir = os.path.join(models_root, version_name)
    latest_model_path = os.path.join(version_dir, 'final_model.pkl')
    stacking_path = os.path.join(version_dir, 'stacking.pkl')
    promotion_json_path = os.path.join(version_dir, 'promotion_choice.json')

    chosen_on_val = None
    if os.path.exists(promotion_json_path):
        try:
            with open(promotion_json_path, 'r') as f:
                pj = json.load(f)
            chosen_on_val = pj.get('chosen_on_val')
            details['promotion_json'] = pj
        except Exception as e:
            details['promotion_json_error'] = str(e)

    # default predictor uses latest (using get_model cache)
    def latest_predict(X):
        mdl = get_model(latest_model_path)
        if mdl is None:
            raise RuntimeError(f"latest model not found: {latest_model_path}")
        return mdl.predict(X)

    # prepare possible stacked predictor
    if chosen_on_val == 'stacked' and os.path.exists(stacking_path):
        # need previous version
        try:
            ver_num = int(''.join(filter(str.isdigit, version_name)))
            prev_name = f'v{ver_num - 1}'
        except Exception:
            prev_name = None

        prev_model_path = os.path.join(models_root, prev_name, 'final_model.pkl') if prev_name else None

        if prev_model_path and os.path.exists(prev_model_path) and os.path.exists(latest_model_path):
            # create stacked predictor using cached loads
            def stacked_predict(X):
                prev = get_model(prev_model_path)
                latest = get_model(latest_model_path)
                meta = get_model(stacking_path)
                if prev is None or latest is None or meta is None:
                    raise RuntimeError('Missing model for stacking (prev/latest/meta)')
                p_prev = prev.predict(X)
                p_latest = latest.predict(X)
                stacked_in = np.column_stack([p_prev, p_latest])
                return meta.predict(stacked_in)

            details['method'] = 'stacked'
            details['stacking_path'] = stacking_path
            details['previous_model_path'] = prev_model_path
            details['latest_model_path'] = latest_model_path
            return {
                'method': 'stacked',
                'fare_predictor': stacked_predict,
                'details': details
            }
        else:
            details['fallback_reason'] = 'missing_prev_or_latest_for_stacking'

    # if chosen_on_val explicitly 'old'
    if chosen_on_val == 'old':
        try:
            ver_num = int(''.join(filter(str.isdigit, version_name)))
            prev_name = f'v{ver_num - 1}'
            prev_model_path = os.path.join(models_root, prev_name, 'final_model.pkl')
            if os.path.exists(prev_model_path):
                def old_predict(X):
                    mdl = get_model(prev_model_path)
                    if mdl is None:
                        raise RuntimeError(f'previous model not found: {prev_model_path}')
                    return mdl.predict(X)
                details['method'] = 'old'
                details['previous_model_path'] = prev_model_path
                return {'method': 'old', 'fare_predictor': old_predict, 'details': details}
        except Exception:
            pass

    # default/latest
    details['method'] = 'latest'
    details['latest_model_path'] = latest_model_path
    return {'method': 'latest', 'fare_predictor': latest_predict, 'details': details}

# ------------------ Placeholder preprocess + helpers ------------------
# TODO: Implementar esta función según tu pipeline de features
# Debe recibir df con columnas: ['Trip_Pickup_DateTime','Passenger_Count','Start_Lon','Start_Lat','End_Lon','End_Lat','Fare_Amt']
# y devolver X (features) listo para pasar al predictor (pandas.DataFrame)

def haversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    # Radio de la Tierra en km
    R = 6371.0
    # Convertir a radianes
    lon1 = np.radians(df['Start_Lon'])
    lat1 = np.radians(df['Start_Lat'])
    lon2 = np.radians(df['End_Lon'])
    lat2 = np.radians(df['End_Lat'])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df['haversine_distance_m'] = R * c * 1000
    cols_to_float32 = ['Start_Lon','Start_Lat','End_Lon','End_Lat','haversine_distance_m']
    df[cols_to_float32] = df[cols_to_float32].astype("float32")
    return df


def time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Asegurar que Trip_Pickup_DateTime sea datetime
    if not np.issubdtype(df['Trip_Pickup_DateTime'].dtype, np.datetime64):
        df['Trip_Pickup_DateTime'] = pd.to_datetime(df['Trip_Pickup_DateTime'])

    # Extraer hora y día de la semana
    hour = df['Trip_Pickup_DateTime'].dt.hour
    weekday = df['Trip_Pickup_DateTime'].dt.weekday  # 0=Monday, 6=Sunday

    # Eliminar columnas con protección
    if 'Trip_Pickup_DateTime' in df.columns:
        del df['Trip_Pickup_DateTime']
    if 'Trip_Dropoff_DateTime' in df.columns:
        del df['Trip_Dropoff_DateTime']

    # Codificación cíclica
    df['pickup_hour_sin'] = np.sin(2 * np.pi * hour/24)
    df['pickup_hour_cos'] = np.cos(2 * np.pi * hour/24)
    df['pickup_weekday_sin'] = np.sin(2 * np.pi * weekday/7)
    df['pickup_weekday_cos'] = np.cos(2 * np.pi * weekday/7)

    del hour
    del weekday

    cyc_cols = ['pickup_hour_sin', 'pickup_hour_cos', 'pickup_weekday_sin', 'pickup_weekday_cos']
    df[cyc_cols] = df[cyc_cols].astype("float32")
    return df


def add_haversine_poi_distances(df: pd.DataFrame, pois: dict) -> pd.DataFrame:
    """
    Calcula distancias Haversine desde Start (pickup) y End (dropoff) a puntos de interés.
    """
    R = 6371.0  # radio de la Tierra en km

    lat_start = np.radians(df['Start_Lat'])
    lon_start = np.radians(df['Start_Lon'])
    lat_end = np.radians(df['End_Lat'])
    lon_end = np.radians(df['End_Lon'])

    for name, (lat_poi, lon_poi) in pois.items():
        lat_poi_rad = np.radians(lat_poi)
        lon_poi_rad = np.radians(lon_poi)

        # Start -> POI
        dlat = lat_poi_rad - lat_start
        dlon = lon_poi_rad - lon_start
        a = np.sin(dlat/2)**2 + np.cos(lat_start) * np.cos(lat_poi_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        df[f'pickup_distance_to_{name}_m'] = (R * c * 1000).astype('float32')

        # End -> POI
        dlat = lat_poi_rad - lat_end
        dlon = lon_poi_rad - lon_end
        a = np.sin(dlat/2)**2 + np.cos(lat_end) * np.cos(lat_poi_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        df[f'dropoff_distance_to_{name}_m'] = (R * c * 1000).astype('float32')
        gc.collect()

    return df


def osrm_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta run_chunk_async y mergea las columnas resultantes en el df original."""
    merged_chunk = asyncio.run(
        run_chunk_async(df[["Start_Lon","Start_Lat","End_Lon","End_Lat"]],
                        osrm_base=osrm_base,
                        concurrency=10)
    )
    # Alinear índices y añadir columnas al df
    merged_chunk.index = df.index
    merged_chunk = merged_chunk.astype({"route_distance_m":"float32", "route_duration_s":"float32"}, copy=False)
    df["route_distance_m"] = merged_chunk["route_distance_m"].values
    df["route_duration_s"] = merged_chunk["route_duration_s"].values
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Implementa tu preprocesamiento aquí. Pequeños parches aplicados."""
    # Asegurar Trip_Pickup_DateTime como datetime (patch)
    if 'Trip_Pickup_DateTime' in df.columns:
        df['Trip_Pickup_DateTime'] = pd.to_datetime(df['Trip_Pickup_DateTime'])

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    impute_path = os.path.join(root_dir, 'artifacts/imputer.pkl')
    scaler_path = os.path.join(root_dir, 'artifacts/scaler_preprocessing.pkl')
    scaler_osrm_path = os.path.join(root_dir, 'artifacts/scaler_osrm_features.pkl')

    # Cargar artefactos (hará throw si no existen — mantener lógica)
    data_imputer = joblib.load(impute_path)
    data_scaler = joblib.load(scaler_path)
    data_scaler_osrm = joblib.load(scaler_osrm_path)
    imputer, cols_to_impute = data_imputer['imputer'], data_imputer['columns']
    del data_imputer
    scaler, cols_to_scale = data_scaler['scaler'], data_scaler['columns']  # <-- corregido
    del data_scaler
    scaler_osrm, cols_to_scale_osrm = data_scaler_osrm['scaler'], data_scaler_osrm['columns']  # <-- corregido
    del data_scaler_osrm

    df = haversine_distance(df)

    # Imputar
    df[cols_to_impute] = imputer.transform(df[cols_to_impute])

    df = time_features(df)

    pois = {
        "nyc": (40.724944, -74.001541),
        "jfk": (40.645494, -73.785937),
        "lga": (40.774071, -73.872067),
        "nla": (40.690764, -74.177721),
        "chp": (41.366138, -73.137393),
        "exp": (40.736000, -74.037500)
    }
    df = add_haversine_poi_distances(df, pois)

    # Escalado
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Normalizar passenger count (mantener tu heurística)
    df['Passenger_Count'] = df['Passenger_Count'] * 0.1

    # OSRM (sin fallback) — si falla, que explote como pediste
    df = osrm_distance(df)

    # Features derivadas
    df['average_speed_m_s'] = (df['route_distance_m'] / df['route_duration_s']).astype("float32")
    df['ratio_haversine_osrm'] = df['route_distance_m'] / df['haversine_distance_m'].astype("float32")

    # Escalar features OSRM
    df[cols_to_scale_osrm] = scaler_osrm.transform(df[cols_to_scale_osrm])

    # Quitar columna objetivo y devolver features
    X = df.drop(columns=['Fare_Amt'], errors='ignore')
    return X

# ------------------ Rutas Flask ------------------
@app.route('/', methods=['GET'])
def index():
    models_root = os.path.join(app.root_path, 'models')
    versions = list_versions(models_root)
    return render_template('index.html', versions=versions)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        version = request.form.get('version')
        origin_lat = request.form.get('origin_lat')
        origin_lon = request.form.get('origin_lon')
        dest_lat = request.form.get('dest_lat')
        dest_lon = request.form.get('dest_lon')
        passengers = int(request.form.get('passengers') or 1)
        pickup_dt = request.form.get('pickup_datetime')

        if not (origin_lat and origin_lon and dest_lat and dest_lon):
            flash('Faltan coordenadas. Usa la búsqueda o ponlas manualmente.', 'danger')
            return redirect(url_for('index'))

        # Normalize types
        start_lat = float(origin_lat)
        start_lon = float(origin_lon)
        end_lat = float(dest_lat)
        end_lon = float(dest_lon)

        # parse datetime
        if pickup_dt:
            try:
                pickup = parser.parse(pickup_dt)
            except Exception:
                pickup = datetime.now()
        else:
            pickup = datetime.now()

        # construir df con la estructura pedida
        df = pd.DataFrame([{
            'Trip_Pickup_DateTime': pickup,  # ahora datetime, patch
            'Passenger_Count': passengers,
            'Start_Lon': start_lon,
            'Start_Lat': start_lat,
            'End_Lon': end_lon,
            'End_Lat': end_lat,
            'Fare_Amt': np.nan
        }])

        # Preprocess
        X = preprocess(df.copy())

        models_root = os.path.join(app.root_path, 'models')
        loader = load_models_for_version(models_root, version)
        method = loader.get('method')
        predictor = loader.get('fare_predictor')
        details = loader.get('details')

        # hacer predicción
        preds = predictor(X)
        fare = float(preds[0])

        breakdown = None
        if method == 'stacked':
            try:
                ver_num = int(''.join(filter(str.isdigit, version)))
                prev_name = f'v{ver_num - 1}'
                prev_model_path = os.path.join(models_root, prev_name, 'final_model.pkl')
                latest_model_path = os.path.join(models_root, version, 'final_model.pkl')
                prev = get_model(prev_model_path)
                latest = get_model(latest_model_path)
                meta = get_model(os.path.join(models_root, version, 'stacking.pkl'))
                p_prev = prev.predict(X)[0] if prev is not None else None
                p_latest = latest.predict(X)[0] if latest is not None else None
                p_stack = meta.predict(np.column_stack([[p_prev],[p_latest]]))[0] if meta is not None else None
                breakdown = {'previous': p_prev, 'latest': p_latest, 'stacked': p_stack}
            except Exception:
                breakdown = details

        # mostrar en template
        return render_template('result.html', fare=round(fare,2), version=version, method=method, breakdown=breakdown)

    except Exception as e:
        app.logger.exception('Error en /predict: %s', e)
        flash(f'Error al predecir: {e}', 'danger')
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
