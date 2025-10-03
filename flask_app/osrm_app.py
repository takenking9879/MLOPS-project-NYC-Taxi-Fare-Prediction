import aiohttp
import asyncio
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import gc
import tempfile

async def fetch_osrm(session, url, max_retries=6, base_delay=0.4):
    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=10) as r:
                if r.status == 200:
                    data = await r.json()
                    if data.get("code") == "Ok" and data.get("routes"):
                        return data['routes'][0]['distance'], data['routes'][0]['duration']
        except Exception:
            pass
        await asyncio.sleep(base_delay * (2 ** attempt))
    return None, None


async def run_chunk_async(df_chunk,
                          osrm_base="http://127.0.0.1:5000/route/v1/driving",
                          concurrency=50):
    """
    Procesa un chunk y devuelve SOLO dos columnas:
    route_distance_m y route_duration_s.
    """
    total_tasks = len(df_chunk)
    results = np.full((total_tasks, 2), np.nan, dtype=np.float32)

    if total_tasks == 0:
        return pd.DataFrame({
            "route_distance_m": results[:, 0],
            "route_duration_s": results[:, 1]
        })

    conn = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=30)

    q: asyncio.Queue = asyncio.Queue()
    coords_np = df_chunk.to_numpy()
    for i, (lon1, lat1, lon2, lat2) in enumerate(coords_np):
        q.put_nowait((i, lon1, lat1, lon2, lat2))

    pbar = tqdm(total=total_tasks, desc="OSRM async", file=sys.stdout, leave=True)

    async def worker(session: aiohttp.ClientSession):
        while True:
            try:
                i, lon1, lat1, lon2, lat2 = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            coords = f"{lon1},{lat1};{lon2},{lat2}"
            url = f"{osrm_base}/{coords}?overview=false"
            dist, dur = await fetch_osrm(session, url)
            if dist is not None:
                results[i, 0] = dist
            if dur is not None:
                results[i, 1] = dur
            pbar.update(1)
            q.task_done()

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        n_workers = min(concurrency, total_tasks)
        tasks = [asyncio.create_task(worker(session)) for _ in range(n_workers)]
        await asyncio.gather(*tasks)

    pbar.close()

    return pd.DataFrame({
        "route_distance_m": results[:, 0],
        "route_duration_s": results[:, 1]
    })

def run_osrm_in_chunks_single_csv(
    df,
    osrm_base="http://127.0.0.1:5000/route/v1/driving",
    concurrency=50,
    chunk_size=50000,
    csv_path=None,
    chunk_progress=True,
    cleanup=True
):
    """
    Ejecuta OSRM por chunks (usando run_chunk_async) y va guardando/append-eando
    los resultados en UN SOLO CSV temporal. Al final lee el CSV y lo une al df original
    por índice, devolviendo (df_out, csv_path).

    - df: DataFrame con columnas ['Start_Lon','Start_Lat','End_Lon','End_Lat'] y con el índice
          correspondiente al DataFrame original (p.ej. RangeIndex o índice significativo).
    - csv_path: ruta del CSV de salida. Si None, crea uno en cwd con tempfile.
    - chunk_size: filas por chunk.
    - concurrency: pasado a run_chunk_async.
    - cleanup: si True, no borra el CSV (deja para inspección). Si False, borra al finalizar.
    """

    # --------------- preparar CSV ---------------
    if csv_path is None:
        tmpdir = tempfile.mkdtemp(prefix="osrm_csv_")
        csv_path = os.path.join(tmpdir, "osrm_routes.csv")
    else:
        
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
        USE_EXISTING_THRESHOLD = 500 * 1024 * 1024  # 500 MB en bytes
        if os.path.exists(csv_path):
            os.remove(csv_path)  # empezar limpio

    total_len = len(df)
    first_write = True

    try:
        for start in range(0, total_len, chunk_size):
            end = min(start + chunk_size, total_len)
            df_chunk = df.iloc[start:end]

            if chunk_progress:
                print(f"[OSRM] Procesando filas {start} a {end}...", file=sys.stdout)

            # Ejecutar el worker async (compatibilidad con event loop)
            try:
                merged_chunk = asyncio.run(
                    run_chunk_async(df_chunk[["Start_Lon","Start_Lat","End_Lon","End_Lat"]],
                                    osrm_base=osrm_base,
                                    concurrency=concurrency)
                )
            except RuntimeError:
                loop = asyncio.get_event_loop()
                merged_chunk = loop.run_until_complete(
                    run_chunk_async(df_chunk[["Start_Lon","Start_Lat","End_Lon","End_Lat"]],
                                    osrm_base=osrm_base,
                                    concurrency=concurrency)
                )

            # Alinear índices del chunk con el df original
            merged_chunk.index = df_chunk.index

            # Forzar dtypes compactos
            merged_chunk = merged_chunk.astype({"route_distance_m":"float32", "route_duration_s":"float32"}, copy=False)

            # Reset index para escribir orig_index como columna en CSV
            out_df = merged_chunk.reset_index().rename(columns={"index":"orig_index"})
            out_df["orig_index"] = out_df["orig_index"].astype("int64", copy=False)

            # Escribir al CSV: primera vez con header, después append sin header
            if first_write:
                out_df.to_csv(csv_path, index=False, float_format="%.6f")
                first_write = False
            else:
                out_df.to_csv(csv_path, index=False, header=False, mode="a", float_format="%.6f")

            # liberar memoria del chunk
            del df_chunk
            del merged_chunk
            del out_df
            gc.collect()

    except Exception as e:
        # intento dejar un estado razonable y propagar
        gc.collect()
        raise

    # --------------- leer CSV final y unir ---------------
    # dtype mapping para que pandas use float32 e int64
    dtypes = {
        "orig_index": np.int64,
        "route_distance_m": np.float32,
        "route_duration_s": np.float32
    }

    # read_csv: si el archivo es muy grande, puede tardar; ajusta chunksize si lo prefieres
    routes_df = pd.read_csv(csv_path, dtype=dtypes)
    routes_df = routes_df.set_index("orig_index").sort_index()

    # Asegurar tipos por si acaso
    routes_df = routes_df.astype({"route_distance_m":"float32","route_duration_s":"float32"}, copy=False)

    # Asignar al df original (hacemos copy para no mutar la entrada si no quieres)
    df_out = df.copy()
    df_out[["route_distance_m", "route_duration_s"]] = routes_df[["route_distance_m", "route_duration_s"]]

    gc.collect()

    # Si cleanup True, deja el CSV (útil para inspección). Si False, lo borra
    if not cleanup:
        try:
            os.remove(csv_path)
        except Exception:
            pass

    return df_out, csv_path

