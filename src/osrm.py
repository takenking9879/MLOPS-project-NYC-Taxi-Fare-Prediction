import aiohttp
import asyncio
import pandas as pd
from tqdm import tqdm
try:
    from tqdm.notebook import tqdm as tqdm_notebook
    _TQDM_NB = True
except Exception:
    tqdm_notebook = tqdm
    _TQDM_NB = False
import os
import sys

async def fetch_osrm(session, url, max_retries=4, base_delay=0.5):
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

async def worker_async(i, lon1, lat1, lon2, lat2, session, osrm_base):
    coords = f"{lon1},{lat1};{lon2},{lat2}"
    url = f"{osrm_base}/{coords}?overview=false"
    dist, dur = await fetch_osrm(session, url)
    return i, dist, dur

async def run_chunk_async(df_chunk,
                          osrm_base="http://127.0.0.1:5000/route/v1/driving",
                          concurrency=50):
    """
    Ejecuta las requests de un chunk en paralelo y muestra una barra de progreso
    que se actualiza por cada request completada. Compatible con notebooks y consola.
    """
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=30)

    # elegir pbar apropiada para entorno
    total_tasks = len(df_chunk)
    pbar = (tqdm_notebook(total=total_tasks, desc="OSRM async")
            if _TQDM_NB else tqdm(total=total_tasks, desc="OSRM async", file=sys.stdout, leave=True))

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        tasks = []

        # definimos la tarea que encapsula semáforo + worker + update de la barra
        async def sem_task(i, row):
            async with sem:
                try:
                    res = await worker_async(i,
                                             row['Start_Lon'],
                                             row['Start_Lat'],
                                             row['End_Lon'],
                                             row['End_Lat'],
                                             session,
                                             osrm_base)
                except Exception:
                    # en caso de fallo individual devolvemos índices con None
                    res = (i, None, None)
                # actualizar barra (estamos en el mismo hilo/event-loop)
                pbar.update(1)
                return res

        # crear tareas
        for i, row in df_chunk.reset_index(drop=True).iterrows():
            tasks.append(asyncio.create_task(sem_task(i, row)))

        results = []
        try:
            # gather esperará a que todas terminen, pero la barra se irá actualizando
            gathered = await asyncio.gather(*tasks, return_exceptions=False)
            # gathered es lista de tuplas (i, dist, dur)
            for tup in gathered:
                i, dist, dur = tup
                results.append({"idx": i, "route_distance_m": dist, "route_duration_s": dur})
        finally:
            # cancelar tareas pendientes si ocurre algo
            for t in tasks:
                if not t.done():
                    t.cancel()
            pbar.close()

    # reconstruir dataframe ordenado por idx
    res_df = pd.DataFrame([r for r in results if r["idx"] is not None])
    if not res_df.empty:
        res_df = res_df.sort_values("idx").reset_index(drop=True)
    else:
        res_df = pd.DataFrame(columns=["idx", "route_distance_m", "route_duration_s"])

    merged = df_chunk.reset_index(drop=True).merge(res_df, left_index=True, right_on="idx", how="left")
    if "idx" in merged.columns:
        merged = merged.drop(columns=["idx"])
    return merged


def run_osrm_in_chunks(df, osrm_base="http://127.0.0.1:5000/route/v1/driving",
                       concurrency=50, chunk_size=50000,
                       out_path="../../data/osrm/osrm_results.parquet", partial_save=False):
    """
    Procesa un DataFrame grande en chunks para no saturar la RAM/OSRM.
    Cada chunk crea su propia barra (con updates por request completada),
    lo cual queda bien en consola y notebooks.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    partial_results = []

    total_rows = len(df)
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        df_chunk = df.iloc[start:end]
        # usamos print aquí para notificar; si quieres no romper la barra usa pbar.write dentro del async
        print(f"Procesando filas {start} a {end}...")

        try:
            # Parche para notebooks donde ya hay un loop corriendo
            try:
                loop = asyncio.get_running_loop()
                import nest_asyncio
                nest_asyncio.apply()
                # con nest_asyncio aplicado podemos usar run_until_complete
                merged_chunk = loop.run_until_complete(run_chunk_async(df_chunk,
                                                                       osrm_base=osrm_base,
                                                                       concurrency=concurrency))
            except RuntimeError:
                # ambiente normal (.py) -> asyncio.run
                merged_chunk = asyncio.run(run_chunk_async(df_chunk,
                                                           osrm_base=osrm_base,
                                                           concurrency=concurrency))
        except RuntimeError as e:
            # En algunos entornos raros, sugerir al usuario cómo proceder
            print("Si estás en un notebook sin nest_asyncio, usa: merged_chunk = await run_chunk_async(df_chunk, ...)")
            raise e

        partial_results.append(merged_chunk)

        # Guardar resultado parcial si se pidió
        if partial_save:
            pd.concat(partial_results, ignore_index=True).to_parquet(out_path, index=False)
            print(f"Guardado parcial hasta fila {end} en {out_path}")

    # concatenar y devolver
    if partial_results:
        return pd.concat(partial_results, ignore_index=True)
    else:
        return pd.DataFrame(columns=list(df.columns) + ["route_distance_m", "route_duration_s"])
