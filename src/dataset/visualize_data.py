import json
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import folium
import random
import re
from typing import List, Tuple
from folium.features import DivIcon
import csv
import os


def mappa_osservazioni_csv_denmark(
        percorso_csv: str,
        file_html: str = "mappa_osservazioni.html",
        ev_file:str ='../../data/ev/denmark/DenamarkEVstations.json',
        zoom_start: int = 12,
        usa_satellite: bool = True,
        disegna_linea: bool = False
):
    """
    Carica il CSV e disegna, per ogni riga, i due punti di osservazione con lo stesso colore.

    Parametri
    ----------
    percorso_csv : str
        Percorso al file .csv.
    file_html : str, opzionale
        Nome del file HTML di output (default 'mappa_osservazioni.html').
    zoom_start : int, opzionale
        Livello di zoom iniziale (default 12).
    usa_satellite : bool, opzionale
        Se True utilizza lo sfondo satellitare Esri World Imagery; altrimenti OpenStreetMap.
    disegna_linea : bool, opzionale
        Se True collega con una polilinea le due osservazioni di ogni riga.

    Ritorna
    -------
    folium.Map
        Oggetto mappa che può essere ulteriormente personalizzato.
    """
    # --- Caricamento dati ----------------------------------------------------
    df = pd.read_csv(percorso_csv)

    # --- Parsing coordinate --------------------------------------------------
    # Ci si assicura che le stringhe siano nel formato "latitudine, longitudine"
    def parse_coord(coord_str):
        lat_str, lon_str = map(str.strip, coord_str.split(','))
        return float(lat_str), float(lon_str)

    coords1 = df['Observation 1 Coordinates'].apply(parse_coord)
    coords2 = df['Observation 2 Coordinates'].apply(parse_coord)

    # --- Centroide iniziale --------------------------------------------------
    # Media delle prime coordinate per centrare la mappa
    lat0, lon0 = coords1.iloc[0]
    mappa = folium.Map(location=[lat0, lon0],
                       zoom_start=zoom_start,
                       tiles=None)  # nessun layer di default

    # Layer satellitare o standard
    if usa_satellite:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr=("Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, "
                  "GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, "
                  "and the GIS User Community"),
            name="Esri World Imagery",
            overlay=False,
            control=True
        ).add_to(mappa)
    else:
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(mappa)

    # --- Funzione per generare colori casuali --------------------------------
    def colore_random() -> str:
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    # --- Aggiunta marker (e polilinee) ---------------------------------------
    for (lat1, lon1), (lat2, lon2) in zip(coords1, coords2):
        colore = colore_random()

        folium.CircleMarker(
            location=[lat1, lon1],
            radius=6,
            color=colore,
            fill=True,
            fill_opacity=0.9
        ).add_to(mappa)

        folium.CircleMarker(
            location=[lat2, lon2],
            radius=6,
            color=colore,
            fill=True,
            fill_opacity=0.9
        ).add_to(mappa)

        if disegna_linea:
            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color=colore,
                weight=2,
                opacity=0.8
            ).add_to(mappa)

    with open(ev_file) as f:
        content = json.load(f)
        print(content)

    list_of_EV_stations = list()
    for elem in content:
        lat = elem['AddressInfo']['Latitude']
        long = elem['AddressInfo']['Longitude']
        list_of_EV_stations.append((lat, long))

    for ev_station in list_of_EV_stations:
        folium.CircleMarker(
            location=[ev_station[0], ev_station[1]],
            radius=6,
            color='#000099',
            fill=True,
            fill_opacity=0.9
        ).add_to(mappa)

    folium.LayerControl().add_to(mappa)
    mappa.save(file_html)
    print(f"Mappa salvata in '{file_html}'")
    return mappa


# Compiliamo una regexp per catturare lat e lon decimali
_TOKEN_RE = re.compile(r'^(-?\d+\.\d+),(-?\d+\.\d+)$')


def haversine(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Restituisce la distanza tra p1 e p2 (lat, lon) in metri.
    """
    R = 6371000  # raggio terrestre in metri
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def parse_link_points(
        seq: str,
        len_decimali_considered: int = 1,
        stampa_distanze: bool = True
) -> List[Tuple[float, float]]:
    """
    Estrae solo token lat,lon con precisione minima in lon e,
    se richiesto, stampa la distanza in metri tra ogni coppia consecutiva.

    :param seq: stringa dei link_points
    :param len_decimali_considered: numero minimo di decimali per la lon
    :param stampa_distanze: se True, stamperà le distanze tra i punti
    :return: lista di (lat, lon) valide
    """
    pts: List[Tuple[float, float]] = []
    if not isinstance(seq, str):
        return pts
    list_dist = list()
    # parsing token validi
    for token in seq.split():
        m = _TOKEN_RE.match(token)
        if not m:
            continue
        lat_s, lon_s = m.group(1), m.group(2)
        if len(lon_s.split('.', 1)[1]) < len_decimali_considered:
            continue
        pts.append((float(lat_s), float(lon_s)))

    # calcolo e stampa distanze
    if stampa_distanze and len(pts) >= 2:
        for i in range(1, len(pts)):
            p_prev, p_cur = pts[i - 1], pts[i]
            d = haversine(p_prev, p_cur)
            print(f"Distanza tra punto {i - 1} {p_prev} e punto {i} {p_cur}: {d:.2f} m")
            list_dist.append(d)

    return pts, list_dist


def process_newyork_ev_stations(ev_csv: str) -> List[Tuple[float, float, str]]:
    """
    Legge un CSV con colonne ['LocID','LocName','Latitude','Longitude']
    e ritorna una lista di (lat, lon, id).
    """
    df_ev = pd.read_csv(ev_csv)
    ev_list: List[Tuple[float, float, str]] = []
    for _, row in df_ev.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        ev_id = str(row.get('LocID', row.get('LocName', '')))
        ev_list.append((lat, lon, ev_id))
    return ev_list


def process_chicago_ev_stations(ev_csv: str) -> List[Tuple[float, float, str]]:
    """
    Legge un CSV con colonne ['LocID','LocName','Latitude','Longitude']
    e ritorna una lista di (lat, lon, id).
    """
    df_ev = pd.read_csv(ev_csv)
    ev_list: List[Tuple[float, float, str]] = []
    for _, row in df_ev.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        ev_id = str(row.get('LocID', row.get('LocName', '')))
        ev_list.append((lat, lon, ev_id))
    return ev_list


def _colore_random(rng: random.Random) -> str:
    return "#{:06x}".format(rng.randint(0, 0xFFFFFF))


def mappa_osservazioni_csv_newyork(
        percorso_csv: str,
        file_html: str = "grafo_stradale.html",
        zoom_start: int = 12,
        usa_satellite: bool = True,
        mostra_nodi: bool = False,
        mostra_popup_id: bool = True,
        mostra_label: bool = False,
        show_ev: bool = False,  # ← flag per EV
        ev_file: Optional[str] = None,  # ← percorso al CSV EV
        seed_colori: Optional[int] = 42,
        weight: int = 3,
        opacity: float = 0.8,
        outlier_value_selector: int = 60
):
    # --- Caricamento e parsing ------------------------------------------------
    df = pd.read_csv(percorso_csv)
    if "link_points" not in df or "id" not in df:
        raise ValueError("Manca colonna 'id' o 'link_points' nel CSV.")

    df["__points"] = None
    df["__distances"] = None
    all_seq_distances = list()
    for idx, row in df.iterrows():
        pts, list_dist = parse_link_points(row["link_points"], stampa_distanze=True)
        all_seq_distances.append(list_dist)
        if len(list_dist) == 1:
            df.at[idx, "__points"] = pts
            df.at[idx, "__distances"] = list_dist
        else:
            x = np.array(list_dist)  # la tua sequenza
            M = np.median(x)
            mad = np.median(np.abs(x - M))
            r = np.abs(x - M) / mad
            mask = r <= outlier_value_selector  # seleziona gli indici senza outlier
            if False in mask:
                print(r)
            list_dist = np.array(list_dist)[mask]
            cont=0
            pass_step = False
            for i in range(len(mask)):
                # I need this since I've 2 distance over the range but only one point is the responsible
                if pass_step:
                    pass_step = False
                    continue
                if not mask[i]:
                    _ = pts.pop(i+1+cont)
                    cont -= 1
                    pass_step = True


            # pts = np.array(pts)[mask]
            df.at[idx, "__points"] = pts
            df.at[idx, "__distances"] = list_dist
    print(all_seq_distances)



    # ##################################################################################################################
    # # Metodo per filtrare distanze elevate
    # cont = 0
    # for seq in all_seq_distances:
    #     if len(seq) == 1:  #TODO: Check punti singoli
    #         continue
    #     x = np.array(seq)  # la tua sequenza
    #     M = np.median(x)
    #     mad = np.median(np.abs(x - M))
    #     r = np.abs(x - M) / mad
    #     # seleziona gli indici senza outlier
    #     mask = r <= 100
    #     x_clean = x[mask]
    #     if len(x_clean) != len(seq):
    #         print(f'------------------------------------- {cont} --------------------------------------')
    #         print(seq)
    #         print(x_clean)
    #         cont += 1
    # ##################################################################################################################

    df_valid = df[df["__points"].map(len) >= 2].copy()
    if df_valid.empty:
        raise ValueError("Nessun link valido (>=2 punti) trovato nel CSV.")
    df['__points'] = df['__points'].apply(lambda x: json.dumps(x))
    df['__distances'] = df['__distances'].apply(lambda x: np.array(x))
    df['__distances'] = df['__distances'].apply(lambda x: json.dumps(x.tolist()))
    # df_valid.to_csv('filter_newyork_map.csv', index=False, sep='|')
    df.to_csv('output.csv', sep=';', quoting=csv.QUOTE_NONNUMERIC, index=False)

    # centro della mappa
    all_pts = [pt for pts in df_valid["__points"] for pt in pts]
    center_lat = sum(p[0] for p in all_pts) / len(all_pts)
    center_lon = sum(p[1] for p in all_pts) / len(all_pts)

    # --- Creazione mappa -------------------------------------------------------
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=zoom_start,
                   tiles=None)
    if usa_satellite:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles © Esri — Source: Esri, i‑cubed, USDA, USGS, AEX, GeoEye, "
                 "Getmapping, Aerogrid, IGN, IGP, UPR‑EGP, GIS User Community",
            name="Esri World Imagery", overlay=False, control=True
        ).add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # --- Plot dei link ---------------------------------------------------------
    rng = random.Random(seed_colori)
    for _, row in df_valid.iterrows():
        pts = row["__points"]
        lid = row["id"]
        colore = _colore_random(rng)
        popup_txt = f"id: {lid}" if mostra_popup_id else None

        folium.PolyLine(
            locations=pts,
            color=colore,
            weight=weight,
            opacity=opacity,
            popup=popup_txt
        ).add_to(m)

        if mostra_nodi:
            for (lat, lon), tag in [(pts[0], f"start {lid}"), (pts[-1], f"end {lid}")]:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4, color=colore, fill=True, fill_opacity=1,
                    popup=tag if mostra_popup_id else None
                ).add_to(m)

        if mostra_label:
            mid = pts[len(pts) // 2]
            folium.map.Marker(
                location=[mid[0], mid[1]],
                icon=DivIcon(
                    icon_size=(0, 0), icon_anchor=(0, 0),
                    html=f'<div style="font-size:10pt;color:{colore};'
                         f'text-shadow:1px 1px 2px white;">{lid}</div>'
                )
            ).add_to(m)

    # --- Plot delle EV stations (opzionale) ------------------------------------
    if show_ev:
        if not ev_file:
            raise ValueError("Per mostrare le EV stations devi passare `ev_csv`.")
        ev_list = process_newyork_ev_stations(ev_file)
        ev_group = folium.FeatureGroup(name="EV Stations").add_to(m)
        for lat, lon, ev_id in ev_list:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="#0000FF",
                fill=True,
                fill_opacity=0.7,
                popup=f"EV station {ev_id}"
            ).add_to(ev_group)

    folium.LayerControl().add_to(m)
    out_path = Path(file_html)
    m.save(out_path)
    print(f"Mappa salvata in '{out_path}'")
    return m


def mappa_osservazioni_csv_chicago(
        traffic_file: str,
        file_html: str = "grafo_stradale.html",
        zoom_start: int = 12,
        usa_satellite: bool = True,
        mostra_nodi: bool = False,
        mostra_popup_id: bool = True,
        mostra_label: bool = False,
        show_ev: bool = True,  # ← flag per EV
        ev_file: Optional[str] = None,  # ← percorso al CSV EV
        seed_colori: Optional[int] = 42,
        weight: int = 3,
        opacity: float = 0.8,
        outlier_value_selector: int = 60
):
    # --- Caricamento e parsing ------------------------------------------------
    # ['id', 'street', 'length', 'start_latitude', 'start_longitude', 'end_latitude', 'end_longitude', 'max_speed']
    df = pd.read_csv(traffic_file)
    # if "link_points" not in df or "id" not in df:
    #     raise ValueError("Manca colonna 'id' o 'link_points' nel CSV.")
    #
    # df["__points"] = None
    # df["__distances"] = None
    # all_seq_distances = list()
    # for idx, row in df.iterrows():
    #     pts, list_dist = parse_link_points(row["link_points"], stampa_distanze=True)
    #     all_seq_distances.append(list_dist)
    #     if len(list_dist) == 1:
    #         df.at[idx, "__points"] = pts
    #         df.at[idx, "__distances"] = list_dist
    #     else:
    #         x = np.array(list_dist)  # la tua sequenza
    #         M = np.median(x)
    #         mad = np.median(np.abs(x - M))
    #         r = np.abs(x - M) / mad
    #         mask = r <= outlier_value_selector  # seleziona gli indici senza outlier
    #         if False in mask:
    #             print(r)
    #         list_dist = np.array(list_dist)[mask]
    #         cont=0
    #         pass_step = False
    #         for i in range(len(mask)):
    #             # I need this since I've 2 distance over the range but only one point is the responsible
    #             if pass_step:
    #                 pass_step = False
    #                 continue
    #             if not mask[i]:
    #                 _ = pts.pop(i+1+cont)
    #                 cont -= 1
    #                 pass_step = True
    #
    #
    #         # pts = np.array(pts)[mask]
    #         df.at[idx, "__points"] = pts
    #         df.at[idx, "__distances"] = list_dist
    # print(all_seq_distances)



    # ##################################################################################################################
    # # Metodo per filtrare distanze elevate
    # cont = 0
    # for seq in all_seq_distances:
    #     if len(seq) == 1:  #TODO: Check punti singoli
    #         continue
    #     x = np.array(seq)  # la tua sequenza
    #     M = np.median(x)
    #     mad = np.median(np.abs(x - M))
    #     r = np.abs(x - M) / mad
    #     # seleziona gli indici senza outlier
    #     mask = r <= 100
    #     x_clean = x[mask]
    #     if len(x_clean) != len(seq):
    #         print(f'------------------------------------- {cont} --------------------------------------')
    #         print(seq)
    #         print(x_clean)
    #         cont += 1
    # ##################################################################################################################

    # df_valid = df[df["__points"].map(len) >= 2].copy()
    # if df_valid.empty:
    #     raise ValueError("Nessun link valido (>=2 punti) trovato nel CSV.")
    # df['__points'] = df['__points'].apply(lambda x: json.dumps(x))
    # df['__distances'] = df['__distances'].apply(lambda x: np.array(x))
    # df['__distances'] = df['__distances'].apply(lambda x: json.dumps(x.tolist()))
    # # df_valid.to_csv('filter_newyork_map.csv', index=False, sep='|')
    # df.to_csv('output.csv', sep=';', quoting=csv.QUOTE_NONNUMERIC, index=False)

    # centro della mappa
    # ['id', 'street', 'length', 'start_latitude', 'start_longitude', 'end_latitude', 'end_longitude', 'max_speed']

    all_pts_lat = [pt for pt in df["start_latitude"]]
    all_pts_lat += [pt for pt in df["end_latitude"]]
    all_pts_long = [pt for pt in df["start_longitude"]]
    all_pts_long += [pt for pt in df["end_longitude"]]

    center_lat = sum(p for p in all_pts_lat) / len(all_pts_lat)
    center_lon = sum(p for p in all_pts_long) / len(all_pts_long)


    # --- Creazione mappa -------------------------------------------------------
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=zoom_start,
                   tiles=None)
    if usa_satellite:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles © Esri — Source: Esri, i‑cubed, USDA, USGS, AEX, GeoEye, "
                 "Getmapping, Aerogrid, IGN, IGP, UPR‑EGP, GIS User Community",
            name="Esri World Imagery", overlay=False, control=True
        ).add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

        # --- Aggiunta marker (e polilinee) ---------------------------------------
    rng = random.Random(seed_colori)
    for _, row in df.iterrows():
        lat1, lon1, lat2, lon2 = row["start_latitude"],row["start_longitude"],row["end_latitude"], row["end_longitude"]
        lid = row["id"]
        popup_txt = f"id: {lid}" if mostra_popup_id else None
        colore = _colore_random(rng)
        folium.CircleMarker(
            location=[lat1, lon1],
            radius=6,
            color=colore,
            fill=True,
            fill_opacity=0.9
        ).add_to(m)

        folium.CircleMarker(
            location=[lat2, lon2],
            radius=6,
            color=colore,
            fill=True,
            fill_opacity=0.9
        ).add_to(m)

        folium.PolyLine(
            locations=[[lat1, lon1], [lat2, lon2]],
            color=colore,
            weight=2,
            opacity=0.8,
            popup = popup_txt
        ).add_to(m)

    # # --- Plot delle EV stations (opzionale) ------------------------------------
    if show_ev:
        if not ev_file:
            raise ValueError("Per mostrare le EV stations devi passare `ev_csv`.")
        ev_list = process_newyork_ev_stations(ev_file)
        ev_group = folium.FeatureGroup(name="EV Stations").add_to(m)
        for lat, lon, ev_id in ev_list:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="#0000FF",
                fill=True,
                fill_opacity=0.7,
                popup=f"EV station {ev_id}"
            ).add_to(ev_group)

    folium.LayerControl().add_to(m)
    out_path = Path(file_html)
    m.save(out_path)
    print(f"Mappa salvata in '{out_path}'")
    return m


if __name__ == '__main__':
    # Choose one in ['denmark', 'newyork', 'chicago']
    datapath = '/mnt/c/Users/Grid/Desktop/PhD/EV/code/EV_GNN_repo/EV_GNN/data'
    dataset = 'newyork'
    if dataset == 'denmark':
        mappa_osservazioni_csv_denmark(
            os.path.join(datapath,"denmark/traffic/observation_traffic_metadata.csv"),
            file_html=os.path.join(datapath,"denmark/other/observation.html"),
            zoom_start=9,
            usa_satellite=True,
            disegna_linea=True,
            ev_file=os.path.join(datapath,"denmark/ev/other/DenamarkEVstations.json"))
    elif dataset == 'newyork':
        mappa_osservazioni_csv_newyork(
            percorso_csv=os.path.join(datapath,"newyork/traffic/stations_meta_data.csv"),
            file_html=os.path.join(datapath,"newyork/other/map_v3.html"),
            zoom_start=12,
            usa_satellite=True,
            mostra_nodi=True,
            mostra_label=True,
            mostra_popup_id=True,
            show_ev=True,
            ev_file=os.path.join(datapath,"newyork/ev/location_meta_data.csv"),
            seed_colori=123
        )
    elif dataset == 'chicago':
        mappa_osservazioni_csv_chicago(
            traffic_file=os.path.join(datapath,"chicago/traffic/location_summary.csv"),
            file_html=os.path.join(datapath,"chicago/other/map_chicago.html"),
            zoom_start=12,
            usa_satellite=True,
            mostra_nodi=True,
            mostra_label=True,
            mostra_popup_id=True,
            show_ev=True,
            ev_file=os.path.join(datapath,"chicago/ev/ev_locations_metadata.csv"),
            seed_colori=123
        )
    else:
        raise ValueError('Dataset not supported')



