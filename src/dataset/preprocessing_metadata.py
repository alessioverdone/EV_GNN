import csv
import json
import os.path

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import folium
import random
from folium.features import DivIcon
from src.dataset.visualize_data import parse_link_points, _colore_random, process_newyork_ev_stations


def process_traffic_metadata_newyork(params,
                             percorso_csv_non_processed: str,
                             file_html: str = "grafo_stradale.html",
                            zoom_start: int = 12,
                            visualize_map: bool = True,
                            save_map: bool = True,
                            usa_satellite: bool = True,
                            mostra_nodi: bool = False,
                            mostra_popup_id: bool = True,
                            mostra_label: bool = False,
                            show_ev: bool = False,  # ← flag per EV
                            ev_csv: Optional[str] = None,  # ← percorso al CSV EV
                            seed_colori: Optional[int] = 42,
                            weight: int = 3,
                            opacity: float = 0.8,
                            outlier_value_selector: int = 60):

    #  Caricamento e parsing
    df = pd.read_csv(percorso_csv_non_processed)
    if "link_points" not in df or "id" not in df:
        raise ValueError("Manca colonna 'id' o 'link_points' nel CSV.")

    # Construct __points and  __distances columns
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
            cont = 0
            pass_step = False
            for i in range(len(mask)):
                # I need this since I've 2 distance over the range but only one point is the responsible
                if pass_step:
                    pass_step = False
                    continue
                if not mask[i]:
                    _ = pts.pop(i + 1 + cont)
                    cont -= 1
                    pass_step = True

            # pts = np.array(pts)[mask]
            df.at[idx, "__points"] = pts
            df.at[idx, "__distances"] = list_dist

    # print(all_seq_distances)
    # Check archi diversi da formato standard
    df_valid = df[df["__points"].map(len) >= 2].copy()
    if df_valid.empty:
        raise ValueError("Nessun link valido (>=2 punti) trovato nel CSV.")

    df['__points'] = df['__points'].apply(lambda x: json.dumps(x))
    df['__distances'] = df['__distances'].apply(lambda x: np.array(x))
    df['__distances'] = df['__distances'].apply(lambda x: json.dumps(x.tolist()))
    path_to_save = os.path.join(params.project_path, 'data', params.dataset_name, f'traffic/processed_{params.dataset_name}_traffic_graph.csv')
    df.to_csv(str(path_to_save), sep=';', quoting=csv.QUOTE_NONNUMERIC, index=False)

    # Visualize original network
    if visualize_map or save_map:
        # --- Creazione mappa -------------------------------------------------------
        # Centro della mappa
        all_pts = [pt for pts in df_valid["__points"] for pt in pts]
        center_lat = sum(p[0] for p in all_pts) / len(all_pts)
        center_lon = sum(p[1] for p in all_pts) / len(all_pts)
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
            if not ev_csv:
                raise ValueError("Per mostrare le EV stations devi passare `ev_csv`.")
            ev_list = process_newyork_ev_stations(ev_csv)
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


def process_traffic_metadata_chicago(params,
                             percorso_csv_non_processed: str,
                             file_html: str = "grafo_stradale.html",
                            zoom_start: int = 12,
                            visualize_map: bool = True,
                            save_map: bool = True,
                            usa_satellite: bool = True,
                            mostra_nodi: bool = False,
                            mostra_popup_id: bool = True,
                            mostra_label: bool = False,
                            show_ev: bool = False,  # ← flag per EV
                            ev_csv: Optional[str] = None,  # ← percorso al CSV EV
                            seed_colori: Optional[int] = 42,
                            weight: int = 3,
                            opacity: float = 0.8,
                            outlier_value_selector: int = 60):

    #  Caricamento e parsing
    df = pd.read_csv(percorso_csv_non_processed)
    if "link_points" not in df or "id" not in df:
        raise ValueError("Manca colonna 'id' o 'link_points' nel CSV.")

    # Construct __points and  __distances columns
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
            cont = 0
            pass_step = False
            for i in range(len(mask)):
                # I need this since I've 2 distance over the range but only one point is the responsible
                if pass_step:
                    pass_step = False
                    continue
                if not mask[i]:
                    _ = pts.pop(i + 1 + cont)
                    cont -= 1
                    pass_step = True

            # pts = np.array(pts)[mask]
            df.at[idx, "__points"] = pts
            df.at[idx, "__distances"] = list_dist

    # print(all_seq_distances)
    # Check archi diversi da formato standard
    df_valid = df[df["__points"].map(len) >= 2].copy()
    if df_valid.empty:
        raise ValueError("Nessun link valido (>=2 punti) trovato nel CSV.")

    df['__points'] = df['__points'].apply(lambda x: json.dumps(x))
    df['__distances'] = df['__distances'].apply(lambda x: np.array(x))
    df['__distances'] = df['__distances'].apply(lambda x: json.dumps(x.tolist()))
    path_to_save = os.path.join(params.project_path, 'data', params.dataset_name, f'traffic/processed_{params.dataset_name}_traffic_graph.csv')
    df.to_csv(str(path_to_save), sep=';', quoting=csv.QUOTE_NONNUMERIC, index=False)

    # Visualize original network
    if visualize_map or save_map:
        # --- Creazione mappa -------------------------------------------------------
        # Centro della mappa
        all_pts = [pt for pts in df_valid["__points"] for pt in pts]
        center_lat = sum(p[0] for p in all_pts) / len(all_pts)
        center_lon = sum(p[1] for p in all_pts) / len(all_pts)
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
            if not ev_csv:
                raise ValueError("Per mostrare le EV stations devi passare `ev_csv`.")
            ev_list = process_newyork_ev_stations(ev_csv)
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
