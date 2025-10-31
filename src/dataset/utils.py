import csv
import json
import math
import ast
import os.path

import numpy as np
import torch
from itertools import combinations
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import folium
import random
import re
from folium.features import DivIcon
from src.dataset.visualize_data import parse_link_points, _colore_random, process_newyork_ev_stations


def append_along_N_torch(x: torch.Tensor, M, fill='mean', dtype=None, device=None):
    if x.ndim != 3:
        raise ValueError("x deve avere shape (N, T, F)")
    N, T, F = x.shape
    if dtype is None:  dtype = x.dtype
    if device is None: device = x.device

    if fill == 'mean':
        base = x.to(torch.float32).mean(dim=0)  # (T, F)
        base = base.to(dtype)
    elif fill == 'ones':
        base = torch.ones((T, F), dtype=dtype, device=device)
    elif fill == 'zeros':
        base = torch.zeros((T, F), dtype=dtype, device=device)
    elif isinstance(fill, (int, float)):
        base = torch.full((T, F), fill, dtype=dtype, device=device)
    else:
        raise ValueError("fill deve essere 'mean', 'ones', 'zeros' oppure uno scalare")

    extra = base.unsqueeze(0).repeat(M, 1, 1)
    return torch.cat([x.to(dtype=dtype, device=device), extra], dim=0)


# Funzione per calcolare la distanza Haversine tra due coordinate
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Raggio della Terra in km
    # Convertire gradi in radianti
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Differenze tra latitudini e longitudini
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Formula dell'Haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distanza in km
    distance = R * c
    return distance


def create_adjacency_matrix_newyork(source_nodes, dest_nodes, num_nodes=None, distance=None):
    # Determina il numero di nodi se non specificato
    if num_nodes is None:
        num_nodes = max(max(source_nodes), max(dest_nodes)) + 1

    # Inizializza la matrice di adiacenza con zeri
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Popola la matrice di adiacenza
    seen_nodes = list()
    double_nodes = list()
    for src, dst, length in zip(source_nodes, dest_nodes, distance):
        if (src, dst) in seen_nodes:
            double_nodes.append((src, dst))
        adj_matrix[src, dst] = length*1000  # Imposta 1 per indicare un arco
        seen_nodes.append((src, dst))


    return adj_matrix, double_nodes

# Funzione per creare la matrice di adiacenza
def create_adjacency_matrix(coordinates, threshold):
    n = len(coordinates)
    # Inizializza la matrice di adiacenza con valori a zero
    adj_matrix = np.zeros((n, n))

    # Calcola le distanze e popola la matrice
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[j]
            distance = haversine(lat1, lon1, lat2, lon2)

            # Se la distanza è inferiore alla soglia, aggiungi l'arco con il peso
            if distance < threshold:
                adj_matrix[i][j] = distance
                adj_matrix[j][i] = distance

    # Normalizzazione: Dividi per la distanza massima per ottenere valori tra 0 e 1
    max_distance = np.max(adj_matrix)
    if max_distance > 0:
        adj_matrix /= max_distance

    return torch.tensor(adj_matrix).to('cuda')


# def augment_graph_df(
#         df,
#         fill_pct: float = 0.0,
#         id_col: str = 'id',
#         points_col: str = '__points'
# ) -> pd.DataFrame:
#     """
#     Restituisce un nuovo DataFrame con archi aggiunti a filter_traffic_raw_df.
#
#     - fill_pct: tra 0.0 e 1.0, frazione di archi sul totale possibile (grafo completo).
#     - L'MST garantisce connessione, poi si aggiungono archi più piccoli per raggiungere fill_pct.
#     """
#     # 1) Copia e assicurati di avere liste di tuple
#     df = df.copy()
#     df[points_col] = df[points_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
#
#     # 2) Estrai nodi unici (tuple) e mappa a indici
#     nodes = sorted({tuple(pt) for seg in df[points_col] for pt in seg})
#     idx_of = {node: i for i, node in enumerate(nodes)}
#     n = len(nodes)
#
#     # 3) Archi esistenti come set di frozenset({i,j})
#     existing = {
#         frozenset((idx_of[tuple(seg[0])], idx_of[tuple(seg[1])]))
#         for seg in df[points_col]
#     }
#
#     # 4) Tutti i possibili archi candidati (i,j, distanza) non già in existing
#     candidates = []
#     for i, j in combinations(range(n), 2):
#         if frozenset((i, j)) not in existing:
#             u, v = nodes[i], nodes[j]
#             dist = np.hypot(u[0] - v[0], u[1] - v[1])
#             candidates.append((i, j, dist))
#     # Ordina per distanza crescente
#     candidates.sort(key=lambda x: x[2])
#
#     # 5) Kruskal per MST (garantisce connessione)
#     parent = list(range(n))
#
#     def find(x):
#         while parent[x] != x:
#             parent[x] = parent[parent[x]]
#             x = parent[x]
#         return x
#
#     def union(a, b):
#         ra, rb = find(a), find(b)
#         if ra == rb:
#             return False
#         parent[rb] = ra
#         return True
#
#     new_edges = []
#     for i, j, _ in candidates:
#         if union(i, j):
#             new_edges.append((i, j))
#
#     # 6) Calcola quanti archi totali vogliamo: fill_pct * [n*(n-1)/2]
#     max_total_edges = n * (n - 1) // 2
#     desired_total = int(fill_pct * max_total_edges)
#
#     # 7) Aggiungi altri archi (i,j) fino a desired_total, sempre prendendo i candidati più piccoli
#     already = len(existing) + len(new_edges)
#     to_add = max(0, desired_total - already)
#     added = 0
#     for i, j, _ in candidates:
#         if added >= to_add:
#             break
#         fs = frozenset((i, j))
#         if fs in existing:
#             continue
#         # evita duplicati con quelli già in new_edges
#         if any({i, j} == set(e) for e in new_edges):
#             continue
#         new_edges.append((i, j))
#         added += 1
#
#     # 8) Crea le nuove righe con id progressivo e __points = [nodo_i, nodo_j]
#     max_id = df[id_col].max()
#     new_rows = []
#     for k, (i, j) in enumerate(new_edges, start=1):
#         new_rows.append({
#             id_col: int(max_id + k),
#             points_col: [nodes[i], nodes[j]]
#         })
#
#     # 9) Ritorna DataFrame esteso
#     return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
#
#
# def augment_graph_df_v2(
#         df,
#         fill_pct: float = 0.0,
#         id_col: str = 'id',
#         points_col: str = '__points'
# ) -> pd.DataFrame:
#     #TODO: Continua qui
#     """
#     Restituisce un nuovo DataFrame con archi aggiunti a filter_traffic_raw_df.
#
#     - fill_pct: tra 0.0 e 1.0, frazione di archi sul totale possibile (grafo completo).
#     - L'MST garantisce connessione, poi si aggiungono archi più piccoli per raggiungere fill_pct.
#     """
#     # 1) Copia e assicurati di avere liste di tuple
#     df = df.copy()
#     # df[points_col] = df[points_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
#
#     # 2) Estrai nodi unici (tuple) e mappa a indici
#     nodes = sorted({tuple(pt) for seg in df[points_col] for pt in seg})
#     idx_of = {node: i for i, node in enumerate(nodes)}
#     n = len(nodes)
#
#     # 3) Archi esistenti come set di frozenset({i,j})
#     existing = {
#         frozenset((idx_of[tuple(seg[0])], idx_of[tuple(seg[1])]))
#         for seg in df[points_col]
#     }
#
#     # 4) Tutti i possibili archi candidati (i,j, distanza) non già in existing
#     candidates = []
#     for i, j in combinations(range(n), 2):
#         if frozenset((i, j)) not in existing:
#             u, v = nodes[i], nodes[j]
#             dist = np.hypot(u[0] - v[0], u[1] - v[1])
#             candidates.append((i, j, dist))
#     # Ordina per distanza crescente
#     candidates.sort(key=lambda x: x[2])
#
#     # 5) Kruskal per MST (garantisce connessione)
#     parent = list(range(n))
#
#     def find(x):
#         while parent[x] != x:
#             parent[x] = parent[parent[x]]
#             x = parent[x]
#         return x
#
#     def union(a, b):
#         ra, rb = find(a), find(b)
#         if ra == rb:
#             return False
#         parent[rb] = ra
#         return True
#
#     new_edges = []
#     for i, j, _ in candidates:
#         if union(i, j):
#             new_edges.append((i, j))
#
#     # 6) Calcola quanti archi totali vogliamo: fill_pct * [n*(n-1)/2]
#     max_total_edges = n * (n - 1) // 2
#     desired_total = int(fill_pct * max_total_edges)
#
#     # 7) Aggiungi altri archi (i,j) fino a desired_total, sempre prendendo i candidati più piccoli
#     already = len(existing) + len(new_edges)
#     to_add = max(0, desired_total - already)
#     added = 0
#     for i, j, _ in candidates:
#         if added >= to_add:
#             break
#         fs = frozenset((i, j))
#         if fs in existing:
#             continue
#         # evita duplicati con quelli già in new_edges
#         if any({i, j} == set(e) for e in new_edges):
#             continue
#         new_edges.append((i, j))
#         added += 1
#
#     # 8) Crea le nuove righe con id progressivo e __points = [nodo_i, nodo_j]
#     max_id = df[id_col].max()
#     new_rows = []
#     for k, (i, j) in enumerate(new_edges, start=1):
#         new_rows.append({
#             id_col: int(max_id + k),
#             points_col: [nodes[i], nodes[j]]
#         })
#
#     # 9) Ritorna DataFrame esteso
#     return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def augment_graph_df_v3(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    fill_pct: float = 0.001,
    id_col: str = "id",
    src_id_col: str = "src_id",
    tgt_id_col: str = "tgt_id",
    src_col: str = "src",
    tgt_col: str = "tgt",
    distance_col: str = "distance",
    metric: str = "haversine",   # "haversine" (metri, default) oppure "euclidean" (sulle coordinate grezze)
    earth_radius_m: float = 6_371_000.0,
    sort_endpoints: bool = True, # normalizza i nuovi archi come (min_id, max_id)
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggiunge archi a un grafo definito da edges_df (archi esistenti) e nodes_df (nodi con lat/lon),
    in modo da renderlo connesso (MST) e poi raggiungere la frazione di archi desiderata (fill_pct).

    Parametri
    ---------
    edges_df : DataFrame con colonne [id, src, tgt, distance, src_id, tgt_id] (src/tgt sono coord tuple)
    nodes_df : DataFrame con colonne [node_id, lat, lon]
    fill_pct : float in [0.0, 1.0]. 0.0 => aggiunge solo gli archi minimi necessari per connettere il grafo.
               1.0 => grafo completo (tutti i possibili archi).
    metric   : "haversine" (metri, consigliata per lat/lon) oppure "euclidean" (sulle coordinate in gradi).
    sort_endpoints : se True, i nuovi archi sono inseriti come (min_id, max_id) per coerenza e deduplica.

    Ritorna
    -------
    (edges_df_nuovo, nodes_df_invariato, added_edges_df)
      - edges_df_nuovo: edges_df con gli archi aggiunti in coda
      - added_edges_df: DataFrame con i soli archi aggiunti
    """
    # --- Validazioni di base ---
    if not (0.0 <= fill_pct <= 1.0):
        raise ValueError("fill_pct dev'essere tra 0.0 e 1.0")

    required_node_cols = {"node_id", "lat", "lon"}
    if not required_node_cols.issubset(nodes_df.columns):
        raise ValueError(f"nodes_df deve contenere le colonne {required_node_cols}")

    required_edge_cols = {src_id_col, tgt_id_col}
    if not required_edge_cols.issubset(edges_df.columns):
        raise ValueError(f"edges_df deve contenere almeno le colonne {required_edge_cols}")

    # --- Mappature e strutture di supporto ---
    # user -> idx compatti [0..n-1] per Union-Find
    nodes_df_local = nodes_df.copy()
    node_ids = nodes_df_local["node_id"].astype(int).tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    idx_to_id = {i: nid for nid, i in id_to_idx.items()}

    # dizionario id_nodo -> (lat, lon)
    coords = {
        int(row["node_id"]): (float(row["lat"]), float(row["lon"]))
        for _, row in nodes_df_local.iterrows()
    }

    n = len(node_ids)
    if n < 2:
        # niente da fare
        return edges_df.copy(), pd.DataFrame(columns=[id_col, src_col, tgt_col, distance_col, src_id_col, tgt_id_col])

    def haversine_m(a, b):
        # a, b: (lat, lon) in gradi
        lat1, lon1 = map(math.radians, a)
        lat2, lon2 = map(math.radians, b)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(s))
        return earth_radius_m * c

    def euclidean(a, b):
        # distanza euclidea su (lat, lon) in gradi (approssimata)
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    dist_fun = haversine_m if metric.lower() == "haversine" else euclidean

    # --- Insieme archi esistenti (non orientati) usando gli id nodo ---
    existing_pairs = set()
    for _, r in edges_df.iterrows():
        u_id = int(r[src_id_col])
        v_id = int(r[tgt_id_col])
        # normalizza eventuali orientamenti
        if sort_endpoints and u_id > v_id:
            u_id, v_id = v_id, u_id
        # escludi archi verso nodi non presenti (se capitasse)
        if u_id in id_to_idx and v_id in id_to_idx and u_id != v_id:
            existing_pairs.add(frozenset((id_to_idx[u_id], id_to_idx[v_id])))

    # --- Genera la lista completa dei candidati non ancora esistenti: (i, j, dist) ---
    candidates = []
    for i, j in combinations(range(n), 2):
        fs = frozenset((i, j))
        if fs in existing_pairs:
            continue
        u_id, v_id = idx_to_id[i], idx_to_id[j]
        d = dist_fun(coords[u_id], coords[v_id])
        candidates.append((i, j, d))

    # ordina per distanza crescente
    candidates.sort(key=lambda x: x[2])

    # --- Union-Find inizializzato con gli archi esistenti (così preserviamo i componenti già connessi) ---
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        # union by rank
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    # unisci gli archi esistenti
    for fs in existing_pairs:
        (i, j) = tuple(fs)
        union(i, j)

    # conta i componenti iniziali
    def components_count():
        roots = {find(i) for i in range(n)}
        return len(roots)

    comps = components_count()

    # --- 1) Aggiungi archi MST minimi necessari per connettere il grafo ---
    mst_edges_idx = []  # lista di (i, j) sugli indici compatti
    if comps > 1:
        for i, j, _ in candidates:
            if union(i, j):
                mst_edges_idx.append((i, j))
                comps -= 1
                if comps == 1:
                    break

    # --- 2) Calcola quanti archi totali vogliamo (sul grafo completo non orientato) ---
    max_total_edges = n * (n - 1) // 2
    desired_total = int(fill_pct * max_total_edges)

    already = len(existing_pairs) + len(mst_edges_idx)
    to_add_extra = max(0, desired_total - already)

    # --- 3) Aggiungi ulteriori archi più corti (oltre all’MST) fino a raggiungere desired_total ---
    added_extra_idx = []
    if to_add_extra > 0:
        # set per lookup rapido di ciò che andremo ad aggiungere
        mst_fs = {frozenset((i, j)) for (i, j) in mst_edges_idx}
        added_fs = set()
        for i, j, _ in candidates:
            if len(added_extra_idx) >= to_add_extra:
                break
            fs = frozenset((i, j))
            if fs in existing_pairs or fs in mst_fs or fs in added_fs:
                continue
            added_extra_idx.append((i, j))
            added_fs.add(fs)

    # --- 4) Costruisci i DataFrame degli archi nuovi (MST + extra) ---
    new_edges_idx = mst_edges_idx + added_extra_idx

    # Normalizza orientamento (src_id < tgt_id) se richiesto; calcola distance e coord
    def row_from_idx(i, j):
        u_id, v_id = idx_to_id[i], idx_to_id[j]
        u_coord, v_coord = coords[u_id], coords[v_id]
        d = dist_fun(u_coord, v_coord)
        if sort_endpoints and u_id > v_id:
            u_id, v_id = v_id, u_id
            u_coord, v_coord = v_coord, u_coord
        return {
            src_id_col: u_id,
            tgt_id_col: v_id,
            src_col: tuple(u_coord),
            tgt_col: tuple(v_coord),
            distance_col: float(d),
        }

    new_rows = [row_from_idx(i, j) for (i, j) in new_edges_idx]

    if not new_rows:
        # nessun arco da aggiungere
        return edges_df.copy(), pd.DataFrame(columns=[id_col, src_col, tgt_col, distance_col, src_id_col, tgt_id_col])

    added_edges_df = pd.DataFrame(new_rows)

    # assegna id progressivi partendo dal max esistente
    if edges_df.shape[0] > 0 and id_col in edges_df.columns and pd.api.types.is_numeric_dtype(edges_df[id_col]):
        start_id = int(edges_df[id_col].max()) + 1
    else:
        start_id = 1
    added_edges_df.insert(0, id_col, range(start_id, start_id + len(added_edges_df)))

    # ordina le colonne in modo "amichevole" se possibile
    col_order = [id_col, src_col, tgt_col, distance_col, src_id_col, tgt_id_col]
    for c in col_order:
        if c not in added_edges_df.columns:
            col_order.remove(c)
    added_edges_df = added_edges_df[col_order + [c for c in added_edges_df.columns if c not in col_order]]

    # --- 5) Concatena agli archi esistenti e restituisci ---
    edges_df_new = pd.concat([edges_df.copy(), added_edges_df], ignore_index=True)

    # --- 5) Concatena agli archi esistenti e restituisci ---
    # --- 6) Controlla archi doppi ---
    # Combina gli archi esistenti con quelli nuovi (MST + extra)
    # all_edges = pd.concat([edges_df, added_edges_df], ignore_index=True)
    #
    # # Crea un set di archi unici (normalizzando l'orientamento)
    # unique_edges = set()
    # duplicates = []
    # #src_id 73 e 75 originali sono uguali
    #
    # for _, row in all_edges.iterrows():
    #     u_id = int(row[src_id_col])
    #     v_id = int(row[tgt_id_col])
    #     # Normalizza l'orientamento degli archi
    #     if u_id > v_id:
    #         u_id, v_id = v_id, u_id
    #     edge = frozenset((u_id, v_id))
    #     if edge in unique_edges:
    #         duplicates.append(row)
    #     else:
    #         unique_edges.add(edge)
    #
    # # Rimuove gli archi duplicati dai nuovi archi da aggiungere
    # # added_edges_df = added_edges_df[~added_edges_df.apply(lambda row:
    # #     frozenset((min(int(row[src_id_col]), int(row[tgt_id_col])),
    # #                max(int(row[src_id_col]), int(row[tgt_id_col])))) in unique_edges, axis=1)]
    # added_edges_df = added_edges_df[~added_edges_df['id'].isin([elem['id'] for elem in duplicates])]
    #
    # # Ricostruisci gli archi unici dopo la rimozione dei duplicati
    # edges_df_new = pd.concat([edges_df, added_edges_df], ignore_index=True)
    return edges_df_new, added_edges_df




# Haversine distance in meters between two (lat, lon) tuples in degrees
def _haversine_m(p, q):
    (lat1, lon1), (lat2, lon2) = p, q
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    return 2 * 6371000.0 * math.asin(math.sqrt(a))  # Earth radius ~ 6371000 m

class NodeIndexer:
    """
    Assigns a stable integer id to geographic points.
    threshold == 0 -> exact tuple equality
    threshold > 0  -> greedy clustering: first centroid within threshold meters
    """
    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)
        self._centroids = []      # list[(lat, lon)]
        self._exact_map = {}      # dict[(lat, lon)] -> id (used when threshold == 0)

    def get_id(self, point):
        if self.threshold == 0.0:
            # exact match only
            if point in self._exact_map:
                return self._exact_map[point]
            nid = len(self._centroids)
            self._centroids.append(point)
            self._exact_map[point] = nid
            return nid
        else:
            # fuzzy: attach to first centroid within threshold meters, else create new
            for nid, c in enumerate(self._centroids):
                if _haversine_m(point, c) <= self.threshold:
                    return nid
            nid = len(self._centroids)
            self._centroids.append(point)
            return nid

    def nodes_df(self):
        # node_id, lat, lon
        return pd.DataFrame(
            [(i, p[0], p[1]) for i, p in enumerate(self._centroids)],
            columns=["node_id", "lat", "lon"]
        )

def build_edges_with_node_ids(df: pd.DataFrame,
                              threshold: float = 0.0,
                              points_col="__points",
                              distances_col=None):
    """
    Parameters
    ----------
    df : DataFrame with columns at least: 'id', '__points' (list of (lat, lon))
           optionally it may have 'distance' or '__distances'
    threshold : meters; 0 means exact equality, >0 merges points within threshold
    points_col : name of the column containing the list of coordinates
    distances_col : if None, auto-picks:
        - 'distance' if present
        - otherwise '__distances' if present
        - otherwise creates a NaN distance

    Returns
    -------
    edges : DataFrame with columns [id, src, tgt, distance, src_id, tgt_id]
    nodes : DataFrame with columns [node_id, lat, lon]
    """
    # build the minimal edge list
    edges = pd.DataFrame({
        "id": df["id"].values,
        "src": df[points_col].map(lambda pts: tuple(pts[0])),
        "tgt": df[points_col].map(lambda pts: tuple(pts[-1])),
    })
    if distances_col is None:
        edges["distance"] = float("nan")
    else:
        # Keep the value as-is to satisfy “distance remains unchanged”.
        # If you prefer total length when '__distances' is a list, replace with: sum(...) accordingly.
        edges["distance"] = df[distances_col].values

    # assign node ids with a single scalar threshold
    indexer = NodeIndexer(threshold=threshold)
    edges["src_id"] = edges["src"].map(indexer.get_id)
    edges["tgt_id"] = edges["tgt"].map(indexer.get_id)

    nodes = indexer.nodes_df()
    return edges, nodes

def build_edges_with_node_ids_chicago(df: pd.DataFrame,
                              threshold: float = 0.0,
                              points_col="__points",
                              distances_col=None):
    """
    Parameters
    ----------
    df : DataFrame with columns at least: 'id', '__points' (list of (lat, lon))
           optionally it may have 'distance' or '__distances'
    threshold : meters; 0 means exact equality, >0 merges points within threshold
    points_col : name of the column containing the list of coordinates
    distances_col : if None, auto-picks:
        - 'distance' if present
        - otherwise '__distances' if present
        - otherwise creates a NaN distance

    Returns
    -------
    edges : DataFrame with columns [id, src, tgt, distance, src_id, tgt_id]
    nodes : DataFrame with columns [node_id, lat, lon]
    """
    # build the minimal edge list
    # edges = pd.DataFrame({
    #     "id": df["id"].values,
    #     "src": df[points_col].map(lambda pts: tuple(pts[0])),
    #     "tgt": df[points_col].map(lambda pts: tuple(pts[-1])),
    # })

    # costruzione punti come tuple (lat, lon)
    df["src"] = list(zip(df["start_latitude"], df["start_longitude"]))
    df["tgt"] = list(zip(df["end_latitude"], df["end_longitude"]))
    edges = df[["id", "src", "tgt"]]

    if distances_col is None:
        edges["distance"] = float("nan")
    else:
        # Keep the value as-is to satisfy “distance remains unchanged”.
        # If you prefer total length when '__distances' is a list, replace with: sum(...) accordingly.
        edges["distance"] = df[distances_col].values

    # assign node ids with a single scalar threshold
    indexer = NodeIndexer(threshold=threshold)
    edges["src_id"] = edges["src"].map(indexer.get_id)
    edges["tgt_id"] = edges["tgt"].map(indexer.get_id)

    nodes = indexer.nodes_df()
    return edges, nodes


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


if __name__ == '__main__':
    """
    Examples of usage for the `build_edges_with_node_ids` function:

    1. To get an exact match between the data with no tolerance:
       edges_df, nodes_df = build_edges_with_node_ids(df, threshold=0.0)

    2. To allow a tolerance of 3 meters, enabling a more flexible match:
       edges_df, nodes_df = build_edges_with_node_ids(df, threshold=3.0)

    3. To allow a tolerance of 10 meters, including more variability in the data:
       edges_df, nodes_df = build_edges_with_node_ids(df, threshold=10.0)
    """

    # Load traffic data from the CSV file to analyze the geographic coordinates
    data = pd.read_csv(r'/mnt/c/Users/Grid/Desktop/PhD/EV/EV_GNN/src/dataset/denmark/trafficMetaData.csv')

    # Create an empty list to store the average points of the stations
    list_of_stations = list()

    # Iterate through the data and calculate the average coordinates of the points
    for row in data.iterrows():
        p1_lat = row[1]['POINT_1_LAT']
        p1_long = row[1]['POINT_1_LNG']
        p2_lat = row[1]['POINT_2_LAT']
        p2_long = row[1]['POINT_2_LNG']

        # Calculate the average latitude and longitude of the segment
        p_mean_lat = (p1_lat + p2_lat) / 2.0
        p_mean_lng = (p1_long + p2_long) / 2.0

        # Add the average point to our list
        list_of_stations.append((p_mean_lat, p_mean_lng))

    # Print the results: the list of all the calculated station points
    print(list_of_stations)

