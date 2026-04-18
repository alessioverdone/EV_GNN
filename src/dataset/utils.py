import math
import numpy as np
import torch
from itertools import combinations
import pandas as pd
from typing import  Tuple


class NodeIndexer:
    """
    Assigns a stable integer id to geographic points. Used to merge neighbours nodes
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
            # Compare new_point with saved point: in new (distance > threshold) assign new id, otherwise old (yet present)
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


def append_along_N_torch(x: torch.Tensor,
                         M,
                         fill='mean',
                         dtype=None,
                         device=None):
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
def haversine(lat1,
              lon1,
              lat2,
              lon2):
    """
    Compute Haversine distance between two coordinates

    :param lat1:
    :param lon1:
    :param lat2:
    :param lon2:
    :return:
    """
    R = 6371000  # Earth radius
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])  # convert degree in radiants

    # Deltas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # distance in Km
    return distance


def create_adjacency_matrix_newyork(source_nodes,
                                    dest_nodes,
                                    num_nodes=None,
                                    distance=None):
    if num_nodes is None:
        num_nodes = max(max(source_nodes), max(dest_nodes)) + 1

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Populete adjacency matrix
    seen_nodes = list()
    double_nodes = list()
    for src, dst, length in zip(source_nodes, dest_nodes, distance):
        if (src, dst) in seen_nodes:
            double_nodes.append((src, dst))
        adj_matrix[src, dst] = length*1000  # Set 1 to indicate the edge
        seen_nodes.append((src, dst))

    return adj_matrix, double_nodes

def create_adjacency_matrix(coordinates,
                            threshold):
    n = len(coordinates)
    adj_matrix = np.zeros((n, n))

    # Compute distances and populate the matrix
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[j]
            distance = haversine(lat1, lon1, lat2, lon2)

            # If distance < threshold, add edge with weight
            if distance < threshold:
                adj_matrix[i][j] = distance
                adj_matrix[j][i] = distance

    # 0/1 Normalization
    max_distance = np.max(adj_matrix)
    if max_distance > 0:
        adj_matrix /= max_distance

    return torch.tensor(adj_matrix).to('cuda')


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
    Adds edges to a graph defined by `edges_df` (existing edges) and `nodes_df` (nodes with lat/lon),
    so that it first becomes connected (via MST) and then reaches the desired edge fraction (`fill_pct`).

    ## Parameters

    `edges_df` : DataFrame with columns `[id, src, tgt, distance, src_id, tgt_id]` (`src`/`tgt` are coordinate tuples)
    `nodes_df` : DataFrame with columns `[node_id, lat, lon]`
    `fill_pct` : float in `[0.0, 1.0]`. `0.0` => adds only the minimum number of edges required to connect the graph.
    `1.0` => complete graph (all possible edges).
    `metric` : `"haversine"` (meters, recommended for lat/lon) or `"euclidean"` (on coordinates in degrees).
    `sort_endpoints` : if `True`, new edges are stored as `(min_id, max_id)` for consistency and deduplication.

    ## Returns

    `(edges_df_new, nodes_df_unchanged, added_edges_df)`

    * edges_df_new: edges_df with the added edges appended
    * added_edges_df: DataFrame containing only the added edges

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

    # def haversine_m(a, b):
    #     # a, b: (lat, lon) in gradi
    #     lat1, lon1 = map(math.radians, a)
    #     lat2, lon2 = map(math.radians, b)
    #     dlat = lat2 - lat1
    #     dlon = lon2 - lon1
    #     s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    #     c = 2 * math.asin(math.sqrt(s))
    #     return earth_radius_m * c

    def euclidean(a, b):
        # distanza euclidea su (lat, lon) in gradi (approssimata)
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    dist_fun = _haversine_m if metric.lower() == "haversine" else euclidean

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
    return edges_df_new, added_edges_df


# Haversine distance in meters between two (lat, lon) tuples in degrees
def _haversine_m(p, q):
    (lat1, lon1), (lat2, lon2) = p, q
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    return 2 * 6371000.0 * math.asin(math.sqrt(a))  # Earth radius ~ 6371000 m


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
                              distances_col=None):
    """
    Parameters
    ----------
    df : DataFrame with columns ['id', 'street', 'length', 'start_latitude', 'start_longitude',
       'end_latitude', 'end_longitude', 'max_speed']
    threshold : meters; 0 means exact equality, >0 merges points within threshold
    distances_col : if None, auto-picks:
        - 'distance' if present
        - otherwise '__distances' if present
        - otherwise creates a NaN distance

    Returns
    -------
    edges : DataFrame with columns [id, src, tgt, distance, src_id, tgt_id]
    nodes : DataFrame with columns [node_id, lat, lon]
    """
    # Build the minimal edge list and constructing points as tuples (lat, lon)
    df["src"] = list(zip(df["start_latitude"], df["start_longitude"]))
    df["tgt"] = list(zip(df["end_latitude"], df["end_longitude"]))
    edges = df[["id", "src", "tgt"]]

    if distances_col is None:
        edges["distance"] = float("nan")
    else:
        # Keep the value as-is to satisfy “distance remains unchanged”.
        # If you prefer total length when '__distances' is a list, replace with: sum(...) accordingly.
        edges["distance"] = df[distances_col].values

    # Assign node ids with a single scalar threshold
    indexer = NodeIndexer(threshold=threshold)
    edges["src_id"] = edges["src"].map(indexer.get_id)
    edges["tgt_id"] = edges["tgt"].map(indexer.get_id)

    nodes = indexer.nodes_df()

    # Build merge_map: orig_id -> merged_id
    # orig_id = sequential index of each unique coordinate in order of first appearance
    # (equivalent to what NodeIndexer(threshold=0) would assign)
    seen: list = []
    seen_set: set = set()
    for pt in list(edges["src"]) + list(edges["tgt"]):
        if pt not in seen_set:
            seen.append(pt)
            seen_set.add(pt)
    orig_id = {pt: i for i, pt in enumerate(seen)}
    merge_map = {orig_id[pt]: indexer.get_id(pt) for pt in seen}

    return edges, nodes, merge_map


def clean_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    tensor shape: (Nodes, Timesteps, Features)
    - Values in [-0.5, 0) are rounded to 0.
    - Values < -0.5 are replaced with the channel mean (node, feature), ignoring negative values.
    """
    tensor = tensor.float()
    # tensor[(tensor >= -0.5) & (tensor < 0)] = 0  # Round values in [-0.5, 0) to 0

    # Replace values < -0 with channel mean (without considering negative values)
    mask = tensor < 0
    tensor[mask] = float('nan')
    means = torch.nanmean(tensor, dim=1, keepdim=True)  # (Nodes, 1, Features)
    nan_mask = torch.isnan(tensor)
    tensor[nan_mask] = means.expand_as(tensor)[nan_mask]
    return tensor

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
    data = pd.read_csv('../../data/denmark/traffic/trafficMetaData.csv')

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

