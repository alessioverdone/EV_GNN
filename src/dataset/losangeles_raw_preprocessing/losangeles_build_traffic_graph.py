from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Optional, Any
import folium
import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# These MUST live at module scope (not inside `__main__`) because they are used
# both by the functions below (``MILE_TO_M`` in ``build_backbone_edges``) and by
# external callers (``DatasetLosangeles.get_edges_traffic`` reads
# ``lag.DEFAULT_META_FILE`` as the fallback metadata path). Previously they were
# defined only inside the ``if __name__ == "__main__"`` block, so importing this
# module and calling ``build_graph`` raised ``NameError``/``AttributeError``.
MILE_TO_M = 1609.344  # 1 mile in meters (PeMS postmiles are in miles)

# Project root: <...>/EV_GNN (parents[3] of this file), i.e. the folder holding data/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LA_TRAFFIC_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "losangeles", "traffic")
# The raw PeMS d07 metadata (with Fwy/Dir/Abs_PM) is stored under `other/traffic/`,
# NOT next to location_summary.csv in `traffic/`.
DEFAULT_META_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "losangeles", "other",
                                 "traffic", "Traffic stations", "d07_text_meta_2023_12_22.txt")
DEFAULT_GRAPH_DIR = os.path.join(LA_TRAFFIC_DIR, "graph")


def haversine_m(lat1,
                lon1,
                lat2,
                lon2):
    """Haversine distance in meters. Accepts scalars or numpy arrays (broadcast)."""
    R = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# 1) Loading stations
# ---------------------------------------------------------------------------
def load_stations(meta_file: str,
                  station_types: Optional[Iterable[str]] = ("ML",)) -> pd.DataFrame:
    """Load PeMS metadata and return the stations as candidate nodes.

    Returns
    -------
    DataFrame with columns: id, fwy, dir, abs_pm, lat, lon, lanes, type, name
    (only stations with valid coordinates and postmile).
    """
    meta_file = Path(meta_file)
    if not meta_file.exists():
        raise FileNotFoundError(f"PeMS metadata file not found: {meta_file}")

    raw = pd.read_csv(meta_file, sep="\t", dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]

    if station_types is not None:
        raw = raw[raw["Type"].isin(list(station_types))]

    df = pd.DataFrame({
        "id": pd.to_numeric(raw["ID"], errors="coerce").astype("Int64"),
        "fwy": pd.to_numeric(raw["Fwy"], errors="coerce").astype("Int64"),
        "dir": raw["Dir"].str.strip(),
        "abs_pm": pd.to_numeric(raw["Abs_PM"].replace("", np.nan), errors="coerce"),
        "lat": pd.to_numeric(raw["Latitude"].replace("", np.nan), errors="coerce"),
        "lon": pd.to_numeric(raw["Longitude"].replace("", np.nan), errors="coerce"),
        "lanes": pd.to_numeric(raw["Lanes"].replace("", np.nan), errors="coerce"),
        "type": raw["Type"].str.strip(),
        "name": raw["Name"].str.strip(),
    })

    before = len(df)
    df = df.dropna(subset=["id", "fwy", "abs_pm", "lat", "lon"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[load_stations] Discarded {dropped} stations without id/fwy/postmile/coordinates.")
    print(f"[load_stations] {len(df)} stations loaded "
          f"({df['fwy'].nunique()} freeways, {df.groupby(['fwy', 'dir']).ngroups} Fwy+Dir corridors).")
    return df



def build_backbone_edges(stations: pd.DataFrame,
                         directed: bool = True) -> pd.DataFrame:
    """Connect consecutive stations of each ``(Fwy, Dir)`` corridor.

    The order is given by ``abs_pm``. In directed mode edges are oriented in the
    direction of traffic: California postmiles grow from South to North and from
    West to East, so for Dir N/E the edge goes from smaller postmile (upstream) to
    larger (downstream), for Dir S/W the opposite.

    Returns
    -------
    DataFrame with columns: src_station, dst_station, distance_m, kind, fwy, dir
    """
    rows = []
    for (fwy, direction), g in stations.groupby(["fwy", "dir"]):
        g = g.sort_values("abs_pm")
        ids = g["id"].tolist()
        pms = g["abs_pm"].tolist()
        for i in range(len(ids) - 1):
            a_id, b_id = ids[i], ids[i + 1]           # a = smaller pm, b = larger pm
            dist_m = abs(pms[i + 1] - pms[i]) * MILE_TO_M
            if directed and direction in ("S", "W"):
                # traffic toward decreasing postmile: upstream = b, downstream = a
                src, dst = b_id, a_id
            else:
                # N / E (or undirected): upstream = a, downstream = b
                src, dst = a_id, b_id
            rows.append({"src_station": int(src), "dst_station": int(dst),
                         "distance_m": float(dist_m), "kind": "freeway",
                         "fwy": int(fwy), "dir": direction})

    edges = pd.DataFrame(rows, columns=["src_station", "dst_station", "distance_m", "kind", "fwy", "dir"])
    print(f"[build_backbone_edges] {len(edges)} freeway edges (backbone).")
    return edges



def build_interchange_edges(stations: pd.DataFrame,
                            threshold_m: float = 300.0,
                            one_per_pair: bool = True) -> pd.DataFrame:
    """Create edges between stations of *different* nearby freeways (interchanges).

    For each pair of distinct freeways the closest pair of stations is searched;
    if the distance is below ``threshold_m`` an interchange edge is added. With
    ``one_per_pair=False`` *all* cross-freeway station pairs below the threshold
    are added (denser graph).

    NOTE: there are no measurements on these edges. They only serve to make the
    network connected. The direction is not defined by traffic, so even in the
    directed graph they are treated as bidirectional downstream.

    Returns
    -------
    DataFrame with columns:
        src_station, dst_station, distance_m, kind, fwy_a, fwy_b, name_a, name_b
    """
    s = stations.reset_index(drop=True)
    lat = s["lat"].to_numpy()
    lon = s["lon"].to_numpy()
    fwy = s["fwy"].to_numpy()
    ids = s["id"].to_numpy()
    names = s["name"].to_numpy()

    freeways = sorted(pd.unique(fwy).tolist())
    rows = []
    for ai in range(len(freeways)):
        for bi in range(ai + 1, len(freeways)):
            fa, fb = freeways[ai], freeways[bi]
            ia = np.where(fwy == fa)[0]
            ib = np.where(fwy == fb)[0]
            # distance matrix between the stations of fa and those of fb
            d = haversine_m(lat[ia][:, None], lon[ia][:, None],
                            lat[ib][None, :], lon[ib][None, :])
            if one_per_pair:
                flat = int(np.argmin(d))
                r, c = divmod(flat, d.shape[1])
                if d[r, c] <= threshold_m:
                    rows.append((ia[r], ib[c], float(d[r, c]), fa, fb))
            else:
                rr, cc = np.where(d <= threshold_m)
                for r, c in zip(rr, cc):
                    rows.append((ia[r], ib[c], float(d[r, c]), fa, fb))

    out = pd.DataFrame([{
        "src_station": int(ids[r]), "dst_station": int(ids[c]),
        "distance_m": dist, "kind": "interchange",
        "fwy_a": int(fa), "fwy_b": int(fb),
        "name_a": names[r], "name_b": names[c],
    } for (r, c, dist, fa, fb) in rows],
        columns=["src_station", "dst_station", "distance_m", "kind",
                 "fwy_a", "fwy_b", "name_a", "name_b"])

    print(f"[build_interchange_edges] {len(out)} interchange edges "
          f"(threshold {threshold_m:.0f} m, one_per_pair={one_per_pair}).")
    return out


# ---------------------------------------------------------------------------
# 4) Graph construction + adjacency matrix
# ---------------------------------------------------------------------------
def _edge_weight(distance_m: np.ndarray,
                 weight: str,
                 sigma: Optional[float],
                 gaussian_threshold: float):
    """Return the edge weights according to the chosen scheme."""
    if weight == "binary":
        return np.ones_like(distance_m, dtype=float)
    if weight == "distance":
        return distance_m.astype(float)
    if weight == "gaussian":
        s = float(np.std(distance_m)) if sigma is None else float(sigma)
        s = s if s > 0 else 1.0
        w = np.exp(-(distance_m ** 2) / (s ** 2))
        w[w < gaussian_threshold] = 0.0
        return w
    raise ValueError(f"unknown weight: {weight!r} (use 'binary'|'distance'|'gaussian')")


def build_graph(meta_file: str,
                output_dir: Optional[str] = None,
                station_types: Optional[Iterable[str]] = ("ML",),
                directed: bool = True,
                add_interchanges: bool = True,
                interchange_threshold_m: float = 300.0,
                interchange_one_per_pair: bool = True,
                weight: str = "gaussian",
                sigma: Optional[float] = None,
                gaussian_threshold: float = 0.0,
                save: bool = True):
    """Build nodes, edges and adjacency matrix of the LA road graph.

    Parameters
    ----------
    directed : bool
        True = directed graph (backbone edges oriented in the direction of
        traffic, bidirectional interchanges). False = undirected graph (symmetric
        adjacency).
    weight : {"binary", "distance", "gaussian"}
        Edge weighting scheme. "distance" saves the meters; "gaussian" applies
        ``exp(-d^2/sigma^2)`` zeroing out below ``gaussian_threshold``.
    add_interchanges : bool
        Whether to add the interchange edges between different freeways.

    Returns
    -------
    dict with keys:
        adjacency (np.ndarray NxN), nodes (DataFrame), edges (DataFrame),
        interchanges (DataFrame), station_to_node (dict)
        :param save:
        :param gaussian_threshold:
        :param sigma:
        :param weight:
        :param interchange_one_per_pair:
        :param interchange_threshold_m:
        :param add_interchanges:
        :param directed:
        :param station_types:
        :param output_dir:
        :param meta_file:
    """
    stations = load_stations(meta_file, station_types=station_types)

    # --- Nodes: one row per station, indexed 0..N-1 (order by id) -----------
    nodes = stations.sort_values("id").reset_index(drop=True).copy()
    nodes.insert(0, "node_id", np.arange(len(nodes)))
    station_to_node = dict(zip(nodes["id"].astype(int), nodes["node_id"]))
    N = len(nodes)

    # --- Edges -------------------------------------------------------------
    backbone = build_backbone_edges(stations, directed=directed)
    if add_interchanges:
        interchanges = build_interchange_edges(
            stations, threshold_m=interchange_threshold_m, one_per_pair=interchange_one_per_pair)
    else:
        interchanges = pd.DataFrame(columns=["src_station", "dst_station", "distance_m", "kind",
                                             "fwy_a", "fwy_b", "name_a", "name_b"])

    all_edges = pd.concat([backbone[["src_station", "dst_station", "distance_m", "kind"]],
                           interchanges[["src_station", "dst_station", "distance_m", "kind"]]],
                          ignore_index=True)
    all_edges["src_node"] = all_edges["src_station"].map(station_to_node)
    all_edges["dst_node"] = all_edges["dst_station"].map(station_to_node)
    all_edges["weight"] = _edge_weight(all_edges["distance_m"].to_numpy(), weight, sigma, gaussian_threshold)

    # --- Adjacency matrix --------------------------------------------------
    A = np.zeros((N, N), dtype=float)
    for _, e in all_edges.iterrows():
        i, j, w, kind = int(e["src_node"]), int(e["dst_node"]), float(e["weight"]), e["kind"]
        if w == 0.0:
            continue
        A[i, j] = w
        # undirected -> symmetric; interchanges always bidirectional
        if (not directed) or kind == "interchange":
            A[j, i] = w

    print(f"[build_graph] Adjacency matrix {A.shape}, "
          f"{int((A != 0).sum())} non-zero entries, directed={directed}, weight='{weight}'.")

    result = {
        "adjacency": A,
        "nodes": nodes,
        "edges": all_edges,
        "interchanges": interchanges,
        "station_to_node": station_to_node,
    }

    if save:
        output_dir = Path(output_dir) if output_dir is not None else Path(DEFAULT_GRAPH_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "directed" if directed else "undirected"
        np.save(output_dir / f"adjacency_matrix_{suffix}.npy", A)
        nodes.to_csv(output_dir / "graph_nodes.csv", index=False)
        all_edges.to_csv(output_dir / f"graph_edges_{suffix}.csv", index=False)
        interchanges.to_csv(output_dir / "interchange_edges.csv", index=False)
        print(f"[build_graph] Saved adjacency/nodes/edges/interchanges in {output_dir}")

    return result


# ---------------------------------------------------------------------------
# 5) Connectivity check (NOT on map)
# ---------------------------------------------------------------------------
def check_connectivity(adjacency: np.ndarray,
                       directed: bool = True) -> dict:
    """Analyze the graph connectivity and print a textual report."""
    G = nx.from_numpy_array(adjacency, create_using=nx.DiGraph if directed else nx.Graph)

    if directed:
        comps = list(nx.weakly_connected_components(G))
        n_strong = nx.number_strongly_connected_components(G)
    else:
        comps = list(nx.connected_components(G))
        n_strong = None

    comps_sorted = sorted(comps, key=len, reverse=True)
    largest = len(comps_sorted[0]) if comps_sorted else 0
    isolated = [n for n in G.nodes if G.degree(n) == 0]
    is_connected = len(comps_sorted) == 1

    print("=" * 60)
    print("[check_connectivity] Connectivity report")
    print(f"  nodes: {G.number_of_nodes()}  |  edges: {G.number_of_edges()}")
    print(f"  {'directed' if directed else 'undirected'} graph")
    print(f"  components {'(weak)' if directed else ''}: {len(comps_sorted)}")
    if directed:
        print(f"  strongly connected components: {n_strong}")
    print(f"  largest component: {largest} nodes "
          f"({100.0 * largest / max(G.number_of_nodes(), 1):.1f}%)")
    print(f"  isolated nodes: {len(isolated)}")
    print(f"  CONNECTED: {is_connected}")
    if not is_connected:
        sizes = [len(c) for c in comps_sorted[:10]]
        print(f"  sizes of first components: {sizes}")
    print("=" * 60)

    return {
        "is_connected": is_connected,
        "n_components": len(comps_sorted),
        "n_strongly_connected": n_strong,
        "largest_component": largest,
        "isolated_nodes": isolated,
        "component_sizes": [len(c) for c in comps_sorted],
    }


# ---------------------------------------------------------------------------
# 6) Visualization on map
# ---------------------------------------------------------------------------
def visualize_graph_on_map(nodes: Any,
                           edges: Any,
                           output_html: str,
                           show_nodes: bool = True):
    """Draw nodes and edges on an interactive map (folium) and save an HTML.

    - nodes: blue dots (geographic position of the stations)
    - freeway edges (backbone): blue lines
    - interchange edges: red lines (thicker)
    """
    node_xy = nodes.set_index("node_id")[["lat", "lon"]].to_dict("index")
    center = [nodes["lat"].mean(), nodes["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")

    backbone_layer = folium.FeatureGroup(name="Freeway edges (backbone)")
    interchange_layer = folium.FeatureGroup(name="Interchange edges")

    for _, e in edges.iterrows():
        a = node_xy.get(int(e["src_node"]))
        b = node_xy.get(int(e["dst_node"]))
        if a is None or b is None:
            continue
        coords = [(a["lat"], a["lon"]), (b["lat"], b["lon"])]
        if e["kind"] == "interchange":
            folium.PolyLine(coords, color="red", weight=4, opacity=0.9).add_to(interchange_layer)
        else:
            folium.PolyLine(coords, color="blue", weight=2, opacity=0.6).add_to(backbone_layer)

    backbone_layer.add_to(fmap)
    interchange_layer.add_to(fmap)

    if show_nodes:
        node_layer = folium.FeatureGroup(name="Stations (nodes)")
        for _, n in nodes.iterrows():
            folium.CircleMarker(
                location=(n["lat"], n["lon"]), radius=2,
                color="#1f3a93", fill=True, fill_opacity=0.8,
                popup=f"node {int(n['node_id'])} | station {int(n['id'])} | Fwy {int(n['fwy'])}-{n['dir']}",
            ).add_to(node_layer)
        node_layer.add_to(fmap)

    folium.LayerControl().add_to(fmap)
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_html))
    print(f"[visualize_graph_on_map] Map saved in {output_html}")
    return str(output_html)



if __name__ == "__main__":
    """
    los_angeles_graph.py
    ====================

    Construction of the road graph of the "losangeles" dataset (PeMS District 7)
    from the station metadata, and saving of the adjacency matrix.

    Core idea
    ---------
    PeMS stations are point-like, but the metadata contain the topological
    information of the road network:

    - ``Fwy``   : freeway number (e.g. 5, 405, 10, ...)
    - ``Dir``   : direction of travel (N, S, E, W)
    - ``Abs_PM``: *absolute postmile*, i.e. the linear distance along the route of
      the freeway. It is a 1D coordinate intrinsic to the road.

    The graph is therefore built in two steps:

    1. **Freeway backbone** — within each ``(Fwy, Dir)`` group the stations are
       ordered by ``Abs_PM`` and connected in sequence. The edge weight is the
       postmile difference (real road distance, follows the curves).

    2. **Interchange edges** — the per-freeway chains are disconnected from each
       other. To connect the corridors, edges are added between stations of
       *different* freeways that are closer than a distance threshold (the
       interchanges). WARNING: there are no measurements on these edges (there is
       no "interchange" station); they only represent the physical connectivity of
       the network. They are marked with ``kind == "interchange"`` and their list
       is printed/saved separately, so that downstream you can treat them as you
       prefer in forecasting.

    Available functions
    -------------------
    - :func:`load_stations`           load and filter the stations from metadata
    - :func:`build_backbone_edges`    consecutive edges per (Fwy, Dir) via postmile
    - :func:`build_interchange_edges` interchange edges between different freeways
    - :func:`build_graph`             orchestrator: nodes, edges, adjacency matrix
    - :func:`check_connectivity`      connectivity report (NOT on map)
    - :func:`visualize_graph_on_map`  HTML map (nodes, edges, freeway edges)
    """

    RUN_BUILD_GRAPH = True
    RUN_CHECK_CONNECTIVITY = True
    RUN_VISUALIZE = True
    DIRECTED = False               # True = directed graph, False = undirected
    STATION_TYPES = ("ML",)       # consistent with metadata and measurements
    WEIGHT = "gaussian"           # "binary" | "distance" | "gaussian"
    INTERCHANGE_THRESHOLD_M = 300.0
    INTERCHANGE_ONE_PER_PAIR = True
    # Path constants (PROJECT_ROOT, LA_TRAFFIC_DIR, DEFAULT_META_FILE,
    # DEFAULT_GRAPH_DIR) and MILE_TO_M now live at module scope (top of file).

    if RUN_BUILD_GRAPH:
        res = build_graph(
            meta_file=DEFAULT_META_FILE,
            output_dir=DEFAULT_GRAPH_DIR,
            station_types=STATION_TYPES,
            directed=DIRECTED,
            add_interchanges=True,
            interchange_threshold_m=INTERCHANGE_THRESHOLD_M,
            interchange_one_per_pair=INTERCHANGE_ONE_PER_PAIR,
            weight=WEIGHT,
            save=True,
        )

        # List of the interchange edges (there are NO measurements on these)
        inter = res["interchanges"]
        print(f"\n--- Interchange edges ({len(inter)}) ---")
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(inter[["fwy_a", "fwy_b", "src_station", "dst_station", "distance_m", "name_a", "name_b"]])

        if RUN_CHECK_CONNECTIVITY:
            check_connectivity(res["adjacency"], directed=DIRECTED)

        if RUN_VISUALIZE:
            visualize_graph_on_map(res["nodes"],
                                   res["edges"],
                                   output_html=os.path.join(DEFAULT_GRAPH_DIR, f"graph_map_{'directed' if DIRECTED else 'undirected'}.html"))