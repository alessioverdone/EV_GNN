import ast
import os
from types import SimpleNamespace

import pandas as pd
import matplotlib.pyplot as plt

from src.config import Parameters
from src.dataset.utils import augment_graph_df_v3


def _build_points_column(dataset_name, traffic_metadata, points_col):
    """Build a homogeneous `points_col` of [[lat, lon], ...] segments regardless
    of the raw traffic metadata layout, which differs per dataset:

    - newyork:  a `__points` column already holding a list of [lat, lon] points
                (only the first and last are kept to draw the segment).
    - chicago:  no points list, but explicit start/end coordinates
                (`start_latitude/longitude`, `end_latitude/longitude`).
    - losangeles: NOT handled here -- see `_build_losangeles_graph` (the raw
                metadata are point-like PeMS stations and the graph topology is
                reconstructed from freeway/direction/postmile information).
    """
    if dataset_name == 'chicago':
        # Each row is a single segment: [start_point, end_point]
        traffic_metadata[points_col] = traffic_metadata.apply(
            lambda row: [[row['start_latitude'], row['start_longitude']],
                         [row['end_latitude'], row['end_longitude']]],
            axis=1)
        return traffic_metadata

    # Default (newyork): parse the stringified list and keep endpoints only
    traffic_metadata[points_col] = traffic_metadata[points_col].apply(ast.literal_eval)
    traffic_metadata[points_col] = (
        traffic_metadata[points_col]
        .apply(lambda pts: [pts[0], pts[-1]] if isinstance(pts, list) and len(pts) > 1 else pts))
    return traffic_metadata


def _build_losangeles_graph(traffic_raw_metadata_file,
                            interchange_threshold_m=100.0):
    """Reconstruct the Los Angeles road-graph *topology* from the raw PeMS d07
    metadata, mirroring the strategy of ``DatasetLosangeles.get_edges_traffic``
    (see ``losangeles.py``):

      - nodes            = PeMS stations (point-like, one row per station);
      - backbone edges   = consecutive stations of each ``(Fwy, Dir)`` corridor,
                           ordered by absolute postmile;
      - interchange edges = links between stations of *different* nearby freeways
                            (< ``interchange_threshold_m``) to bridge the corridors.

    Only the *raw* topology is returned (no intersection with the stations that
    actually have temporal data, and no ``augment_graph_df_v3`` connectivity
    edges), consistent with how the raw visualization of chicago/newyork shows the
    plain metadata.

    Returns
    -------
    (nodes, edges) : the DataFrames produced by ``build_graph`` -- ``nodes`` has at
        least ``node_id/lat/lon``; ``edges`` has ``src_node/dst_node/kind``.
    """
    from src.dataset.losangeles_raw_preprocessing import losangeles_build_traffic_graph as lag

    # Same call as DatasetLosangeles.get_edges_traffic (output_dir unused: save=False).
    graph = lag.build_graph(
        meta_file=traffic_raw_metadata_file,
        station_types=("ML",),
        directed=False,                 # undirected, coherent with the raw view
        add_interchanges=True,
        interchange_threshold_m=float(interchange_threshold_m),
        weight="distance",
        save=False)
    return graph["nodes"], graph["edges"]


def _plot_losangeles_graph(traffic_raw_metadata_file,
                           interchange_threshold_m=100.0):
    """Draw the LA raw topology onto the current matplotlib axes: freeway backbone
    edges, interchange edges (highlighted) and the station nodes."""
    nodes, edges = _build_losangeles_graph(traffic_raw_metadata_file,
                                           interchange_threshold_m)
    print(f'Loaded LA graph: {len(nodes)} stations (nodes), {len(edges)} edges!')

    node_xy = nodes.set_index('node_id')[['lat', 'lon']].to_dict('index')

    # Draw edges once per kind so the legend stays compact (label only the first).
    backbone_labelled = False
    interchange_labelled = False
    for _, e in edges.iterrows():
        a = node_xy.get(int(e['src_node']))
        b = node_xy.get(int(e['dst_node']))
        if a is None or b is None:
            continue
        lons = [a['lon'], b['lon']]
        lats = [a['lat'], b['lat']]
        if e['kind'] == 'interchange':
            plt.plot(lons, lats, color='red', linewidth=1.5, alpha=0.9,
                     label=None if interchange_labelled else 'Interchange edges')
            interchange_labelled = True
        else:
            plt.plot(lons, lats, color='C1', linewidth=0.8, alpha=0.6,
                     label=None if backbone_labelled else 'Freeway edges (backbone)')
            backbone_labelled = True

    # Station nodes on top of the edges
    plt.scatter(nodes['lon'], nodes['lat'], c='C1', marker='o', s=6,
                edgecolor='black', linewidth=0.2, label='Traffic nodes')


def visualize_real_original_graph(dataset_name,
                                  preprocessed_dataset_metadata,
                                  traffic_metadata_file,
                                  ev_metadata_csv,
                                  sep=',',
                                  augment_graph = False,
                                  augment_factor = 0.0001,
                                  points_col = '__points',
                                  traffic_raw_metadata_file = None,
                                  interchange_threshold_m = 100.0):


    # Load EV data. The metadata layout matches between chicago/newyork
    # ('Latitude'/'Longitude'), while losangeles uses 'lat'/'lng' (see losangeles.py),
    # so resolve the coordinate column names instead of hardcoding them.
    ev_metadata = pd.read_csv(ev_metadata_csv)
    print(f'Loaded ev_metadata with shape: {ev_metadata.shape[0]}!')
    if 'Latitude' in ev_metadata.columns:
        ev_lat_col, ev_lon_col = 'Latitude', 'Longitude'
    elif 'lat' in ev_metadata.columns:
        ev_lat_col, ev_lon_col = 'lat', 'lng'
    else:
        raise KeyError(f"EV metadata {ev_metadata_csv} has no recognized "
                       f"lat/lon columns (got {list(ev_metadata.columns)}).")

    # Define coordinates constraints and filter
    min_lat, max_lat = preprocessed_dataset_metadata[dataset_name]['min_max_lat']
    min_long, max_long = preprocessed_dataset_metadata[dataset_name]['min_max_long']
    filtered_ev_df = ev_metadata.loc[
        ev_metadata[ev_lat_col].between(min_lat, max_lat) &
        ev_metadata[ev_lon_col].between(min_long, max_long)].copy()

    ev_coords = filtered_ev_df[[ev_lat_col, ev_lon_col]].to_numpy()
    ev_lats = ev_coords[:, 0]
    ev_lons = ev_coords[:, 1]

    # Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(
        ev_lons, ev_lats,
        c='C0',
        marker='s',
        s=30,
        edgecolor='black',
        linewidth=0.5,
        label='EV points')

    if dataset_name == 'losangeles':
        # Los Angeles: the raw traffic metadata are point-like PeMS stations; the
        # graph topology is reconstructed (backbone + interchanges) as in losangeles.py
        if traffic_raw_metadata_file is None:
            raise ValueError("losangeles requires `traffic_raw_metadata_file` "
                             "(the raw PeMS d07 metadata with Fwy/Dir/Abs_PM).")
        _plot_losangeles_graph(traffic_raw_metadata_file, interchange_threshold_m)
    else:
        # chicago / newyork: each row is a (multi-point) traffic segment
        traffic_metadata = pd.read_csv(traffic_metadata_file, sep=sep)
        print(f'Loaded traffic_metadata with shape: {traffic_metadata.shape[0]}!')

        # Homogenize the per-dataset layout into `points_col`
        traffic_metadata = _build_points_column(dataset_name, traffic_metadata, points_col)

        # Augment graph connections (if graph is disconnected)
        if augment_graph:
            filter_traffic_metadata = augment_graph_df_v3(traffic_metadata,
                                                          fill_pct=augment_factor)
        else:
            filter_traffic_metadata = traffic_metadata

        # Lines and traffic segments, id as label
        cmap = plt.get_cmap('tab20')
        for idx, row in filter_traffic_metadata.iterrows():
            seg_id = row['id']
            pts = row[points_col]
            seg_lats = [pt[0] for pt in pts]
            seg_lons = [pt[1] for pt in pts]
            plt.plot(
                seg_lons, seg_lats,
                marker='s',
                markersize=4,
                linestyle='-',
                label=f"Segment {seg_id}",
                color=cmap(idx % cmap.N)
            )

    # Show
    plt.xlabel('Longitudine')
    plt.ylabel('Latitudine')
    plt.title('EV points e segmenti di traffico insieme')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Datasets to render. Each iteration rebuilds `Parameters` with the given
    # dataset_name, so all the dataset-dependent paths (traffic_metadata_file,
    # ev_metadata_file, ...) are recomputed by Parameters.__init__.
    DATASETS = ["chicago", "newyork", "losangeles"]

    for _dataset in DATASETS:
        # A SimpleNamespace (not a dict): Parameters.__init__ reads `params.verbose`
        # before converting, so it expects a Namespace-like object.
        run_params = Parameters(SimpleNamespace(dataset_name=_dataset, verbose=False))

        # Newyork has more roads connected as a single link
        if run_params.dataset_name == 'newyork':
            sep = ';'
        else:
            sep = ','

        # Los Angeles builds the graph from the raw PeMS d07 metadata (Fwy/Dir/Abs_PM),
        # which is NOT `location_summary.csv` and lives under `.../<ds>/other/traffic/...`.
        traffic_raw_metadata_file = None
        if run_params.dataset_name == 'losangeles':
            traffic_raw_metadata_file = getattr(run_params, 'traffic_raw_metadata_file', None)
            if traffic_raw_metadata_file is None:
                raw_ds_dir = os.path.dirname(os.path.dirname(run_params.traffic_metadata_file))
                traffic_raw_metadata_file = os.path.join(
                    raw_ds_dir, 'other', 'traffic', 'Traffic stations',
                    'd07_text_meta_2023_12_22.txt')

        # Visualize raw graph
        visualize_real_original_graph(run_params.dataset_name,
                                      run_params.datasets_hyperparameters,
                                      run_params.traffic_metadata_file,
                                      run_params.ev_metadata_file,
                                      sep=sep,
                                      traffic_raw_metadata_file=traffic_raw_metadata_file,
                                      interchange_threshold_m=getattr(
                                          run_params, 'graph_distance_threshold', 100.0))

