import os
import ast
import pandas as pd
import matplotlib.pyplot as plt


from src.dataset.utils import augment_graph_df_v3


def visualize_real_original_graph(dataset_name,
                                  preprocessed_dataset_metadata,
                                  traffic_metadata_csv,
                                  filter_traffic_csv,
                                  ev_metadata_csv,
                                  augment_graph = False,
                                  augment_factor = 0.0001,
                                  points_col = '__points'):


    # Load data
    traffic_metadata = pd.read_csv(traffic_metadata_csv)
    filter_traffic_metadata = pd.read_csv(filter_traffic_csv, sep=';')
    ev_metadata = pd.read_csv(ev_metadata_csv)
    print(f'Loaded traffic_metadata with shape: {traffic_metadata.shape[0]}!')
    print(f'Loaded filter_traffic_metadata with shape: {filter_traffic_metadata.shape[0]}!')
    print(f'Loaded ev_metadata with shape: {ev_metadata.shape[0]}!')

    # Define coordinates constraints and filter
    min_lat, max_lat = preprocessed_dataset_metadata[dataset_name]['min_max_lat']
    min_long, max_long = preprocessed_dataset_metadata[dataset_name]['min_max_long']
    filtered_ev_df = ev_metadata.loc[
        ev_metadata['Latitude'].between(min_lat, max_lat) &
        ev_metadata['Longitude'].between(min_long, max_long)].copy()

    ev_coords = filtered_ev_df[['Latitude', 'Longitude']].to_numpy()
    ev_lats = ev_coords[:, 0]
    ev_lons = ev_coords[:, 1]

    # Prepare traffic columns (__points yet in filter_traffic_metadata)
    filter_traffic_metadata[points_col] = filter_traffic_metadata[points_col].apply(ast.literal_eval)

    # Maintain only initial and final points of the list
    filter_traffic_metadata[points_col] = (
        filter_traffic_metadata[points_col]
        .apply(lambda pts: [pts[0], pts[-1]] if isinstance(pts, list) and len(pts) > 1 else pts))

    # Augment graph connections (if graph is disconnected
    if augment_graph:
        filter_traffic_metadata = augment_graph_df_v3(filter_traffic_metadata,
                                                      fill_pct=augment_factor)

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
    # Choose one in ['denmark', 'newyork', 'chicago']
    datapath = '../../data'
    dataset_name = 'newyork'
    preprocessed_dataset_metadata = {
        "newyork": {
            "min_max_lat" : (40.4, 40.95),
            "min_max_long": (-74.5, -73.5)
        },
        "chicago":{
            "min_max_lat" : (41.5, 42.1),
            "min_max_long": (-87.9, -87.4)
        }
    }

    if dataset_name == 'denmark':
        pass
    elif dataset_name == 'chicago':
        # Paths
        traffic_metadata_csv = os.path.join(datapath, dataset_name, 'traffic', 'stations_meta_data.csv')
        filter_traffic_csv = os.path.join(datapath, dataset_name, 'traffic', 'processed_newyork_traffic_graph.csv')
        ev_metadata_csv = os.path.join(datapath, dataset_name, 'ev', 'location_meta_data.csv')

        # Visualize original graph processed
        visualize_real_original_graph(dataset_name,
                                      preprocessed_dataset_metadata,
                                      traffic_metadata_csv,
                                      filter_traffic_csv,
                                      ev_metadata_csv)
    elif dataset_name == 'newyork':
        # Paths
        traffic_metadata_csv = os.path.join(datapath, dataset_name, 'traffic', 'stations_meta_data.csv')
        filter_traffic_csv = os.path.join(datapath, dataset_name, 'traffic', 'processed_newyork_traffic_graph.csv')
        ev_metadata_csv = os.path.join(datapath, dataset_name, 'ev', 'location_meta_data.csv')

        # Visualize original graph processed
        visualize_real_original_graph(dataset_name,
                                      preprocessed_dataset_metadata,
                                      traffic_metadata_csv,
                                      filter_traffic_csv,
                                      ev_metadata_csv)
    else:
        raise ValueError('Invalid dataset')
