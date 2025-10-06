import ast
import matplotlib.pyplot as plt
import pandas as pd

from src.dataset.utils import augment_graph_df

traffic_metadata_csv = r"/mnt/c/Users/Grid/Desktop/PhD/EV/code/EV_GNN/data/newyork/traffic/stations_meta_data.csv"
filter_traffic_csv = r"/mnt/c/Users/Grid/Desktop/PhD/EV/code/EV_GNN/data/newyork/traffic/filter_newyork_map.csv"
ev_metadata_csv = r"/mnt/c/Users/Grid/Desktop/PhD/EV/code/EV_GNN/data/newyork/ev/location_meta_data.csv"
augment_graph = True
augment_factor = 0.0001

traffic_metadata = pd.read_csv(traffic_metadata_csv)
filter_traffic_metadata = pd.read_csv(filter_traffic_csv)
ev_metadata = pd.read_csv(ev_metadata_csv)

print(f'Loaded traffic_metadata con righe {traffic_metadata.shape[0]}!')
print(f'Loaded filter_traffic_metadata con righe {filter_traffic_metadata.shape[0]}!')
print(f'Loaded ev_metadata con righe {ev_metadata.shape[0]}!')

# 1) Definisci i vincoli
min_lat, max_lat = 40.4, 40.95
min_long, max_long = -74.5, -73.5

# 2) Filtra il DataFrame in un colpo solo
filtered_ev_df = ev_metadata.loc[
    ev_metadata['Latitude'].between(min_lat, max_lat) &
    ev_metadata['Longitude'].between(min_long, max_long)].copy()

# 3) Estrai le coordinate come array numpy
ev_coords = filtered_ev_df[['Latitude', 'Longitude']].to_numpy()
ev_lats = ev_coords[:, 0]
ev_lons = ev_coords[:, 1]

# 2) Prepara i segmenti di traffico (colonna __points già in filter_traffic_metadata)
points_col = '__points'
filter_traffic_metadata[points_col] = filter_traffic_metadata[points_col].apply(ast.literal_eval)

# 2b) Mantieni solo il punto iniziale e quello finale
filter_traffic_metadata[points_col] = (
    filter_traffic_metadata[points_col]
    .apply(lambda pts: [pts[0], pts[-1]] if isinstance(pts, list) and len(pts) > 1 else pts)
)

# Augment graph connections
if augment_graph:
    filter_traffic_metadata = augment_graph_df(filter_traffic_metadata, fill_pct=augment_factor)

# 3) Crea la figura unica
plt.figure(figsize=(10, 10))

# 3a) Scatter EV
plt.scatter(
    ev_lons, ev_lats,
    c='C0',
    marker='s',
    s=30,
    edgecolor='black',
    linewidth=0.5,
    label='EV points'
)

# 3b) Linee segmenti di traffico, con id come etichetta
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

# 4) Rifiniture
plt.xlabel('Longitudine')
plt.ylabel('Latitudine')
plt.title('EV points e segmenti di traffico insieme')
plt.grid(True)
# plt.legend(loc='best', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()


# continua a creare tensore temporale prendendo in considerazione archi nuovi e agglomera ev in ev nodes