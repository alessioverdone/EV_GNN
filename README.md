# GNN for Electric Vehicles
**Graph Neural Networks for Smart Electric Vehicle Optimization**

---

## Overview

This repository develops **Graph Neural Networks (GNNs)** to model and forecast EV charging station availability in urban road networks. The core idea is to jointly represent **traffic conditions** and **EV charger availability** as signals on a spatial graph derived from the road network, and then train temporal GNN models to predict future EV availability.

The pipeline covers data collection (web scraping), preprocessing (temporal alignment, graph construction, feature engineering), model training, and inference with result export.

The primary dataset used is **Chicago**, which is the most complete and stable dataset in this project. New York City data has also been processed but is not fully up-to-date. Denmark data was explored but presents significant issues (sparse EV station coverage and large amounts of missing data) that prevent reaching satisfactory results.

---

## Repository Structure

```
EV_GNN/
├── data/
│   ├── raw/                          # Raw, unprocessed source data
│   │   ├── chicago/
│   │   │   ├── traffic/
│   │   │   │   ├── traffic/          # Per-segment traffic CSV files
│   │   │   │   └── location_summary.csv  # Traffic segment metadata
│   │   │   └── ev/
│   │   │       ├── ev_locations_availability/  # Per-station EV CSV files
│   │   │       └── ev_location_metadata.csv    # EV station metadata
│   │   ├── newyork/
│   │   └── denmark/
│   └── processed/                    # Preprocessed, model-ready data
│       ├── chicago/
│       ├── newyork/
│       └── denmark/
├── registry/
│   ├── checkpoints/                  # Saved model checkpoints
│   ├── configurations/               # Run configuration JSON files
│   ├── logs/                         # Training and evaluation logs
│   └── inference_outputs/            # Inference results per dataset/run
├── src/
│   ├── config.py                     # Global run parameters
│   ├── dataset/
│   │   ├── dataset.py                # Dataset classes (Denmark, NewYork, Chicago)
│   │   ├── utils.py                  # Graph utilities (NodeIndexer, clean_tensor, augment_graph_df_v3, ...)
│   │   ├── resamplling.py            # Temporal resampling (resample_to_common_time)
│   │   ├── preprocessing_metadata.py # Traffic metadata processing
│   │   ├── build_graph.py            # Graph visualization utilities
│   │   ├── visualize_data.py         # Map and graph visualization
│   │   └── data_analysis.py          # Exploratory data analysis helpers
│   ├── model/
│   │   ├── tf_model.py               # Main model wrapper (Lightning)
│   │   ├── dcrnn.py                  # DCRNN model
│   │   ├── gcn1d.py                  # GCN with 1D convolution
│   │   ├── gconvrnn.py               # Graph convolutional RNN
│   │   └── mini_rnn.py               # Lightweight RNN variants
│   ├── optimization/
│   │   ├── chicago/v1, v2/           # EV routing optimization (Chicago)
│   │   ├── new_york/                 # EV routing optimization (NYC)
│   │   └── denmark/                  # EV routing optimization (Denmark)
│   ├── scripts/
│   │   ├── run_train.py              # Training entry point
│   │   ├── run_inference.py          # Inference entry point
│   │   └── run_grid_search.py        # Hyperparameter grid search
│   ├── utils/
│   │   ├── utils.py                  # Shared utilities (graph conversion, metrics, ...)
│   │   └── inference_utils.py        # Inference-specific helpers
│   └── webscraping/
│       ├── ev_stations_avalibility/  # Scripts for scraping EV availability data
│       └── denmark_traffic/          # Scripts for scraping Denmark traffic data
├── env.yml                           # Conda environment specification
└── README.md
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/EV_GNN.git
   cd EV_GNN
   ```

2. Install dependencies:
   ```bash
   conda env create -f env.yml
   conda activate tf_env
   ```

3. Configure run parameters in `src/config.py`. Key parameters include:
   - `dataset_name`: one of `'chicago'`, `'newyork'`, `'denmark'`
   - `model`: GNN architecture (e.g., `'GraphWavenet'`, `'DCRNN'`, `'AGCRNModel'`)
   - `lags`, `prediction_window`: input/output time window sizes
   - `num_of_traffic_nodes_limit`, `num_of_ev_nodes_limit`: limit nodes for faster runs (`-1` = all)

---

## Model

The project supports multiple spatio-temporal GNN architectures, all wrapped in a unified PyTorch Lightning interface (`src/model/tf_model.py`):

| Model | Description |
|---|---|
| `GraphWavenet` | Adaptive graph + dilated WaveNet convolutions |
| `DCRNN` | Diffusion Convolutional RNN (from TSL) |
| `AGCRNModel` | Adaptive Graph Convolutional RNN |
| `gConvGRU` / `gConvLSTM` | Graph convolutional GRU/LSTM |
| `gcn1d` | GCN with 1D temporal convolution |
| `miniGRU` / `miniLSTM` | Lightweight RNN variants |

The model receives a graph snapshot `(x, edge_index, edge_attr)` at each step, where `x` has shape `(N_nodes, T_lags, F_features)` and predicts the next `prediction_window` steps.

---

## Registry — Inference Outputs

Trained models produce outputs saved under `registry/inference_outputs/<dataset>/<run_id>/`. Each run folder contains:
- Predicted vs. ground-truth time series (CSV or `.pt` tensors)
- Run configuration (`registry/configurations/config_<run_id>.json`)
- Model checkpoints (`registry/checkpoints/ckpt_<dataset>/`)
- Training logs (`registry/logs/log.txt`)

The run ID is controlled by `Parameters.id_run` in `src/config.py`.

---

## Data and Preprocessing

### Raw vs. Processed Data

All source data lives under `data/raw/<dataset>/` and must be preprocessed before use. The preprocessing is computationally expensive (graph construction, temporal alignment, EV-to-traffic projection), so the result is cached in `data/processed/<dataset>/` and loaded directly on subsequent runs. If the processed files are found, the dataset class skips all preprocessing steps entirely.

The Chicago dataset (`DatasetChicago` in `src/dataset/dataset.py`) is the reference implementation for the preprocessing pipeline. It is also the most complete and stable dataset in this project.

---

### The Chicago Dataset

#### Raw Data

| Source | Location | Description |
|---|---|---|
| Traffic segments | `data/raw/chicago/traffic/traffic/` | One CSV per road segment, columns: `speed`, `length`, `time` |
| Traffic metadata | `data/raw/chicago/traffic/location_summary.csv` | Segment IDs with start/end coordinates and street info |
| EV availability | `data/raw/chicago/ev/ev_locations_availability/` | One CSV per station, columns: `timestamp`, `Available`, `Total`, `Offline` |
| EV metadata | `data/raw/chicago/ev/ev_location_metadata.csv` | Station IDs, names, and GPS coordinates |

#### Preprocessing Pipeline

The preprocessing is driven by `DatasetChicago.__init__()` in `src/dataset/dataset.py` and proceeds in the following stages. If all processed files already exist under `data/processed/chicago/`, the constructor loads them directly and jumps to normalization and dataset assembly.

---

##### Stage 1 — Temporal Window Alignment (`check_traffic_ev_time`)

Traffic and EV time series are collected independently and often cover different periods with different lengths. The first step identifies the **maximum overlapping time window** between all traffic segments and all EV stations, setting `self.start_time` and `self.end_time`. All subsequent loading steps cut data to this window, ensuring that traffic and EV tensors cover exactly the same period.

---

##### Stage 2 — Loading Traffic Data (`_load_traffic_data`)

Each traffic CSV covers one road segment. Different segments may have slightly different numbers of rows even after windowing due to occasional missing recordings. Data is aligned **by row position** (`parsing_traffic_procedure = 'by_rows'`): all DataFrames are trimmed to the same number of rows using their tail (most recent observations), then stacked into a single tensor of shape `(N_edges, T, F_traffic)`.

This approach avoids the pitfalls of alignment by timestamp index, which is unreliable because absolute timestamps differ across segment files by small amounts that prevent a clean union index.

---

##### Stage 3 — Loading EV Data (`_load_ev_data`)

EV data is loaded similarly: stations outside the bounding box `[41.5–42.1 lat, -87.9 to -87.4 lon]` are discarded. Data is cut to the traffic temporal window, aligned by timestamp union, and padded with `-1.0` where a station has no measurement for a given timestamp. The result is a tensor of shape `(N_ev, T_ev, F_ev)`.

---

##### Stage 4 — Temporal Resampling (`resample_to_common_time`)

**Problem:** Traffic data and EV data are collected at different and sometimes irregular sampling frequencies. Traffic segments in Chicago are sampled approximately every 10 minutes, while EV availability data arrives at coarser or irregular intervals. This mismatch makes it impossible to directly concatenate the two tensors along the feature axis.

**Solution:** Both tensors are resampled to the same number of timesteps using `resample_to_common_time` (`src/dataset/resamplling.py`). By default, the traffic resolution is adopted as the target (`target="A"`) and linear interpolation is used along the time axis. The function supports four methods:
- `"linear"`: PyTorch 1D linear interpolation (uniform grid)
- `"nearest"`: nearest-neighbor
- `"previous"`: zero-order hold (step-wise, no new values introduced)
- `"spline"`: natural cubic spline via SciPy (most accurate but slowest)

The operation acts independently on each node and each feature channel, reshaping each tensor from `(N, T_src, F)` to `(N, T_target, F)`. After this step, both tensors share the same `T` dimension and can be processed jointly.

---

##### Stage 5 — Graph Construction (`get_edges_traffic`)

**Problem 1 — Near-duplicate intersection nodes:** The raw Chicago traffic data describes road *segments* (edges), not intersections (nodes). In the original dataset, a road crossing is typically represented by **4 distinct segment endpoints** — one per incoming road arm — that are geographically nearly identical but assigned different coordinates. Using these directly would produce a highly fragmented graph where a single physical intersection is split into 4 disconnected nodes.

**Solution:** `build_edges_with_node_ids_chicago` (`src/dataset/utils.py`) assigns a stable integer node ID to each segment endpoint using a `NodeIndexer` with a spatial merging threshold (`threshold=0.001` degrees, approximately 100 m). Endpoints within this radius are collapsed into the same node (the first-seen centroid is kept). The result is a compact `nodes_df` (columns: `node_id`, `lat`, `lon`) and an `edges_df` (columns: `id`, `src`, `tgt`, `distance`, `src_id`, `tgt_id`). A `merged_traffic_nodes_map` dictionary records the original-to-merged node ID mapping for traceability.

After merging, some pairs of edges may now share the same `(src_id, tgt_id)` pair (they were distinct in raw data but point to the same merged intersection). These duplicate edges are detected, their temporal data is **summed** before deletion, and the resulting deduplicated edge table and tensor are used from this point forward.

**Problem 2 — Disconnected graph components:** Even after merging, the road network extracted from the raw data may consist of several disconnected components. A disconnected graph prevents GNN message passing from propagating information between components, severely limiting the model's ability to capture spatial dependencies.

**Solution:** `augment_graph_df_v3` (`src/dataset/utils.py`) checks connectivity using a Union-Find (disjoint-set) structure. If there are multiple components, it computes the set of edges needed to connect them by running a Minimum Spanning Tree algorithm over all candidate node pairs sorted by Haversine distance, adding only the **minimum number of virtual edges** required to achieve full connectivity. Additional edges can optionally be added beyond the MST to reach a target edge density (`fill_pct`). The virtual edges are stored separately in `added_edges_df` for traceability.

The temporal features for the virtual edges are initialized using `append_along_N_torch` (`src/dataset/utils.py`) with `fill='mean'`: each virtual edge receives a feature tensor equal to the **mean over all real edges** at each timestep. This avoids introducing zero-valued features that would distort normalization, while not artificially inflating the signal.

The final graph is undirected. The edge list `edge_index` (PyG format) and `edge_weights` (Haversine distances in meters) are derived from the augmented adjacency matrix and converted to undirected form via `directed_to_undirected` (`src/utils/utils.py`).

---

##### Stage 6 — EV Features Projection onto Traffic Nodes (`assign_ev_node_to_traffic_node`)

**Problem:** EV charging stations and traffic measurement points are geographically distinct entities. The GNN operates on traffic nodes derived from the road graph, so EV availability signals must be transferred to the traffic graph's coordinate system.

**Solution — Nearest-neighbor assignment:** Each EV station is mapped to the nearest traffic node by minimum Haversine distance. This produces two mappings:
- `map_ev_node_traffic_node`: ordinal EV index (0-based, order in which stations were loaded) → traffic `node_id`
- `map_real_ev_node_traffic_node`: original EV `LocID` (from the metadata CSV) → traffic `node_id`

When multiple EV stations map to the same traffic node, their temporal feature vectors are **summed**. Traffic nodes with no nearby EV station receive a zero-valued EV feature tensor.

**Solution — Edge-level projection and aggregation:** Since raw traffic features are defined per edge (not per node), EV node features are first propagated to edges. For each edge `(u, v)`, the EV contribution is computed as:

```
ev_edge[e] = ev_node[u] / deg[u]  +  ev_node[v] / deg[v]
```

where `deg[x]` is the degree of node `x`. This degree normalization prevents high-degree nodes from disproportionately dominating edge features.

The traffic features and EV edge features are then concatenated along the feature axis (`dim=-1`), producing a combined tensor of shape `(N_edges, T, F_traffic + F_ev)`. Finally, this is collapsed from edge-level to node-level via `edge_to_node_aggregation` (`src/utils/utils.py`), which averages the features of all incident edges for each node:

```python
node_features = (scatter(edge_attr, edge_index[0], reduce='mean') +
                 scatter(edge_attr, edge_index[1], reduce='mean')) / 2
```

The final result is `final_temporal_merged_data`, shape `(N_nodes, T, F_traffic + F_ev)`.

---

##### Stage 7 — Missing Value Cleaning (`clean_tensor`)

**Problem:** EV stations frequently report negative measurements. These arise when a station is offline, has temporarily stopped transmitting, or when a measurement falls between `-0.5` and `0` due to rounding. The padding value used throughout loading is also `-1.0`. These invalid values would corrupt normalization and distort model training.

**Solution:** `clean_tensor` (`src/dataset/utils.py`) operates on a tensor of shape `(N, T, F)`:
- All negative values are replaced with `NaN`.
- Each `NaN` position is then filled with the **mean of valid (non-negative) values** for that specific `(node, feature)` pair, computed via `torch.nanmean` along the time axis (shape: `(N, 1, F)`).

```python
# src/dataset/utils.py — clean_tensor
mask = tensor < 0
tensor[mask] = float('nan')
means = torch.nanmean(tensor, dim=1, keepdim=True)  # (N, 1, F)
nan_mask = torch.isnan(tensor)
tensor[nan_mask] = means.expand_as(tensor)[nan_mask]
```

This ensures a clean, non-negative signal without discarding entire time windows.

---

##### Stage 8 — Normalization and Dataset Assembly (`preprocess_and_assemble_data`)

The cleaned tensor is normalized **channel-wise** using Min-Max scaling. The minimum and maximum values are computed over all nodes and timesteps for each feature channel independently, and stored in `dataset_config.pt` for consistent denormalization during inference.

The normalized data is then sliced into fixed-size sliding windows of size `(lags, prediction_window)` with stride `time_series_step`, producing a list of `torch_geometric.data.Data` objects. Each object contains:
- `x`: input window, shape `(N, lags, F)`
- `y`: target window, shape `(N, prediction_window * F)` (flattened)
- `edge_index`, `edge_attr`: graph structure
- `time_input`, `time_output`: corresponding timestamps

---

### Processed Data Files (`data/processed/chicago/`)

After preprocessing, the following files are saved and used for all subsequent runs:

| File | Type | Description |
|---|---|---|
| `final_temporal_merged_data.pt` | `torch.Tensor (N, T, F)` | Cleaned, merged traffic+EV feature tensor on traffic nodes |
| `time_column.pt` | `pd.Series` | Timestamp series for the traffic time window |
| `edge_index_traffic.pt` | `torch.Tensor (2, E)` | Undirected edge list (PyG format) |
| `edge_weights_traffic.pt` | `torch.Tensor (E,)` | Edge weights in meters |
| `nodes_df.csv` | DataFrame | Node table: `node_id`, `lat`, `lon` — the merged intersection nodes |
| `edges_df.csv` | DataFrame | Full edge table (real + virtual): `id`, `src`, `tgt`, `distance`, `src_id`, `tgt_id` |
| `added_edges_df.csv` | DataFrame | Virtual edges added to ensure graph connectivity |
| `map_ev_node_traffic_node.pt` | `dict` | Mapping: ordinal EV index → traffic node ID |
| `map_real_ev_node_traffic_node.pt` | `dict` | Mapping: original EV `LocID` → traffic node ID |
| `merged_traffic_nodes_map.pt` | `dict` | Mapping: original node coordinate ID → merged node ID |
| `dataset_config.pt` | `dict` | Metadata: feature counts, normalization params, time bounds, sampling resolutions |
| `processed_graph.html` | HTML | Interactive Folium map of the final graph (real edges + highlighted virtual edges) |

These files contain everything needed to work with the dataset without re-running preprocessing:
- **Graph structure** (`nodes_df.csv`, `edges_df.csv`, `edge_index_traffic.pt`) to reconstruct or visualize the road graph.
- **Temporal features** (`final_temporal_merged_data.pt`, `time_column.pt`) for training, analysis, or visualization.
- **ID traceability** (`map_ev_node_traffic_node.pt`, `map_real_ev_node_traffic_node.pt`, `merged_traffic_nodes_map.pt`) to trace any node back to its original EV station `LocID` or raw traffic segment endpoint.
- **Virtual edge bookkeeping** (`added_edges_df.csv`) to distinguish real road edges from synthetic connectivity edges — relevant if edge-level predictions or per-edge analyses are needed downstream.
- **Normalization and run config** (`dataset_config.pt`) for consistent inference without re-running preprocessing.

---

### Other Datasets

| Dataset | Status | Notes |
|---|---|---|
| **Chicago** | Primary, fully functional | Reference implementation for all preprocessing stages |
| **New York City** | Usable, not fully up-to-date | Preprocessing pipeline implemented in `DatasetNewyork`; some steps may need a refresh for the latest data |
| **Denmark** | Exploratory, not production-ready | EV stations are too sparse and data coverage is insufficient for meaningful graph-level modelling |

---

## Running the Code

### Training
```bash
python src/scripts/run_train.py --dataset_name chicago --model GraphWavenet --lags 24 --prediction_window 24
```

### Inference
```bash
python src/scripts/run_inference.py --dataset_name chicago --checkpoint <path_to_ckpt>
```

### Grid Search
```bash
python src/scripts/run_grid_search.py
```
