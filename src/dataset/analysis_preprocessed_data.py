"""
Inspection & sanity-checking of the *processed* spatio-temporal datasets.

Each city (chicago / newyork / losangeles) is preprocessed into a graph whose
nodes carry a multi-channel time series:

    final_temporal_merged_data : Tensor [N_nodes, T, F]

    - the first  `traffic_features` channels are TRAFFIC signals
      (e.g. speed, length, travel_time, total_flow, ...)
    - the last   `ev_features`     channels are EV signals
      (e.g. AvailabilityRate, the quantity we ultimately forecast)

The problem is then modelled as a spatio-temporal graph on which we perform
forecasting. This module does NOT touch the raw EV/traffic files: it only reads
the artefacts produced by the preprocessing pipeline and reports, in a uniform
and readable style, everything worth knowing about them:

    * files overview   -> which artefacts exist and their shape/meaning
    * graph summary     -> nodes, edges (real vs virtual MST), degree, EV mapping
    * temporal summary  -> time span, sampling resolution, regularity of the grid
    * per-channel stats  -> range / mean / std / quantiles for every feature
    * distribution       -> text histogram + skewness per channel
    * integrity checks   -> NaN / Inf / sentinel(-1) / shape & index consistency
    * sliding windows    -> train / val / test split implied by the config

Everything is parametric through `AnalysisOptions`, so you can switch each block
(and the per-variable `verbose` explanations) on/off independently.
"""

import os
import sys
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

# Make `src` importable regardless of the current working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Parameters  # noqa: E402


# --------------------------------------------------------------------------- #
#  Presentation helpers  (single place -> consistent look across every block)  #
# --------------------------------------------------------------------------- #
BOX_W = 72
_BLOCKS = " ▁▂▃▄▅▆▇█"

# Short, human-readable meaning of each channel we may encounter.
CHANNEL_HELP = {
    'speed':          'traffic speed on the road segment',
    'length':         'road-segment length (static geometric attribute)',
    'travel_time':    'travel time across the road segment',
    'avg_speed':      'average traffic speed measured at the sensor',
    'total_flow':     'total vehicle flow (count) at the sensor',
    'avg_occupancy':  'average lane occupancy at the sensor',
    'AvailabilityRate': 'EV charging-station availability rate  ← forecasting target',
}


def _rule(char: str = "=", width: int = BOX_W) -> str:
    return char * width


def _title(text: str, char: str = "=") -> None:
    print(_rule(char))
    print(f"  {text}")
    print(_rule(char))


def _section(text: str) -> None:
    print()
    print(f"  {text}")
    print("  " + "-" * (BOX_W - 2))


def _kv(label: str, value, note: Optional[str] = None,
        verbose: bool = False, width: int = 26) -> None:
    """Aligned 'label : value' row, optionally followed by an explanation."""
    print(f"  {label:<{width}}: {value}")
    if verbose and note:
        print(f"  {'':<{width}}    ↳ {note}")


def _sparkhist(values: np.ndarray, bins: int = 32) -> str:
    """One-line unicode histogram of a 1-D array (empty string if no data)."""
    if values.size == 0:
        return ""
    hist, _ = np.histogram(values, bins=bins)
    top = hist.max()
    if top == 0:
        return ""
    return "".join(_BLOCKS[int(round(h / top * 8))] for h in hist)


def _fmt(x, nd: int = 3) -> str:
    """Compact number formatting that stays readable for tiny/huge values."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return str(x)
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e6):
        return f"{x:.{nd}e}"
    return f"{x:,.{nd}f}"


# --------------------------------------------------------------------------- #
#  Options                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisOptions:
    """Toggle every analysis block and its explanations independently."""
    verbose: bool = True          # print a short note explaining each quantity
    files_overview: bool = True   # list & describe every processed artefact
    graph_summary: bool = True    # nodes / edges / degree / EV mapping
    temporal_summary: bool = True  # time span / sampling / grid regularity
    channel_stats: bool = True    # per-channel numeric statistics
    distribution: bool = True     # quantiles + text histogram per channel
    integrity: bool = True        # NaN / Inf / sentinel / consistency checks
    windows: bool = True          # sliding-window train/val/test split
    plots: bool = False           # draw matplotlib figures (blocking)
    plot_channels: Optional[List[int]] = None  # channel idx to plot (None=all)
    sentinel: float = -1.0        # value used to mark missing observations
    hist_bins: int = 32           # resolution of the text histogram
    quantiles: List[float] = field(
        default_factory=lambda: [5, 25, 50, 75, 95])


# --------------------------------------------------------------------------- #
#  Loading                                                                     #
# --------------------------------------------------------------------------- #
def _processed_dir(dataset: str) -> str:
    return os.path.join(PROJECT_ROOT, 'data', 'processed', dataset)


def _load_processed(dataset: str) -> dict:
    """Load every artefact of a processed dataset into a single dict."""
    p = _processed_dir(dataset)
    load_pt = lambda name: torch.load(os.path.join(p, name), weights_only=False)
    return {
        'path': p,
        'added_edges_df': pd.read_csv(os.path.join(p, 'added_edges_df.csv')),
        'edges_df': pd.read_csv(os.path.join(p, 'edges_df.csv')),
        'nodes_df': pd.read_csv(os.path.join(p, 'nodes_df.csv')),
        'config': load_pt('dataset_config.pt'),
        'edge_index': load_pt('edge_index_traffic.pt'),
        'edge_weights': load_pt('edge_weights_traffic.pt'),
        'data': load_pt('final_temporal_merged_data.pt'),
        'map_ev': load_pt('map_ev_node_traffic_node.pt'),
        'map_real_ev': load_pt('map_real_ev_node_traffic_node.pt'),
        'merged_map': load_pt('merged_traffic_nodes_map.pt'),
        'time_column': load_pt('time_column.pt'),
    }


def _channel_layout(config: dict) -> tuple:
    """Return (channel_names, channel_groups) ordered as in the data tensor."""
    traffic_cols = list(config['traffic_columns_used_in_data'])
    ev_cols = list(config['ev_columns_used_in_data'])
    names = [str(c) for c in traffic_cols] + [str(c) for c in ev_cols]
    groups = ['traffic'] * len(traffic_cols) + ['EV'] * len(ev_cols)
    return names, groups


# --------------------------------------------------------------------------- #
#  Analysis blocks                                                             #
# --------------------------------------------------------------------------- #
def _block_files(store: dict, opt: AnalysisOptions) -> None:
    _section("Processed artefacts")
    v = opt.verbose
    cfg = store['config']
    _kv("nodes_df", f"{store['nodes_df'].shape}", verbose=v,
        note="graph nodes after spatial merging (node_id, lat, lon)")
    _kv("edges_df", f"{store['edges_df'].shape}", verbose=v,
        note="all graph edges (real road links + added MST links)")
    _kv("added_edges_df", f"{store['added_edges_df'].shape}", verbose=v,
        note="virtual edges added to make the graph connected")
    _kv("edge_index_traffic", f"{tuple(store['edge_index'].shape)}", verbose=v,
        note="COO connectivity [2, E] fed to the GNN")
    _kv("edge_weights_traffic", f"{tuple(store['edge_weights'].shape)}", verbose=v,
        note="per-edge weight (typically inverse distance)")
    _kv("final_temporal_merged_data", f"{tuple(store['data'].shape)}", verbose=v,
        note="node signals [N_nodes, T, F=traffic+EV] -> the model input")
    _kv("time_column", f"{tuple(store['time_column'].shape)}", verbose=v,
        note="timestamp of each of the T timesteps")
    _kv("map_ev_node_traffic_node", f"{len(store['map_ev'])} entries", verbose=v,
        note="EV station id -> merged traffic node id")
    _kv("merged_traffic_nodes_map", f"{len(store['merged_map'])} entries", verbose=v,
        note="raw segment endpoint -> merged node id")
    _kv("dataset_config", f"{len(cfg)} keys", verbose=v,
        note="metadata (features, columns, time range, resolutions)")


def _block_graph(store: dict, opt: AnalysisOptions) -> None:
    _section("Graph summary")
    v = opt.verbose
    nodes_df, edges_df = store['nodes_df'], store['edges_df']
    added = store['added_edges_df']
    edge_index = store['edge_index']

    n_nodes = len(nodes_df)
    raw_endpoints = len(store['merged_map'])
    real_edges = len(edges_df) - len(added)
    virtual_edges = len(added)
    ev_stations = len(store['map_ev'])

    # Degree from the actual edge_index used by the model.
    ei = edge_index.reshape(2, -1)
    deg = torch.bincount(ei.reshape(-1).long(), minlength=n_nodes)
    isolated = int((deg == 0).sum())
    avg_deg = float(deg.float().mean())

    _kv("Raw segment endpoints", f"{raw_endpoints:,}", verbose=v,
        note="original coordinate ids before ~100 m spatial merging")
    _kv("Merged nodes (N)", f"{n_nodes:,}", verbose=v,
        note="graph nodes after merging = rows of the data tensor")
    _kv("Real edges", f"{real_edges:,}", verbose=v,
        note="edges coming from the real road topology")
    _kv("Virtual edges (MST)", f"{virtual_edges:,}", verbose=v,
        note="minimum-spanning-tree edges added for connectivity")
    _kv("Edges in edge_index", f"{ei.shape[1]:,}", verbose=v,
        note="directed entries actually fed to the GNN")
    _kv("Average node degree", _fmt(avg_deg, 2), verbose=v,
        note="mean number of incident edges per node")
    _kv("Isolated nodes", f"{isolated:,}", verbose=v,
        note="nodes with no edge (should normally be 0)")
    _kv("EV stations mapped", f"{ev_stations:,}", verbose=v,
        note="charging stations attached to the graph")


def _block_temporal(store: dict, opt: AnalysisOptions) -> None:
    _section("Temporal summary")
    v = opt.verbose
    cfg = store['config']
    tc = store['time_column']
    ts = pd.to_datetime(pd.Series(np.asarray(tc)).reset_index(drop=True))

    T = len(ts)
    start, end = ts.iloc[0], ts.iloc[-1]
    span = end - start

    diffs = ts.diff().dropna()
    dt_med = diffs.median().total_seconds() / 60.0
    dt_min = diffs.min().total_seconds() / 60.0
    dt_max = diffs.max().total_seconds() / 60.0
    monotonic = bool((diffs.dt.total_seconds() > 0).all())
    duplicates = int((diffs.dt.total_seconds() == 0).sum())
    # A "gap" is a step noticeably longer than the typical sampling interval.
    gaps = int((diffs.dt.total_seconds() > 1.5 * diffs.median().total_seconds()).sum())

    _kv("Timesteps (T)", f"{T:,}", verbose=v,
        note="length of every node time series")
    _kv("Time span", f"{start.date()} → {end.date()}  ({span.days} days)",
        verbose=v, note="first and last observed timestamp")
    _kv("Sampling (median)", f"{_fmt(dt_med, 1)} min", verbose=v,
        note="typical spacing between consecutive timesteps")
    _kv("Sampling (min / max)", f"{_fmt(dt_min, 1)} / {_fmt(dt_max, 1)} min",
        verbose=v, note="tightest and widest gap on the time grid")
    _kv("Monotonic increasing", monotonic, verbose=v,
        note="timestamps strictly ordered in time")
    _kv("Duplicate timestamps", f"{duplicates:,}", verbose=v,
        note="repeated instants (should be 0)")
    _kv("Irregular gaps", f"{gaps:,}", verbose=v,
        note="steps > 1.5x the median interval (possible missing periods)")
    _kv("Config resolutions", f"traffic={_fmt(cfg.get('traffic_resolution'), 2)} min"
        f"  |  ev={_fmt(cfg.get('ev_resolution'), 2)} min", verbose=v,
        note="mean native resolution declared during preprocessing")


def _channel_stats(x: torch.Tensor, sentinel: float) -> dict:
    """Numeric summary of a single channel tensor `x` of shape [N, T]."""
    x = x.detach().to(torch.float64).cpu()
    flat = x.reshape(-1)
    total = flat.numel()

    nan_mask = torch.isnan(flat)
    inf_mask = torch.isinf(flat)
    n_nan, n_inf = int(nan_mask.sum()), int(inf_mask.sum())

    finite = flat[~(nan_mask | inf_mask)]
    sent_mask = finite == sentinel
    n_sent = int(sent_mask.sum())
    valid = finite[~sent_mask].numpy()

    # Nodes whose value never changes over time = static / dead sensors.
    node_min = x.min(dim=1).values
    node_max = x.max(dim=1).values
    n_const = int((node_min == node_max).sum())

    stats = {
        'total': total, 'n_nan': n_nan, 'n_inf': n_inf,
        'n_sent': n_sent, 'n_valid': valid.size,
        'n_zero': int((valid == 0).sum()),
        'n_const_nodes': n_const, 'n_nodes': x.shape[0],
    }
    if valid.size:
        mean, std = float(valid.mean()), float(valid.std())
        skew = float((((valid - mean) / std) ** 3).mean()) if std > 0 else 0.0
        stats.update({
            'min': float(valid.min()), 'max': float(valid.max()),
            'mean': mean, 'std': std, 'skew': skew, 'valid': valid,
        })
    return stats


def _block_channels(store: dict, opt: AnalysisOptions) -> None:
    _section("Per-channel statistics")
    v = opt.verbose
    data = store['data']
    names, groups = _channel_layout(store['config'])
    F = data.shape[-1]

    if v:
        print("      ↳ statistics are computed on VALID values only")
        print(f"      ↳ (excluding NaN/Inf and the sentinel {opt.sentinel} used for 'missing')")

    for c in range(F):
        name = names[c] if c < len(names) else f"channel_{c}"
        group = groups[c] if c < len(groups) else "?"
        s = _channel_stats(data[:, :, c], opt.sentinel)

        print()
        print(f"  [{c}] {name:<16} ({group})")
        if v and name in CHANNEL_HELP:
            print(f"      ↳ {CHANNEL_HELP[name]}")

        if s['n_valid'] == 0:
            print("      (no valid values)")
            continue

        pct = lambda n: f"{100.0 * n / s['total']:.2f}%"
        print(f"      range     : {_fmt(s['min'])} → {_fmt(s['max'])}"
              f"     mean {_fmt(s['mean'])}   std {_fmt(s['std'])}   skew {_fmt(s['skew'], 2)}")
        if opt.distribution:
            qs = np.percentile(s['valid'], opt.quantiles)
            qtxt = "  ".join(f"p{int(q)} {_fmt(val)}"
                             for q, val in zip(opt.quantiles, qs))
            print(f"      quantiles : {qtxt}")
            print(f"      hist      : {_sparkhist(s['valid'], opt.hist_bins)}")
        print(f"      missing   : {s['n_sent']:,} ({pct(s['n_sent'])}) sentinel"
              f"   zeros {s['n_zero']:,} ({pct(s['n_zero'])})"
              f"   nan {s['n_nan']:,}   inf {s['n_inf']:,}")
        print(f"      static    : {s['n_const_nodes']:,}/{s['n_nodes']:,} nodes "
              f"constant over time")


def _block_integrity(store: dict, opt: AnalysisOptions) -> None:
    _section("Integrity & consistency checks")
    v = opt.verbose
    data = store['data']
    cfg = store['config']
    names, _ = _channel_layout(store['config'])

    N, T, F = data.shape
    n_nodes = len(store['nodes_df'])
    T_time = len(store['time_column'])
    exp_F = int(cfg['traffic_features']) + int(cfg['ev_features'])

    def flag(ok: bool) -> str:
        return "OK" if ok else "!! MISMATCH"

    _kv("N == rows(nodes_df)", f"{N} vs {n_nodes}   [{flag(N == n_nodes)}]",
        verbose=v, note="tensor node dimension must match the node table")
    _kv("T == len(time_column)", f"{T} vs {T_time}   [{flag(T == T_time)}]",
        verbose=v, note="tensor time dimension must match the timestamps")
    _kv("F == traffic+ev", f"{F} vs {exp_F}   [{flag(F == exp_F)}]", verbose=v,
        note="channel count must match the config feature counts")
    _kv("F == len(channel names)", f"{F} vs {len(names)}   [{flag(F == len(names))}]",
        verbose=v, note="every channel must have a declared name")

    # Global NaN / Inf / sentinel across the whole tensor.
    flat = data.reshape(-1)
    n_nan = int(torch.isnan(flat).sum())
    n_inf = int(torch.isinf(flat).sum())
    n_sent = int((flat == opt.sentinel).sum())
    tot = flat.numel()
    _kv("Total NaN", f"{n_nan:,}", verbose=v, note="not-a-number entries in the data")
    _kv("Total Inf", f"{n_inf:,}", verbose=v, note="infinite entries in the data")
    _kv("Total sentinel", f"{n_sent:,} ({100.0 * n_sent / tot:.2f}%)", verbose=v,
        note=f"entries equal to the missing marker {opt.sentinel}")

    # Edge-index / edge-weight sanity.
    ei = store['edge_index'].reshape(2, -1).long()
    ew = store['edge_weights'].reshape(-1)
    idx_ok = bool((ei.min() >= 0) and (ei.max() < N))
    n_selfloops = int((ei[0] == ei[1]).sum())
    _kv("edge_index in [0, N)", f"[{int(ei.min())}, {int(ei.max())}]   [{flag(idx_ok)}]",
        verbose=v, note="all edge endpoints must reference existing nodes")
    _kv("Self-loops", f"{n_selfloops:,}", verbose=v,
        note="edges connecting a node to itself")
    _kv("edge_weight range", f"{_fmt(float(ew.min()))} → {_fmt(float(ew.max()))}",
        verbose=v, note="min/max of the edge weights")
    _kv("edge_weight neg/nan", f"{int((ew < 0).sum())} / {int(torch.isnan(ew).sum())}",
        verbose=v, note="negative or NaN weights (should be 0)")


def _block_windows(store: dict, opt: AnalysisOptions, params: Parameters) -> None:
    _section("Sliding-window split (from config)")
    v = opt.verbose
    T = len(store['time_column'])
    lags = params.lags
    pred_win = params.prediction_window
    step = params.time_series_step

    total_windows = len(range(0, T - lags - pred_win, step))
    train_windows = int(params.train_ratio * total_windows)
    val_test = total_windows - train_windows
    val_windows = int(params.val_test_ratio * val_test)
    test_windows = val_test - val_windows

    _kv("Window geometry", f"lag={lags}  pred={pred_win}  step={step}", verbose=v,
        note="input length / forecast horizon / stride between windows")
    _kv("Total windows", f"{total_windows:,}", verbose=v,
        note="number of (input, target) samples extractable from T")
    _kv("Train windows", f"{train_windows:,}  ({params.train_ratio:.0%})", verbose=v,
        note="samples used for training")
    _kv("Val windows", f"{val_windows:,}  ({params.val_test_ratio:.0%} of remainder)",
        verbose=v, note="samples used for validation")
    _kv("Test windows", f"{test_windows:,}", verbose=v,
        note="samples held out for testing")


def _block_plots(store: dict, opt: AnalysisOptions) -> None:
    import matplotlib.pyplot as plt
    data = store['data']
    names, groups = _channel_layout(store['config'])
    F = data.shape[-1]
    channels = opt.plot_channels if opt.plot_channels is not None else range(F)
    site = random.randint(0, data.shape[0] - 1)

    for c in channels:
        name = names[c] if c < len(names) else f"channel_{c}"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2))
        ax1.plot(data[site, :, c].cpu())
        ax1.set_title(f"{name} — node {site} (time series)")
        ax1.set_xlabel("timestep")
        vals = data[:, :, c].reshape(-1).cpu().numpy()
        vals = vals[np.isfinite(vals) & (vals != opt.sentinel)]
        ax2.hist(vals, bins=opt.hist_bins)
        ax2.set_title(f"{name} — distribution")
        fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
#  Public entry point                                                         #
# --------------------------------------------------------------------------- #
def analyze_processed_data(dataset: str = 'chicago',
                           options: Optional[AnalysisOptions] = None,
                           params: Optional[Parameters] = None) -> None:
    """
    Detailed, parametric inspection of a processed spatio-temporal dataset.

    Parameters
    ----------
    dataset : one of 'chicago' / 'newyork' / 'losangeles'
    options : AnalysisOptions controlling which blocks are printed and whether
              per-variable explanations (`verbose`) are shown. Defaults to a
              full, verbose report.
    params  : Parameters instance used for the sliding-window split (created on
              demand if not provided).
    """
    opt = options or AnalysisOptions()
    params = params or Parameters()

    store = _load_processed(dataset)
    _title(f"Processed dataset report — {dataset.capitalize()}")
    print(f"  source : {store['path']}")

    if opt.files_overview:
        _block_files(store, opt)
    if opt.graph_summary:
        _block_graph(store, opt)
    if opt.temporal_summary:
        _block_temporal(store, opt)
    if opt.channel_stats:
        _block_channels(store, opt)
    if opt.integrity:
        _block_integrity(store, opt)
    if opt.windows:
        _block_windows(store, opt, params)
    if opt.plots:
        _block_plots(store, opt)

    print(_rule())
    print()


# --------------------------------------------------------------------------- #
#  Small standalone utilities (kept for quick manual checks)                   #
# --------------------------------------------------------------------------- #
def check_negative_values_in_dataset(dataset: str = 'chicago',
                                     sentinel: float = -1.0,
                                     plot: bool = False) -> None:
    """Quick look at the sentinel/negative values of a merged data tensor."""
    path = os.path.join(_processed_dir(dataset), 'final_temporal_merged_data.pt')
    db = torch.load(path, weights_only=False)
    n_sent = int((db <= sentinel).sum())

    _section(f"Negative/sentinel check — {dataset}")
    _kv("shape", tuple(db.shape))
    _kv(f"values <= {sentinel}", f"{n_sent:,} ({100.0 * n_sent / db.numel():.2f}%)")
    _kv("min / max", f"{_fmt(float(db.min()))} / {_fmt(float(db.max()))}")
    if plot:
        plot_site(db)


def plot_site(db: torch.Tensor, site_id: Optional[int] = None,
              channel: int = 0) -> None:
    """Plot the time series of one node/channel (random node if unspecified)."""
    import matplotlib.pyplot as plt
    if site_id is None:
        site_id = random.randint(0, db.shape[0] - 1)
    print(f"Plotting node {site_id}, channel {channel}")
    plt.plot(db[site_id, :, channel].cpu())
    plt.xlabel("timestep")
    plt.title(f"node {site_id} — channel {channel}")
    plt.show()


def check_inference_values(dataset: str = 'chicago', id: str = '005') -> None:
    """Scan inference CSV outputs for sentinel(-1) values and value range."""
    repo_csv = os.path.join(PROJECT_ROOT, 'registry', 'inference_outputs',
                            dataset, id, 'csv_out')
    _section(f"Inference outputs — {dataset}/{id}")
    print(f"  source : {repo_csv}")
    for elem in sorted(os.listdir(repo_csv)):
        if not elem.endswith('.csv'):
            continue
        pred = pd.read_csv(os.path.join(repo_csv, elem))
        pred_np = np.array(pred)[:, 1:]
        pred_pt = torch.tensor(np.array(pred_np, dtype=float)[:, 1:])
        n_neg = int((pred_pt == -1.).sum())
        _kv(elem, f"neg={n_neg:,}  min={_fmt(float(pred_np.min()))}  "
                  f"max={_fmt(float(pred_np.max()))}")


# --------------------------------------------------------------------------- #
#  Script entry                                                               #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    DATASETS = ["chicago", "newyork", "losangeles"]

    # Tune here what to analyse/print for every dataset.
    OPTIONS = AnalysisOptions(
        verbose=True,
        files_overview=True,
        graph_summary=True,
        temporal_summary=True,
        channel_stats=True,
        distribution=True,
        integrity=True,
        windows=True,
        plots=False,
    )

    params = Parameters()
    for _dataset in DATASETS:
        print()
        print("#" * BOX_W)
        print(f"#  Analysis — {_dataset}")
        print("#" * BOX_W)
        analyze_processed_data(_dataset, options=OPTIONS, params=params)