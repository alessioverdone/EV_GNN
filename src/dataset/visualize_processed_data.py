import ast
from pathlib import Path
from typing import Optional
import folium
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


# Project root: <...>/EV_GNN (parents[2] of this file), i.e. the folder holding data/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_data_dir(dataset_name: str) -> Path:
    """Default processed-data folder for a dataset: data/processed/<dataset_name>."""
    return _PROJECT_ROOT / "data" / "processed" / dataset_name


def _parse_coord(val):
    """Parse a "(lat, lon)" cell into a (lat, lon) float tuple.

    Returns None for missing/empty/unparseable values (e.g. losangeles original
    edges store empty src/tgt strings and rely on src_id/tgt_id instead).
    """
    if val is None:
        return None
    if isinstance(val, (tuple, list)):
        return float(val[0]), float(val[1])
    if isinstance(val, float) and pd.isna(val):
        return None
    try:
        parsed = ast.literal_eval(val)
        return float(parsed[0]), float(parsed[1])
    except (ValueError, SyntaxError, TypeError):
        return None


def _node_coord_map(nodes_df: pd.DataFrame) -> dict:
    """node_id -> (lat, lon) lookup, shared by all datasets."""
    return {int(r["node_id"]): (float(r["lat"]), float(r["lon"]))
            for _, r in nodes_df.iterrows()}


def _attach_edge_coords(df: pd.DataFrame, node_xy: dict) -> pd.DataFrame:
    """Resolve each edge's endpoints as (lat, lon) tuples into `src`/`tgt`.

    Coordinates are taken from `src_id`/`tgt_id` looked up in `node_xy` (works for
    all datasets, including losangeles where the string `src`/`tgt` columns are
    empty for the original edges). If the id is missing or not mapped, we fall back
    to parsing the string `src`/`tgt` tuple; if that also fails the endpoint is None
    and the edge is skipped when plotting.
    """
    df = df.copy()

    def resolve(row, id_col, str_col):
        if id_col in row and pd.notna(row.get(id_col)):
            coord = node_xy.get(int(row[id_col]))
            if coord is not None:
                return coord
        return _parse_coord(row.get(str_col))

    df["src"] = df.apply(lambda r: resolve(r, "src_id", "src"), axis=1)
    df["tgt"] = df.apply(lambda r: resolve(r, "tgt_id", "tgt"), axis=1)
    return df


def _load_data(data_dir: Path):
    edges_df = pd.read_csv(data_dir / "edges_df.csv")
    nodes_df = pd.read_csv(data_dir / "nodes_df.csv")

    added_path = data_dir / "added_edges_df.csv"
    added_edges_df = pd.read_csv(added_path) if added_path.exists() else pd.DataFrame()

    # Endpoint coordinates are resolved from node ids (robust across datasets).
    node_xy = _node_coord_map(nodes_df)
    edges_df = _attach_edge_coords(edges_df, node_xy)
    if not added_edges_df.empty:
        added_edges_df = _attach_edge_coords(added_edges_df, node_xy)

    return edges_df, added_edges_df, nodes_df


def visualize_processed_graph(dataset_name: str = "chicago",
                              data_dir: Optional[str] = None,
                              output_dir: Optional[str] = None,
                              output_prefix: Optional[str] = None,
                              # --- Nodi ---
                              show_nodes: bool = True,
                              node_color: str = "#2ca02c",
                              node_size: int = 15,
                              node_marker: str = "o",
                              # --- Archi originali ---
                              edge_color: str = "#1f77b4",
                              edge_width: float = 0.8,
                              edge_style: str = "solid",
                              edge_alpha: float = 0.6,
                              # --- Archi aggiunti ---
                              show_added_edges: bool = True,
                              added_edge_color: str = "#d62728",
                              added_edge_width: float = 1.5,
                              added_edge_style: str = "dashed",
                              added_edge_alpha: float = 0.9,
                              # --- Sfondo (solo matplotlib) ---
                              satellite: bool = False,
                              satellite_zoom: int = 12,
                              # --- HTML (folium) ---
                              html_edge_weight: int = 3,
                              html_added_edge_weight: int = 2,
                              html_edge_opacity: float = 0.7,
                              html_zoom: int = 11,
                              html_satellite: bool = True,
                              # --- Output ---
                              save_png: bool = True,
                              save_pdf: bool = True,
                              save_html: bool = True,
                              show_plot: bool = False,
                              figsize: tuple = (14, 14),
                              dpi: int = 200) -> dict:
    """
    View the processed graph of a dataset and save .png, .pdf, and .html files.

    Generalist over the three datasets (chicago, newyork, losangeles): it reads
    only the *processed* artifacts in ``data/processed/<dataset_name>``
    (``edges_df.csv``, ``nodes_df.csv``, ``added_edges_df.csv``). Edge endpoints are
    resolved from ``src_id``/``tgt_id`` via ``nodes_df``, so it works uniformly even
    for losangeles, whose original edges store empty ``src``/``tgt`` coordinates.

    Parameters
    ---------
    dataset_name : str
        One of {"chicago", "newyork", "losangeles"}. Selects the default
        ``data/processed/<dataset_name>`` folder and the output prefix/title.
    data_dir : str, optional
        Folder containing edges_df.csv, nodes_df.csv, and added_edges_df.csv.
        Default: data/processed/<dataset_name> relative to the project root.
    output_dir : str, optional
        Folder to save the files in. Default: data_dir/outputs.
    output_prefix : str, optional
        File name prefix. Default: "<dataset_name>_graph".

    Nodes
    ----
    show_nodes : bool – show/hide nodes
    node_color : str – hexadecimal color ("#2ca02c")
    node_size : int – matplotlib scatter size
    node_marker : str – matplotlib marker ('o', 's', '^', ...)

    Original edges
    ---------------
    edge_color : str – solid color of the original edges
    edge_width : float – line thickness
    edge_style : str – 'solid' | 'dashed' | 'dotted' | 'dashdot'
    edge_alpha : float – opacity [0, 1]

    Added edges (added_edges_df.csv)
    ------------------------------------
    show_added_edges : bool – show/hide added edges
    added_edge_color : str – color
    added_edge_width : float – thickness
    added_edge_style : str – line style
    added_edge_alpha : float – opacity

    Background (matplotlib)
    -------------------
    satellite : bool – use Esri satellite background via contextily
    satellite_zoom : int – zoom level requested by contextily

    HTML (folium)
    -------------
    html_edge_weight : int – the weight of the original edges in the map
    html_added_edge_weight : int – the weight of the added edges
    html_edge_opacity : float – the opacity of the edges in the map
    html_zoom : int – the initial zoom
    html_satellite : bool – the Esri background in the HTML map

    Output
    ------
    save_png   : bool – save .png
    save_pdf   : bool – save .pdf
    save_html  : bool – save .html
    show_plot  : bool – call plt.show()
    figsize    : tuple
    dpi        : int

    Returns
    -----------
    dict with keys 'png', 'pdf', 'html' → Path of saved files (or None).
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir(dataset_name)
    output_dir = Path(output_dir) if output_dir else (data_dir / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_prefix or f"{dataset_name}_graph"

    print(f"[visualize_processed_graph] Caricamento dati da '{data_dir}'")
    edges_df, added_edges_df, nodes_df = _load_data(data_dir)

    added_ids = set(added_edges_df["id"].tolist()) if not added_edges_df.empty else set()
    if show_added_edges and added_ids:
        orig_edges = edges_df[~edges_df["id"].isin(added_ids)].copy()
        add_edges = edges_df[edges_df["id"].isin(added_ids)].copy()
    else:
        orig_edges = edges_df.copy()
        add_edges = pd.DataFrame()

    # Map center: average over the resolved (non-null) endpoint coordinates.
    coords = [c for c in edges_df["src"].tolist() + edges_df["tgt"].tolist() if c is not None]
    if not coords:
        raise ValueError(f"[visualize_processed_graph] No edge coordinates could be "
                         f"resolved for '{dataset_name}' in {data_dir}.")
    center_lat = sum(c[0] for c in coords) / len(coords)
    center_lon = sum(c[1] for c in coords) / len(coords)

    saved = {"png": None, "pdf": None, "html": None}

    # =========================================================================
    # MATPLOTLIB (PNG + PDF)
    # =========================================================================
    if save_png or save_pdf or show_plot:
        fig, ax = plt.subplots(figsize=figsize)

        for _, row in orig_edges.iterrows():
            if row["src"] is None or row["tgt"] is None:
                continue
            (lat0, lon0), (lat1, lon1) = row["src"], row["tgt"]
            ax.plot(
                [lon0, lon1], [lat0, lat1],
                color=edge_color,
                linewidth=edge_width,
                linestyle=edge_style,
                alpha=edge_alpha,
            )

        if not add_edges.empty:
            first = True
            for _, row in add_edges.iterrows():
                if row["src"] is None or row["tgt"] is None:
                    continue
                (lat0, lon0), (lat1, lon1) = row["src"], row["tgt"]
                ax.plot(
                    [lon0, lon1], [lat0, lat1],
                    color=added_edge_color,
                    linewidth=added_edge_width,
                    linestyle=added_edge_style,
                    alpha=added_edge_alpha,
                    label="Additional edges" if first else None,
                )
                first = False

        if show_nodes and not nodes_df.empty:
            ax.scatter(
                nodes_df["lon"].tolist(),
                nodes_df["lat"].tolist(),
                c=node_color,
                s=node_size,
                marker=node_marker,
                zorder=5,
                edgecolors="black",
                linewidths=0.3,
            )

        if satellite:
            try:
                import contextily as ctx
                ctx.add_basemap(
                    ax,
                    crs="EPSG:4326",
                    source=ctx.providers.Esri.WorldImagery,
                    zoom=satellite_zoom,
                    alpha=0.6,
                )
            except ImportError:
                print("[visualize_processed_graph] contextily non installato — sfondo satellitare ignorato.")

        legend_handles = [
            Line2D([0], [0], color=edge_color, linewidth=1.5,
                   linestyle=edge_style, label=f"Additional edges ({len(orig_edges)})"),
        ]
        if not add_edges.empty:
            legend_handles.append(
                Line2D([0], [0], color=added_edge_color, linewidth=1.5,
                       linestyle=added_edge_style,
                       label=f"Additional edges ({len(add_edges)})")
            )
        if show_nodes and not nodes_df.empty:
            legend_handles.append(
                Line2D([0], [0], marker=node_marker, color="w",
                       markerfacecolor=node_color, markersize=7,
                       markeredgecolor="black", markeredgewidth=0.3,
                       label=f"Nodes ({len(nodes_df)})")
            )

        ax.legend(handles=legend_handles, loc="best", fontsize=9)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Processed graph {dataset_name} — edges and nodes")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        if save_png:
            png_path = output_dir / f"{output_prefix}.png"
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
            saved["png"] = png_path
            print(f"[visualize_processed_graph] Salvato PNG → '{png_path}'")

        if save_pdf:
            pdf_path = output_dir / f"{output_prefix}.pdf"
            fig.savefig(pdf_path, bbox_inches="tight")
            saved["pdf"] = pdf_path
            print(f"[visualize_processed_graph] Salvato PDF → '{pdf_path}'")

        if show_plot:
            plt.show()

        plt.close(fig)

    # =========================================================================
    # FOLIUM (HTML)
    # =========================================================================
    if save_html:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=html_zoom, tiles=None)

        if html_satellite:
            folium.TileLayer(
                tiles=(
                    "https://server.arcgisonline.com/ArcGIS/rest/services/"
                    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ),
                attr=(
                    "Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, "
                    "Getmapping, Aerogrid, IGN, IGP, UPR-EGP, GIS User Community"
                ),
                name="Esri World Imagery",
                overlay=False,
                control=True,
            ).add_to(m)
        else:
            folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

        grp_orig = folium.FeatureGroup(name=f"Original edges ({len(orig_edges)})").add_to(m)
        for _, row in orig_edges.iterrows():
            if row["src"] is None or row["tgt"] is None:
                continue
            (lat0, lon0), (lat1, lon1) = row["src"], row["tgt"]
            folium.PolyLine(
                locations=[[lat0, lon0], [lat1, lon1]],
                color=edge_color,
                weight=html_edge_weight,
                opacity=html_edge_opacity,
                popup=f"edge id: {row['id']}",
            ).add_to(grp_orig)

        if not add_edges.empty:
            grp_added = folium.FeatureGroup(
                name=f"Additional edges ({len(add_edges)})"
            ).add_to(m)
            for _, row in add_edges.iterrows():
                if row["src"] is None or row["tgt"] is None:
                    continue
                (lat0, lon0), (lat1, lon1) = row["src"], row["tgt"]
                folium.PolyLine(
                    locations=[[lat0, lon0], [lat1, lon1]],
                    color=added_edge_color,
                    weight=html_added_edge_weight,
                    opacity=html_edge_opacity,
                    popup=f"added edge id: {row['id']}",
                    dash_array="6 4",
                ).add_to(grp_added)

        if show_nodes and not nodes_df.empty:
            grp_nodes = folium.FeatureGroup(name=f"Nodes ({len(nodes_df)})").add_to(m)
            for _, row in nodes_df.iterrows():
                folium.CircleMarker(
                    location=[float(row["lat"]), float(row["lon"])],
                    radius=4,
                    color=node_color,
                    fill=True,
                    fill_color=node_color,
                    fill_opacity=0.85,
                    popup=f"node id: {row['node_id']}",
                ).add_to(grp_nodes)

        folium.LayerControl().add_to(m)
        html_path = output_dir / f"{output_prefix}.html"
        m.save(html_path)
        saved["html"] = html_path
        print(f"[visualize_processed_graph] Salvato HTML → '{html_path}'")

    return saved


if __name__ == "__main__":
    """
    View the processed graph of a dataset from data/processed/<dataset_name>.

    Save the results in .png, .pdf, and .html with full styling options
    for nodes, original edges, added edges, and satellite background.

    Quick Use
    --------------
    python visualize_processed_data.py            # runs all three datasets

    Use from code
    ------------------
    from visualize_processed_data import visualize_processed_graph
    visualize_processed_graph("losangeles", satellite=True, show_added_edges=True)
    """
    # Datasets to render (each reads only data/processed/<dataset_name>).
    DATASETS = ["chicago", "newyork", "losangeles"]

    for _dataset in DATASETS:
        visualize_processed_graph(dataset_name=_dataset,
                                  show_nodes=True,
                                  node_color="#e377c2",
                                  node_size=12,
                                  node_marker="o",
                                  edge_color="#1f77b4",
                                  edge_width=0.7,
                                  edge_style="solid",
                                  edge_alpha=0.6,
                                  show_added_edges=True,
                                  added_edge_color="#d62728",
                                  added_edge_width=1.5,
                                  added_edge_style="dashed",
                                  added_edge_alpha=0.9,
                                  satellite=False,
                                  html_satellite=True,
                                  html_zoom=11,
                                  save_png=True,
                                  save_pdf=True,
                                  save_html=True,
                                  show_plot=False)
