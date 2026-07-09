from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1) Loading raw metadata (per unit)
# ---------------------------------------------------------------------------
def load_raw_metadata(meta_file: os.PathLike | str) -> pd.DataFrame:
    """Load ``stations_metadata.csv`` (one row per unit)."""
    meta_file = Path(meta_file)
    if not meta_file.exists():
        raise FileNotFoundError(f"Raw EV metadata not found: {meta_file}")
    df = pd.read_csv(meta_file, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    # useful numeric types
    for col in ("lat", "lng", "max_power_kw"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")
    print(f"[load_raw_metadata] {len(df)} units "
          f"({df['station_id'].nunique()} distinct stations).")
    return df


# ---------------------------------------------------------------------------
# 2) Metadata aggregation at the station_id level
# ---------------------------------------------------------------------------
def _first_non_empty(series: pd.Series) -> str:
    """First non-empty value of the series (string), otherwise ''."""
    for v in series:
        if isinstance(v, str) and v.strip():
            return v.strip()
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip():
            return str(v).strip()
    return ""


def _unique_joined(series: pd.Series) -> str:
    """Distinct non-empty values, sorted, joined by ' | '."""
    vals = sorted({str(v).strip() for v in series if str(v).strip()})
    return " | ".join(vals)


def aggregate_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    """Merge the metadata by ``station_id`` (one row per physical station).

    The coordinates are constant within the station; the heterogeneous
    attributes (connector types, power) are summarized.
    """
    rows = []
    for station_id, g in raw.groupby("station_id", sort=True):
        lat = pd.to_numeric(g["lat"], errors="coerce").dropna()
        lng = pd.to_numeric(g["lng"], errors="coerce").dropna()
        power = pd.to_numeric(g["max_power_kw"], errors="coerce").dropna()
        rows.append({
            "station_id": station_id,
            "title": _first_non_empty(g["title"]),
            "supplier_name": _first_non_empty(g["supplier_name"]),
            "lat": float(lat.iloc[0]) if len(lat) else np.nan,
            "lng": float(lng.iloc[0]) if len(lng) else np.nan,
            "num_units": int(len(g)),
            "connector_types": _unique_joined(g["connector_type_name"]),
            "max_power_kw": float(power.max()) if len(power) else np.nan,
            "access": _unique_joined(g["access"]),
            "house_number": _first_non_empty(g["house_number"]),
            "street": _first_non_empty(g["street"]),
            "city": _first_non_empty(g["city"]),
            "district": _first_non_empty(g["district"]),
            "county": _first_non_empty(g["county"]),
            "state_code": _first_non_empty(g["state_code"]),
            "postal_code": _first_non_empty(g["postal_code"]),
            "country_code": _first_non_empty(g["country_code"]),
            "here_station_id": _first_non_empty(g["here_station_id"]),
            "first_seen_utc": g["first_seen_utc"].min(),
            "last_seen_utc": g["last_seen_utc"].max(),
        })
    meta = pd.DataFrame(rows)
    print(f"[aggregate_metadata] {len(meta)} aggregated stations "
          f"(total units: {meta['num_units'].sum()}).")
    return meta


# ---------------------------------------------------------------------------
# 3) Aggregating the ev_locations_availability of a station (sum over the units)
# ---------------------------------------------------------------------------
def aggregate_station_availability(station_id: str,
                                   unit_ids: list[str],
                                   raw_avail_dir: os.PathLike | str) -> Optional[pd.DataFrame]:
    """Concatenate the units of a station and sum the measures per timestamp.

    Returns
    -------
    DataFrame with columns :data:`_AVAIL_OUT_COLS`, sorted by timestamp,
    or ``None`` if no unit has data.
    """
    raw_avail_dir = Path(raw_avail_dir)
    frames = []
    for uid in unit_ids:
        fpath = raw_avail_dir / f"{uid}.csv"
        if not fpath.exists():
            print(f"[aggregate_station_availability] WARNING: missing file {fpath.name}")
            continue
        u = pd.read_csv(fpath)
        if "collected_at" not in u.columns or u.empty:
            continue
        u["collected_at"] = pd.to_datetime(u["collected_at"], errors="coerce")
        u = u.dropna(subset=["collected_at"])
        for col in _RAW_NUMERIC_COLS:
            u[col] = pd.to_numeric(u.get(col), errors="coerce").fillna(0)
        frames.append(u[["collected_at", *_RAW_NUMERIC_COLS]])

    if not frames:
        return None

    allu = pd.concat(frames, ignore_index=True)
    agg = (allu.groupby("collected_at", as_index=False)[_RAW_NUMERIC_COLS]
                .sum()
                .sort_values("collected_at"))

    agg = agg.rename(columns=_AVAIL_RENAME)
    agg.insert(0, "location_id", station_id)
    agg = agg.rename(columns={"collected_at": "timestamp"})
    return agg[_AVAIL_OUT_COLS]


# ---------------------------------------------------------------------------
# 4) Writing all the ev_locations_availability files
# ---------------------------------------------------------------------------
def build_availability(raw: pd.DataFrame,
                       out_avail_dir: os.PathLike | str,
                       raw_avail_dir: os.PathLike | str) -> int:
    """Write an aggregated time-series CSV for each ``station_id``.

    Returns
    -------
    Number of ev_locations_availability files written.
    """
    out_avail_dir = Path(out_avail_dir)
    out_avail_dir.mkdir(parents=True, exist_ok=True)

    station_to_units = raw.groupby("station_id")["unit_id"].apply(list)
    n_written = 0
    n_total = len(station_to_units)
    for i, (station_id, unit_ids) in enumerate(station_to_units.items(), start=1):
        agg = aggregate_station_availability(station_id, unit_ids, raw_avail_dir)
        if agg is None or agg.empty:
            print(f"[build_availability] {station_id}: no data, skipped.")
            continue
        agg.to_csv(out_avail_dir / f"{station_id}.csv", index=False)
        n_written += 1
        if i % 200 == 0 or i == n_total:
            print(f"[build_availability] {i}/{n_total} stations processed "
                  f"({n_written} files written).")
    print(f"[build_availability] Done: {n_written} files in {out_avail_dir}")
    return n_written


# ---------------------------------------------------------------------------
# 5) Orchestrator
# ---------------------------------------------------------------------------
def run_preprocessing(raw_meta_file: os.PathLike | str,
                      raw_avail_dir: os.PathLike | str,
                      out_meta_file: os.PathLike | str,
                      out_avail_dir: os.PathLike | str,
                      write_availability: bool = True) -> pd.DataFrame:
    """Run the whole preprocessing: aggregated metadata + ev_locations_availability.

    Returns
    -------
    The DataFrame of the aggregated metadata (at station level).
    """
    raw = load_raw_metadata(raw_meta_file)

    meta = aggregate_metadata(raw)
    out_meta_file = Path(out_meta_file)
    out_meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_meta_file, index=False)
    print(f"[run_preprocessing] Metadata saved in {out_meta_file}")

    if write_availability:
        build_availability(raw,
                           out_avail_dir=out_avail_dir,
                           raw_avail_dir=raw_avail_dir)

    return meta


# ---------------------------------------------------------------------------
# 6) Map visualization of the EV stations
# ---------------------------------------------------------------------------
def visualize_ev_on_map(metadata: pd.DataFrame | os.PathLike | str,
                        output_html: os.PathLike | str) -> str:
    """Draw the EV stations on an interactive map (folium) and save an HTML.

    The style follows :func:`los_angeles_graph.visualize_graph_on_map`: dots on
    the position of each station, with an informative popup. The marker radius
    grows (weakly) with the number of units of the station.

    Parameters
    ----------
    metadata : DataFrame or path
        Metadata at station level (output of :func:`aggregate_metadata`).
        If it is a path, the CSV is loaded.
    """
    if not isinstance(metadata, pd.DataFrame):
        metadata = pd.read_csv(metadata)

    df = metadata.dropna(subset=["lat", "lng"]).copy()
    if df.empty:
        raise ValueError("No station with valid coordinates to visualize.")

    center = [df["lat"].mean(), df["lng"].mean()]
    fmap = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")
    layer = folium.FeatureGroup(name="EV stations")

    for _, s in df.iterrows():
        n_units = int(s["num_units"]) if "num_units" in s and pd.notna(s["num_units"]) else 1
        radius = 3 + min(n_units, 8)  # 4..11, saturates at 8 units
        popup = (
            f"<b>{s.get('station_id', '')}</b><br>"
            f"{s.get('title', '')}<br>"
            f"units: {n_units}<br>"
            f"max_power_kw: {s.get('max_power_kw', '')}<br>"
            f"{s.get('street', '')} {s.get('house_number', '')}, {s.get('city', '')}"
        )
        folium.CircleMarker(
            location=(float(s["lat"]), float(s["lng"])),
            radius=radius,
            color="#1f8a3a", fill=True, fill_color="#2ecc71", fill_opacity=0.75,
            popup=folium.Popup(popup, max_width=300),
        ).add_to(layer)

    layer.add_to(fmap)
    folium.LayerControl().add_to(fmap)

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_html))
    print(f"[visualize_ev_on_map] {len(df)} stations | map saved in {output_html}")
    return str(output_html)



if __name__ == "__main__":
    """
    Preprocessing of the ev_locations_availability data of the Los Angeles EV charging stations.

    Context
    -------
    The raw data live in::

        data/raw/losangeles_raw/EV Stations/LA_ev_stations_extracted/
            stations_metadata.csv          # 1 row per UNIT (unit_id)
            ev_locations_availability/<unit_id>.csv      # 1 file per UNIT, time series

    A physical "station" (``station_id``, e.g. ``S00001``) may contain several
    "units" (``unit_id``, e.g. ``S00001-1``, ``S00001-2``, ...) that share the
    same coordinates. We want to work at the **station** level, so here:

    1. **Metadata** — we merge the rows by ``station_id`` obtaining one row per
       station (coordinates, address, number of units/connectors, connector types,
       maximum power, observation time window).

    2. **Availability** — for each station we concatenate the files of its units
       and sum the measures aligning them by timestamp (``collected_at``). The
       output per station is an aggregated time series.

    The output format is modeled on ``data/raw/chicago/ev`` (single metadata file
    + one folder with a time-series CSV per station)::

        data/raw/losangeles/ev/
            ev_location_metadata.csv
            ev_locations_availability/<station_id>.csv   # location_id,timestamp,Available,Total,Offline,In_use

    Mapping of the ev_locations_availability columns (raw unit -> aggregated station)::

        Available <- sum(number_of_available)
        Total     <- sum(number_of_connectors)
        Offline   <- sum(number_of_out_of_service)
        In_use    <- sum(number_of_in_use)     # extra compared to Chicago

    The first five columns (``location_id, timestamp, Available, Total, Offline``)
    coincide with the Chicago format; ``In_use`` is a useful addition for LA.

    Available functions
    -------------------
    - :func:`load_raw_metadata`     load ``stations_metadata.csv`` (per unit)
    - :func:`aggregate_metadata`    merge the metadata at the ``station_id`` level
    - :func:`aggregate_station_availability`  sum the units of ONE station
    - :func:`build_availability`    write a time-series CSV per station
    - :func:`run_preprocessing`     orchestrator: metadata + ev_locations_availability
    - :func:`visualize_ev_on_map`   HTML map (folium) of the EV stations
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    DEFAULT_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "losangeles_raw", "EV Stations", "LA_ev_stations_extracted")
    DEFAULT_RAW_META_FILE = os.path.join(DEFAULT_RAW_DIR, "stations_metadata.csv")
    DEFAULT_RAW_AVAIL_DIR = os.path.join(DEFAULT_RAW_DIR, "ev_locations_availability")
    DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "losangeles", "ev")
    DEFAULT_OUT_META_FILE = os.path.join(DEFAULT_OUT_DIR, "ev_location_metadata.csv")
    DEFAULT_OUT_AVAIL_DIR = os.path.join(DEFAULT_OUT_DIR, "ev_locations_availability")

    # Numeric columns of the raw ev_locations_availability files (per unit) to be summed.
    _RAW_NUMERIC_COLS = ["number_of_available",
                         "number_of_in_use",
                         "number_of_out_of_service",
                         "number_of_connectors"]

    # Rename (aggregated raw -> final Chicago-style format).
    _AVAIL_RENAME = {"number_of_available": "Available",
                     "number_of_connectors": "Total",
                     "number_of_out_of_service": "Offline",
                     "number_of_in_use": "In_use"}

    # Final column order: the first 5 coincide with Chicago.
    _AVAIL_OUT_COLS = ["location_id", "timestamp", "Available", "Total", "Offline", "In_use"]

    RUN_PREPROCESSING = True
    RUN_VISUALIZE = True


    if RUN_PREPROCESSING:
        meta_df = run_preprocessing(raw_meta_file=DEFAULT_RAW_META_FILE,
                                    raw_avail_dir=DEFAULT_RAW_AVAIL_DIR,
                                    out_meta_file=DEFAULT_OUT_META_FILE,
                                    out_avail_dir=DEFAULT_OUT_AVAIL_DIR,
                                    write_availability=True)
    else:
        meta_df = None

    if RUN_VISUALIZE:
        visualize_ev_on_map(meta_df if meta_df is not None else DEFAULT_OUT_META_FILE,
                            output_html=DEFAULT_OUT_DIR / "ev_stations_map.html")