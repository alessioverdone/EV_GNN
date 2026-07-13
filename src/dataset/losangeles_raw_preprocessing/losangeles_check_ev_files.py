import argparse
from pathlib import Path
import pandas as pd


# Colonne necessarie al quality check (usate da is_good_ev_file/check_file).
# Valore di default cosi' i chiamanti possono invocare is_good_ev_file(path).
DEFAULT_REQUIRED_COLUMNS = ["Available", "Total", "Offline", "In_use"]


def check_file(csv_path: Path,
               REQUIRED_COLUMNS) -> int:
    """Return the number of rows in `csv_path` that violate the condition.

    Available + Offline + In_use == Total
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing}")

    expected_total = df["Available"] + df["Offline"] + df["In_use"]
    violations = expected_total != df["Total"]
    return int(violations.sum())


def has_only_zero_available(csv_path: Path) -> bool:
    """Return True if the `Available` column contains only zero values."""
    df = pd.read_csv(csv_path)

    if "Available" not in df.columns:
        raise ValueError(f"{csv_path.name}: missing column 'Available'")

    return bool((df["Available"] == 0).all())


def has_only_zero_column(csv_path: Path,
                         column: str) -> bool:
    """Return True if `column` contains only zero values."""
    df = pd.read_csv(csv_path)

    if column not in df.columns:
        raise ValueError(f"{csv_path.name}: missing column '{column}'")

    return bool((df[column] == 0).all())




def is_good_ev_file(csv_path,
                    REQUIRED_COLUMNS=DEFAULT_REQUIRED_COLUMNS) -> bool:
    """Return True if the EV ev_locations_availability CSV passes the strict quality check.

    A file is "good" if it meets ALL the following conditions:
    1. for each row, "Available + Offline + In_use == Total";
    2. the "Available" column is not entirely zero;
    3. the "In_use" column is not entirely zero.

    Intended to be used as a per-file filter (e.g., within a loading loop).
    It reads the CSV only once. If there are missing columns or the file is
    unreadable, it returns "False" (file discarded).

    Parameters
    ----------
    csv_path : str | os.PathLike
        Percorso del file CSV di disponibilita' di una stazione.
    """
    try:
        df = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS)
    except Exception:  # noqa: BLE001 - colonne mancanti o file illeggibile
        return False

    if not set(REQUIRED_COLUMNS).issubset(df.columns):
        return False

    # 1) condizione sulla somma per ogni riga
    expected_total = df["Available"] + df["Offline"] + df["In_use"]
    if (expected_total != df["Total"]).any():
        return False

    # 2) e 3) Available e In_use non tutte zero
    if (df["Available"] == 0).all():
        return False
    if (df["In_use"] == 0).all():
        return False

    return True


def check_directory(directory: Path,
                    REQUIRED_COLUMNS) -> dict:
    """Check every CSV file in `directory` and print a summary.

    Returns
    -------
    dict con le liste degli ``station_id`` (nome file senza estensione)
    classificati come ``good`` / ``not_good``, piu' la lista degli errori.
    """
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return {"good": [], "not_good": [], "errors": []}

    bad_files = []
    zero_available_files = []
    good_files = []
    not_good_files = []
    error_files = []

    for csv_path in csv_files:
        station_id = csv_path.stem
        try:
            n_violations = check_file(csv_path, REQUIRED_COLUMNS)
            only_zero = has_only_zero_available(csv_path)
            only_zero_in_use = has_only_zero_column(csv_path, "In_use")
        except Exception as exc:  # noqa: BLE001 - report and continue
            error_files.append((csv_path.name, str(exc)))
            continue

        if n_violations > 0:
            bad_files.append((csv_path.name, n_violations))
        if only_zero:
            zero_available_files.append(csv_path.name)
        # Files that satisfy the main condition AND have neither
        # 'Available' nor 'In_use' entirely zero.
        if n_violations == 0 and not only_zero and not only_zero_in_use:
            good_files.append(station_id)
        else:
            not_good_files.append(station_id)

    total = len(csv_files)
    print(f"Checked {total} CSV file(s) in {directory}")
    print(f"Files that DO NOT satisfy the condition: {len(bad_files)}")

    # if bad_files:
    #     print("\nDetails (file: number of violating rows):")
    #     for name, n in bad_files:
    #         print(f"  {name}: {n}")

    print(
        f"\nFiles with 'Available' column all zeros: "
        f"{len(zero_available_files)}"
    )
    # if zero_available_files:
    #     print("\nDetails (files with only-zero 'Available'):")
    #     for name in zero_available_files:
    #         print(f"  {name}")

    print(
        f"\nFiles that satisfy the condition AND have neither 'Available' "
        f"nor 'In_use' all zeros: {len(good_files)}"
    )

    if error_files:
        print(f"\nFiles that could not be read: {len(error_files)}")
        for name, msg in error_files:
            print(f"  {name}: {msg}")

    return {
        "good": good_files,
        "not_good": not_good_files,
        "errors": [name for name, _ in error_files],
    }


def visualize_quality_on_map(good_ids,
                             not_good_ids,
                             meta_file,
                             output_html,) -> str:
    """Draws the 'good' (blue) and 'bad' (red) files on the map.

    The coordinates are taken from the ``meta_file`` (one line per ``station_id``).
    Stations without valid coordinates are ignored.

    Parameters
    ----------
    good_ids, not_good_ids : iterable di station_id
        Identificativi (nome file senza estensione) delle due categorie.
    """
    import folium

    meta = pd.read_csv(meta_file)
    meta = meta.dropna(subset=["lat", "lng"])
    coords = meta.set_index("station_id")[["lat", "lng", "title", "num_units"]]

    center = [coords["lat"].mean(), coords["lng"].mean()]
    fmap = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")

    categories = [
        ("File buoni", good_ids, "#1f5fd6", "#3b82f6"),       # blu
        ("File non buoni", not_good_ids, "#a31515", "#ef4444"),  # rosso
    ]

    n_plotted = 0
    n_missing = 0
    for layer_name, ids, color, fill in categories:
        layer = folium.FeatureGroup(name=layer_name)
        for sid in ids:
            if sid not in coords.index:
                n_missing += 1
                continue
            s = coords.loc[sid]
            popup = (
                f"<b>{sid}</b><br>"
                f"{s.get('title', '')}<br>"
                f"unita': {s.get('num_units', '')}<br>"
                f"stato: {layer_name}"
            )
            folium.CircleMarker(
                location=(float(s["lat"]), float(s["lng"])),
                radius=5,
                color=color, fill=True, fill_color=fill, fill_opacity=0.8,
                weight=1,
                popup=folium.Popup(popup, max_width=300),
            ).add_to(layer)
            n_plotted += 1
        layer.add_to(fmap)

    folium.LayerControl().add_to(fmap)

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_html))
    print(
        f"[visualize_quality_on_map] {n_plotted} stazioni disegnate "
        f"({len(good_ids)} buone, {len(not_good_ids)} non buone; "
        f"{n_missing} senza coordinate). Mappa salvata in {output_html}"
    )
    return str(output_html)


def main() -> None:
    """Check EV ev_locations_availability CSV files for consistency.

    For every row in each CSV file of a given directory, the columns must satisfy:
        Available + Offline + In_use == Total

    The script reports how many files do NOT satisfy this condition for all of
    their rows.
    """

    # Default directory to check, relative to the repository root.
    DEFAULT_DIR = "/home/user/Scrivania/PhD/EV/code/EV_GNN_repo_2b/EV_GNN/data/raw/losangeles/ev/availability"

    # Metadati a livello di stazione (contiene station_id, lat, lng).
    DEFAULT_META_FILE = "/home/user/Scrivania/PhD/EV/code/EV_GNN_repo_2b/EV_GNN/data/raw/losangeles/ev/metadata.csv"

    # Mappa HTML di output.
    DEFAULT_MAP_HTML = "/home/user/Scrivania/PhD/EV/code/EV_GNN_repo_2b/EV_GNN/data/raw/losangeles/ev/ev_files_quality_map.html"

    REQUIRED_COLUMNS = ["Available", "Offline", "In_use", "Total"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        default=DEFAULT_DIR,
        help=f"Directory containing the CSV files (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--map",
        action="store_true",
        default=True,
        help="It also generates an HTML map (good ones in blue, bad ones in red).",
    )
    args = parser.parse_args()
    result = check_directory(Path(args.directory),
                             REQUIRED_COLUMNS)

    if args.map:
        visualize_quality_on_map(result["good"],
                                 result["not_good"],
                                 meta_file=DEFAULT_META_FILE,
                                 output_html=DEFAULT_MAP_HTML)


if __name__ == "__main__":
    main()
