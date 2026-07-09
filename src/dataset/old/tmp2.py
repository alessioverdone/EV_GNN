#!/usr/bin/env python3
"""
PEMS08 Graph Visualizer — Download Automatico + Analisi Componenti Connesse
===========================================================================

Questo script visualizza il grafo di PEMS08.

Cosa fa:
  • scarica automaticamente l'edge-list PEMS08 se non è già presente;
  • opzionalmente scarica anche PeMS08.npz per stampare shape e info del dataset;
  • costruisce il grafo NetworkX;
  • controlla se il grafo è connesso, quante componenti ha, nodi isolati, ecc.;
  • genera una mappa HTML interattiva con Folium.

Output:
    pems08_graph_map.html

Uso:
    python visualize_pems08.py

Dipendenze:
    pip install folium pandas numpy networkx

Coordinate reali:
    Nei benchmark PEMS03/04/07/08 l'edge-list usa indici 0..N-1.
    Per usare coordinate GPS reali servono due file locali:

      • PEMS08_sensor_ids.csv  → mapping indice grafo -> sensor ID PeMS reale
      • ca_meta.csv            → metadati LargeST con ID, Lat, Lng, ecc.

    Se questi file non sono presenti o non matchano, lo script visualizza
    un layout topologico/schematic, NON geografico.
"""

import os
import sys
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


# ─── Controlla dipendenze ────────────────────────────────────────────────────
missing = []
for pkg in ["folium", "networkx", "pandas", "numpy"]:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"[!] Librerie mancanti: {', '.join(missing)}")
    print(f"    pip install {' '.join(missing)}")
    sys.exit(1)

import folium
from folium.plugins import HeatMap, MiniMap


# ─── Configurazione file ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

DATASET_NAME = "PEMS08"
EXPECTED_N_NODES = 170

FILE_EDGES = SCRIPT_DIR / "PEMS08_W.csv"
FILE_NPZ = SCRIPT_DIR / "PeMS08.npz"
FILE_CA_META = SCRIPT_DIR / "ca_meta.csv"
OUTPUT_HTML = SCRIPT_DIR / "pems08_graph_map.html"

# Edge-list PEMS08 pubblica: colonne from,to,cost
PEMS08_EDGE_URLS = [
    "https://raw.githubusercontent.com/peakdemo/CGLGCN/main/PeMS08.csv",
]

# Serie temporale PEMS08, non indispensabile per disegnare il grafo.
# La scarico solo per stampare info/shape se possibile.
PEMS08_NPZ_URLS = [
    "https://raw.githubusercontent.com/peakdemo/CGLGCN/main/PeMS08.npz",
]

AUTO_DOWNLOAD_EDGES = True
AUTO_DOWNLOAD_NPZ = True

# Nomi alternativi per il file archi
EDGE_FILE_CANDIDATES = [
    "PEMS08_W.csv",
    "PEMS08_edges.csv",
    "PEMS08_adj.csv",
    "PeMS08.csv",
    "PEMS08.csv",
    "W.csv",
    "distances.csv",
    "distance.csv",
]

# File opzionali per coordinate reali
SENSOR_ID_FILE_CANDIDATES = [
    "PEMS08_sensor_ids.csv",
    "PEMS08_ids.csv",
    "PEMS08_nodes.csv",
    "sensor_ids.csv",
    "nodes.csv",
]


# ════════════════════════════════════════════════════════════════════════════
# STEP 0 — Download automatico
# ════════════════════════════════════════════════════════════════════════════

def download_file(urls: list[str], out_path: Path, description: str) -> bool:
    """
    Prova a scaricare out_path da una lista di URL.
    Restituisce True se il file esiste alla fine.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"      OK già presente: {out_path.name}")
        return True

    for url in urls:
        try:
            print(f"      Download {description}:")
            print(f"      {url}")
            urllib.request.urlretrieve(url, out_path)
            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"      Salvato: {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
                return True
        except Exception as e:
            print(f"      [!] Download fallito da questo URL: {e}")

    if out_path.exists() and out_path.stat().st_size == 0:
        try:
            out_path.unlink()
        except Exception:
            pass

    print(f"      [!] Non sono riuscito a scaricare {description}.")
    return False


def ensure_downloads() -> None:
    """
    Scarica automaticamente i file principali se mancano.
    """
    print("\n[0/7] Controllo/download file PEMS08 …")

    existing_edge = find_edge_file(SCRIPT_DIR, quiet=True)
    if existing_edge is None:
        if AUTO_DOWNLOAD_EDGES:
            download_file(PEMS08_EDGE_URLS, FILE_EDGES, "edge-list PEMS08")
        else:
            print("      Edge-list mancante e AUTO_DOWNLOAD_EDGES=False.")
    else:
        print(f"      Edge-list trovata: {existing_edge.name}")

    if AUTO_DOWNLOAD_NPZ:
        # Non blocca lo script se fallisce: per visualizzare il grafo non è necessario.
        download_file(PEMS08_NPZ_URLS, FILE_NPZ, "serie temporale PeMS08.npz")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Carica eventuale dataset NPZ
# ════════════════════════════════════════════════════════════════════════════

def print_npz_info(path: Path) -> None:
    """
    Stampa informazioni sul file PeMS08.npz, se disponibile.
    """
    print("[1/7] Controllo serie temporale PeMS08.npz …")

    if not path.exists():
        print("      File NPZ non presente. Continuo solo con il grafo.")
        return

    try:
        obj = np.load(path, allow_pickle=True)
        print(f"      File trovato: {path.name}")
        print(f"      Chiavi: {list(obj.keys())}")

        for key in obj.keys():
            arr = obj[key]
            print(f"      {key}: shape={arr.shape}, dtype={arr.dtype}")

        if "data" in obj:
            data = obj["data"]
            if data.ndim == 3:
                print(f"      Interpretazione probabile: timestep={data.shape[0]}, nodi={data.shape[1]}, feature={data.shape[2]}")
    except Exception as e:
        print(f"      [!] Impossibile leggere NPZ: {e}")
        print("      Continuo solo con il grafo.")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Carica edge-list del grafo
# ════════════════════════════════════════════════════════════════════════════

def find_edge_file(script_dir: Path, quiet: bool = False) -> Path | None:
    for name in EDGE_FILE_CANDIDATES:
        path = script_dir / name
        if path.exists() and path.stat().st_size > 0:
            return path

    if not quiet:
        print("  [!] File archi non trovato. Nomi cercati:")
        for f in EDGE_FILE_CANDIDATES:
            print(f"       • {f}")
    return None


def normalize_edge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza colonne edge-list verso: from, to, cost.
    """
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        if c in ("from", "src", "source", "start", "origin", "i"):
            rename[c] = "from"
        if c in ("to", "dst", "dest", "end", "target", "j"):
            rename[c] = "to"
        if c in ("cost", "weight", "distance", "dist", "w"):
            rename[c] = "cost"

    df = df.rename(columns=rename)

    if {"from", "to", "cost"}.issubset(df.columns):
        out = df[["from", "to", "cost"]].copy()
    elif df.shape[1] >= 3:
        out = df.iloc[:, :3].copy()
        out.columns = ["from", "to", "cost"]
    else:
        raise ValueError("Il CSV non contiene abbastanza colonne per una edge-list.")

    out["from"] = pd.to_numeric(out["from"], errors="coerce")
    out["to"] = pd.to_numeric(out["to"], errors="coerce")
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")
    out = out.dropna()

    out["from"] = out["from"].astype(int)
    out["to"] = out["to"].astype(int)
    out["cost"] = out["cost"].astype(float)

    return out


def load_edges(path: Path) -> pd.DataFrame:
    """
    Carica una edge-list PEMS08.

    Supporta:
      • CSV normale con header from,to,cost
      • CSV senza header
      • fallback regex se il file è finito su una sola riga
    """
    # Tentativo standard
    try:
        df = pd.read_csv(path)
        out = normalize_edge_columns(df)
        if len(out) > 0:
            return out
    except Exception:
        pass

    # Tentativo senza header
    try:
        df = pd.read_csv(path, header=None)
        out = normalize_edge_columns(df)
        if len(out) > 0:
            return out
    except Exception:
        pass

    # Fallback robusto: cerca triple tipo 9,153,310.6
    text = path.read_text(encoding="utf-8", errors="ignore")
    triples = re.findall(r"(-?\d+)\s*,\s*(-?\d+)\s*,\s*([0-9]+(?:\.[0-9]+)?)", text)

    rows = []
    for a, b, c in triples:
        # Evita di interpretare eventuale header strano.
        rows.append((int(a), int(b), float(c)))

    if not rows:
        raise ValueError(f"{path} non sembra una edge-list valida.")

    return pd.DataFrame(rows, columns=["from", "to", "cost"])


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Coordinate reali opzionali
# ════════════════════════════════════════════════════════════════════════════

def find_sensor_id_file(script_dir: Path) -> Path | None:
    for name in SENSOR_ID_FILE_CANDIDATES:
        path = script_dir / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def load_sensor_ids(path: Path) -> list[int]:
    """
    Legge un file indice->sensor_id.
    Formati supportati:
      • una colonna senza header;
      • header sensor_id;
      • header id;
      • prima colonna qualsiasi.
    """
    raw = pd.read_csv(path, header=None, nrows=3)
    first_val = str(raw.iloc[0, 0]).strip()

    if first_val.lstrip("-").isdigit():
        df = pd.read_csv(path, header=None, names=["id"])
        return df["id"].astype(int).tolist()

    df = pd.read_csv(path)
    cols_lower = {c.lower().strip(): c for c in df.columns}
    col = cols_lower.get("sensor_id") or cols_lower.get("id") or df.columns[0]
    return df[col].astype(int).tolist()


def load_ca_meta(path: Path) -> pd.DataFrame:
    """
    Legge ca_meta.csv da LargeST.
    Normalizza varianti:
      LargeST: ID, Lat, Lng, District, County, Fwy, Lane, Type, Direction
      altri:   ID, Latitude, Longitude, ...
    """
    df = pd.read_csv(path)

    # Se c'è una colonna indice finta, rimuovila
    if len(df.columns) > 1 and str(df.columns[0]).lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])

    cols_lower = {c.lower().strip(): c for c in df.columns}

    def get(candidates, default=None):
        for c in candidates:
            if c in cols_lower:
                return df[cols_lower[c]]
        return pd.Series([default] * len(df))

    result = pd.DataFrame({
        "sensor_id": get(["id", "sensor_id"]).astype(int),
        "lat": pd.to_numeric(get(["lat", "latitude"]), errors="coerce"),
        "lon": pd.to_numeric(get(["lng", "lon", "longitude"]), errors="coerce"),
        "fwy": get(["fwy"], "N/A").astype(str),
        "district": get(["district"], "N/A"),
        "county": get(["county"], "N/A"),
        "direction": get(["direction", "dir"], "N/A").astype(str),
    })

    return result.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def try_build_real_coords(G: nx.Graph):
    """
    Prova a costruire coordinate reali tramite:
      PEMS08_sensor_ids.csv + ca_meta.csv.

    Ritorna:
      coords, sensor_ids, meta_lookup, using_real_coords
    """
    sensor_file = find_sensor_id_file(SCRIPT_DIR)

    if sensor_file is None:
        print("      Nessun file sensor IDs trovato.")
        print("      Cerco uno di:", ", ".join(SENSOR_ID_FILE_CANDIDATES))
        return {}, [], {}, False

    if not FILE_CA_META.exists():
        print(f"      Trovato {sensor_file.name}, ma manca ca_meta.csv.")
        return {}, [], {}, False

    try:
        sensor_ids = load_sensor_ids(sensor_file)
        meta_df = load_ca_meta(FILE_CA_META)
        meta_lookup = meta_df.set_index("sensor_id").to_dict(orient="index")

        coords = {}
        matched = 0
        unmatched = 0

        for node_idx in G.nodes():
            if node_idx < len(sensor_ids):
                sid = sensor_ids[node_idx]
                if sid in meta_lookup:
                    row = meta_lookup[sid]
                    coords[node_idx] = (float(row["lat"]), float(row["lon"]))
                    matched += 1
                else:
                    unmatched += 1
            else:
                unmatched += 1

        print(f"      File sensor IDs: {sensor_file.name}")
        print(f"      ca_meta.csv righe: {len(meta_df)}")
        print(f"      Nodi con coordinate reali: {matched}")
        print(f"      Nodi senza coordinate: {unmatched}")

        if matched >= max(10, int(0.5 * G.number_of_nodes())):
            return coords, sensor_ids, meta_lookup, True

        print("      [!] Match troppo basso: uso layout topologico.")
        return {}, sensor_ids, meta_lookup, False

    except Exception as e:
        print(f"      [!] Errore coordinate reali: {e}")
        return {}, [], {}, False


def make_schematic_coords(G: nx.Graph) -> dict:
    """
    Crea coordinate schematiche, non geografiche, usando spring_layout.
    Le coordinate sono arbitrarie e vengono usate con CRS.Simple in Folium.
    """
    print("      Genero layout topologico spring_layout, NON coordinate GPS …")

    pos = nx.spring_layout(
        G,
        seed=42,
        iterations=250,
        weight="inv_cost",
    )

    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    coords = {}
    for node, (x, y) in pos.items():
        # Scala in range circa 0..1000 per Leaflet CRS.Simple
        sx = 1000 * (x - x_min) / (x_max - x_min + 1e-9)
        sy = 1000 * (y - y_min) / (y_max - y_min + 1e-9)
        coords[node] = (sy, sx)

    return coords


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Analisi componenti connesse
# ════════════════════════════════════════════════════════════════════════════

CC_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


def analyze_components(G: nx.Graph) -> dict:
    """
    Calcola componenti connesse e mapping nodo->componente/colore.
    """
    ccs = sorted(nx.connected_components(G), key=len, reverse=True)
    n_cc = len(ccs)

    node_to_cc = {}
    node_to_color = {}

    for cc_idx, cc in enumerate(ccs):
        color = CC_PALETTE[cc_idx % len(CC_PALETTE)]
        for node in cc:
            node_to_cc[node] = cc_idx
            node_to_color[node] = color

    stats = {
        "n_components": n_cc,
        "is_connected": n_cc == 1,
        "components": ccs,
        "node_to_cc": node_to_cc,
        "node_to_color": node_to_color,
        "sizes": [len(cc) for cc in ccs],
    }
    return stats


def print_component_report(stats: dict, sensor_ids: list[int] | None = None) -> None:
    """
    Stampa report dettagliato componenti connesse.
    """
    sensor_ids = sensor_ids or []

    print()
    print("═" * 60)
    print("  ANALISI COMPONENTI CONNESSE — PEMS08")
    print("═" * 60)

    if stats["is_connected"]:
        print("  ✅  Il grafo è CONNESSO: una sola componente.")
    else:
        n = stats["n_components"]
        print(f"  ⚠️   Il grafo NON è connesso: {n} componenti separate.\n")
        print(f"  {'CC':>4}  {'Nodi':>6}  {'Colore':<10}  Nodi/sensori esempio")
        print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*40}")

        for i, (cc, size) in enumerate(zip(stats["components"], stats["sizes"])):
            color = CC_PALETTE[i % len(CC_PALETTE)]
            sample = sorted(cc)[:8]

            if sensor_ids and max(sample) < len(sensor_ids):
                sample_ids = [sensor_ids[n] for n in sample]
            else:
                sample_ids = sample

            suffix = " …" if size > 8 else ""
            print(f"  {i:>4}  {size:>6}  {color:<10}  {sample_ids}{suffix}")

    total = sum(stats["sizes"]) if stats["sizes"] else 0
    print()
    print(f"  Nodi totali   : {total}")

    if total > 0:
        print(f"  CC più grande : {stats['sizes'][0]} nodi ({stats['sizes'][0] / total * 100:.1f}%)")

    if len(stats["sizes"]) > 1:
        print(f"  CC più piccola: {stats['sizes'][-1]} nodi")
        isolated = sum(1 for s in stats["sizes"] if s == 1)
        if isolated:
            print(f"  Nodi isolati  : {isolated}")

    print("═" * 60)
    print()


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Costruisci mappa Folium
# ════════════════════════════════════════════════════════════════════════════

def build_map(
    G: nx.Graph,
    coords: dict,
    sensor_ids: list[int],
    meta_lookup: dict,
    stats: dict,
    using_real_coords: bool,
) -> folium.Map:

    lats = [c[0] for c in coords.values()]
    lons = [c[1] for c in coords.values()]

    if using_real_coords:
        center = [float(np.mean(lats)), float(np.mean(lons))]
        m = folium.Map(
            location=center,
            zoom_start=10,
            tiles="CartoDB dark_matter",
            control_scale=True,
        )
        MiniMap(toggle_display=True, tile_layer="CartoDB dark_matter").add_to(m)
    else:
        # CRS.Simple = piano cartesiano, non mappa geografica.
        m = folium.Map(
            location=[500, 500],
            zoom_start=-1,
            tiles=None,
            crs="Simple",
            control_scale=True,
        )

        folium.Rectangle(
            bounds=[[0, 0], [1000, 1000]],
            color="#222244",
            fill=True,
            fill_color="#080818",
            fill_opacity=1.0,
            weight=1,
        ).add_to(m)

        m.fit_bounds([[0, 0], [1000, 1000]])

    n_cc = stats["n_components"]
    node_cc = stats["node_to_cc"]
    node_cl = stats["node_to_color"]

    # ── Layer archi ──────────────────────────────────────────────────────────
    edge_group = folium.FeatureGroup(name="🔗 Archi (connessioni)", show=True)

    costs = [d["weight"] for _, _, d in G.edges(data=True)]
    c_min, c_max = min(costs), max(costs)

    def cost_color(c: float) -> str:
        t = (c - c_min) / (c_max - c_min + 1e-9)
        r = int(220 * t)
        g = int(200 * (1 - t))
        return f"#{r:02x}{g:02x}44"

    for u, v, d in G.edges(data=True):
        if u not in coords or v not in coords:
            continue

        lat1, lon1 = coords[u]
        lat2, lon2 = coords[v]

        folium.PolyLine(
            locations=[[lat1, lon1], [lat2, lon2]],
            color=cost_color(float(d["weight"])),
            weight=1.4,
            opacity=0.58,
            tooltip=f"Arco {u}↔{v} | cost: {float(d['weight']):.3f}",
        ).add_to(edge_group)

    edge_group.add_to(m)

    # ── Layer nodi per componente connessa ───────────────────────────────────
    cc_groups = {}

    for i in range(n_cc):
        label = f"CC{i} ({stats['sizes'][i]} nodi)" if n_cc > 1 else "📍 Sensori"
        cc_groups[i] = folium.FeatureGroup(
            name=f"📍 {label}",
            show=True,
        )

    degree = dict(G.degree())
    max_degree = max(degree.values()) if degree else 1

    for node_idx in sorted(G.nodes()):
        if node_idx not in coords:
            continue

        lat, lon = coords[node_idx]
        cc_idx = node_cc.get(node_idx, 0)
        color = node_cl.get(node_idx, "#00ffcc")
        deg = degree.get(node_idx, 0)
        radius = 3.5 + 4 * (deg / max_degree)

        sid = sensor_ids[node_idx] if node_idx < len(sensor_ids) else node_idx

        if using_real_coords:
            meta_row = meta_lookup.get(sid, {})
            fwy = meta_row.get("fwy", "N/A")
            county = meta_row.get("county", "N/A")
            district = meta_row.get("district", "N/A")
            direction = meta_row.get("direction", "N/A")

            popup_html = (
                f"<b>Sensore #{sid}</b><br>"
                f"Indice grafo: {node_idx}<br>"
                f"Autostrada: {fwy} {direction}<br>"
                f"Contea: {county} (D{district})<br>"
                f"Grado: {deg}<br>"
                f"Lat: {lat:.5f} &nbsp; Lon: {lon:.5f}<br>"
                f"<span style='color:{color}'>● CC{cc_idx} ({stats['sizes'][cc_idx]} nodi)</span>"
            )
            tooltip = f"#{sid} | {fwy} | grado {deg} | CC{cc_idx}"
        else:
            popup_html = (
                f"<b>Nodo indice {node_idx}</b><br>"
                f"Dataset: PEMS08<br>"
                f"Coordinate: layout topologico, non GPS<br>"
                f"Grado: {deg}<br>"
                f"X/Y layout: {lon:.2f}, {lat:.2f}<br>"
                f"<span style='color:{color}'>● CC{cc_idx} ({stats['sizes'][cc_idx]} nodi)</span>"
            )
            tooltip = f"Nodo {node_idx} | grado {deg} | CC{cc_idx}"

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.88,
            weight=0.6,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(cc_groups[cc_idx])

    for grp in cc_groups.values():
        grp.add_to(m)

    # ── HeatMap solo se coordinate reali ─────────────────────────────────────
    if using_real_coords:
        heat_group = folium.FeatureGroup(name="🌡️ Heatmap densità", show=False)
        HeatMap(
            [[coords[n][0], coords[n][1]] for n in G.nodes() if n in coords],
            radius=18,
            blur=22,
            min_opacity=0.3,
        ).add_to(heat_group)
        heat_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ── Titolo e legenda ─────────────────────────────────────────────────────
    if n_cc > 1:
        badges = "".join(
            f'<span style="background:{CC_PALETTE[i % len(CC_PALETTE)]};'
            f'color:#000;padding:1px 6px;border-radius:3px;margin:2px;'
            f'font-size:10px;display:inline-block;">CC{i}: {stats["sizes"][i]}</span>'
            for i in range(min(n_cc, 8))
        )
        more = f'<span style="font-size:10px;opacity:.7"> +{n_cc - 8} altre</span>' if n_cc > 8 else ""
        cc_info = (
            f'<div style="margin-top:6px;font-size:11px;opacity:.85;">'
            f'⚠️ <b>{n_cc} componenti sconnesse</b><br>'
            f'{badges}{more}'
            f'</div>'
        )
    else:
        cc_info = '<div style="margin-top:5px;font-size:11px;color:#3f3;">✅ Grafo connesso</div>'

    coord_mode = (
        "coordinate GPS reali"
        if using_real_coords
        else "layout topologico — non geografico"
    )

    tiles_note = (
        "San Bernardino/Riverside · District 8"
        if using_real_coords
        else "coordinate generate da spring_layout"
    )

    legend_html = f"""
    <div style="
        position:fixed; top:14px; left:50%; transform:translateX(-50%);
        z-index:9999; background:rgba(8,8,24,0.92);
        color:#e0f0ff; padding:10px 20px 10px 20px;
        border-radius:9px; font-family:'Courier New',monospace;
        font-size:13px; border:1px solid #00ffcc44;
        pointer-events:none; text-align:center; max-width:560px;">
      <b style="font-size:15px;color:#00ffcc;">PEMS08 — Traffic Sensor Graph</b><br>
      <span style="font-size:11px;opacity:.78;">
        {G.number_of_nodes()} nodi · {G.number_of_edges()} archi · {tiles_note}
      </span><br>
      <span style="font-size:11px;opacity:.9;">
        Modalità: <b>{coord_mode}</b>
      </span><br>
      <span style="font-size:11px;">
        Archi: <span style="color:#00ff28;">■</span> costo basso &nbsp;→&nbsp;
               <span style="color:#dc4400;">■</span> costo alto
      </span>
      {cc_info}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — Statistiche grafo
# ════════════════════════════════════════════════════════════════════════════

def print_graph_stats(G: nx.Graph, stats: dict) -> None:
    degrees = [d for _, d in G.degree()]

    print("─" * 60)
    print("  STATISTICHE GRAFO — PEMS08")
    print("─" * 60)
    print(f"  Nodi totali          : {G.number_of_nodes()}")
    print(f"  Archi totali         : {G.number_of_edges()}")
    print(f"  Grado medio          : {np.mean(degrees):.2f}")
    print(f"  Grado massimo        : {max(degrees)} (nodo {max(G.degree(), key=lambda x: x[1])[0]})")
    print(f"  Grado minimo         : {min(degrees)}")
    print(f"  Densità grafo        : {nx.density(G):.5f}")

    try:
        gcc = G.subgraph(stats["components"][0]).copy()
        print(f"  Diametro CC0         : {nx.diameter(gcc)}")
    except Exception:
        pass

    try:
        gcc = G.subgraph(stats["components"][0]).copy()
        print(f"  Path medio CC0       : {nx.average_shortest_path_length(gcc):.2f}")
    except Exception:
        pass

    print("─" * 60)
    print()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ensure_downloads()

    print_npz_info(FILE_NPZ)

    print("[2/7] Carico archi del grafo PEMS08 …")
    edge_path = find_edge_file(SCRIPT_DIR)

    if edge_path is None:
        print("  [!] Edge-list non trovata e download automatico fallito.")
        sys.exit(1)

    print(f"      Uso: {edge_path.name}")
    edges_df = load_edges(edge_path)
    print(f"      {len(edges_df)} archi caricati.")
    print(f"      Prime righe:")
    print(edges_df.head().to_string(index=False))

    print("[3/7] Costruisco il grafo NetworkX …")
    G = nx.Graph()

    # Aggiunge tutti i nodi attesi, così eventuali isolati vengono contati.
    G.add_nodes_from(range(EXPECTED_N_NODES))

    for _, row in edges_df.iterrows():
        u = int(row["from"])
        v = int(row["to"])
        w = float(row["cost"])

        if u == v:
            continue

        # Peso originale = cost/distance.
        # Peso inverso = utile per layout: archi con cost più basso più "forti".
        G.add_edge(
            u,
            v,
            weight=w,
            cost=w,
            inv_cost=1.0 / (w + 1e-9),
        )

    print(f"      Nodi: {G.number_of_nodes()}")
    print(f"      Archi: {G.number_of_edges()}")

    observed_max_node = max(max(edges_df["from"]), max(edges_df["to"]))
    if observed_max_node >= EXPECTED_N_NODES:
        print(f"      [!] Attenzione: trovato nodo massimo {observed_max_node}, ma EXPECTED_N_NODES={EXPECTED_N_NODES}.")

    print("[4/7] Provo coordinate reali opzionali …")
    coords, sensor_ids, meta_lookup, using_real_coords = try_build_real_coords(G)

    if not using_real_coords:
        print("      Uso layout topologico/schematic.")
        coords = make_schematic_coords(G)
        sensor_ids = []

    print("[5/7] Analisi componenti connesse …")
    stats = analyze_components(G)
    print_component_report(stats, sensor_ids=sensor_ids)

    print_graph_stats(G, stats)

    print("[6/7] Genero mappa interattiva …")
    m = build_map(
        G=G,
        coords=coords,
        sensor_ids=sensor_ids,
        meta_lookup=meta_lookup,
        stats=stats,
        using_real_coords=using_real_coords,
    )

    m.save(OUTPUT_HTML)

    print("[7/7] Fine.")
    print(f"\n  ✅  Mappa salvata in: {OUTPUT_HTML}")
    print("      Apri questo file nel browser.\n")

    if not using_real_coords:
        print("  Nota:")
        print("      La mappa usa un layout topologico, non coordinate geografiche reali.")
        print("      Per GPS reali servono PEMS08_sensor_ids.csv + ca_meta.csv nella stessa cartella.")
        print()


if __name__ == "__main__":
    main()