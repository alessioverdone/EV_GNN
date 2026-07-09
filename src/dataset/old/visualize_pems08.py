"""
PEMS08 Graph Visualizer — Coordinate Reali + Analisi Componenti Connesse
=========================================================================

File richiesti nella stessa directory:
  • ca_meta.csv   → metadati LargeST (Kaggle: liuxu77/largest)
                    colonne: ID, Lat, Lng, District, County, Fwy, Lanes, Type, Direction, ID2
  • PEMS08.csv    → archi del grafo PEMS08 (from, to, cost)

Mapping nodo → coordinate:
  I 170 sensori di PEMS08 corrispondono a ID2 nell'intervallo [665, 834].
  graph_idx = ID2 - 665

Area geografica: San Francisco Bay Area (District 4 — Alameda, Contra Costa, Santa Clara)

Uso:
    python visualize_pems08.py

Output:
    pems08_graph_map.html

Dipendenze:
    pip install folium pandas numpy networkx
"""

import os, sys
import numpy as np
import pandas as pd
import networkx as nx

# ── Dipendenze ────────────────────────────────────────────────────────────────
missing = []
for pkg in ["folium", "networkx", "pandas", "numpy"]:
    try: __import__(pkg)
    except ImportError: missing.append(pkg)
if missing:
    print(f"[!] Librerie mancanti: {', '.join(missing)}")
    print(f"    pip install {' '.join(missing)}")
    sys.exit(1)

import folium
from folium.plugins import HeatMap, MiniMap

# ── Percorsi file ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
FILE_CA_META = os.path.join(SCRIPT_DIR, "ca_meta.csv")
FILE_EDGES   = os.path.join(SCRIPT_DIR, "PEMS08.csv")
OUTPUT_HTML  = os.path.join(SCRIPT_DIR, "pems08_graph_map.html")

# Offset ID2 in ca_meta per PEMS08 (170 sensori, District 4 — Bay Area)
PEMS08_ID2_OFFSET = 665
PEMS08_N_NODES    = 170

# ── Palette colori per componenti connesse ────────────────────────────────────
CC_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#e6beff", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffe119",
]


# ════════════════════════════════════════════════════════════════════════════
# ANALISI COMPONENTI CONNESSE
# ════════════════════════════════════════════════════════════════════════════

def analyze_components(G: nx.Graph) -> dict:
    ccs = sorted(nx.connected_components(G), key=len, reverse=True)
    node_to_cc    = {}
    node_to_color = {}
    for cc_idx, cc in enumerate(ccs):
        color = CC_PALETTE[cc_idx % len(CC_PALETTE)]
        for node in cc:
            node_to_cc[node]    = cc_idx
            node_to_color[node] = color
    return {
        "n_components":  len(ccs),
        "is_connected":  len(ccs) == 1,
        "components":    ccs,
        "node_to_cc":    node_to_cc,
        "node_to_color": node_to_color,
        "sizes":         [len(cc) for cc in ccs],
    }


def print_component_report(G: nx.Graph, stats: dict, meta_df: pd.DataFrame) -> None:
    print()
    print("═" * 65)
    print("  ANALISI COMPONENTI CONNESSE — PEMS08")
    print("═" * 65)

    if stats["is_connected"]:
        print("  ✅  Il grafo è CONNESSO (una sola componente).")
        try:
            print(f"  Diametro: {nx.diameter(G)}")
        except Exception:
            pass
    else:
        n = stats["n_components"]
        print(f"  ⚠️   Il grafo NON è connesso: {n} componenti separate.\n")
        print(f"  {'#':>3}  {'Nodi':>5}  {'%':>5}  {'Colore':<10}  Autostrade principali")
        print(f"  {'-'*3}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*35}")

        total = G.number_of_nodes()
        for i, cc in enumerate(stats["components"]):
            color   = CC_PALETTE[i % len(CC_PALETTE)]
            size    = len(cc)
            pct     = size / total * 100
            cc_meta = meta_df[meta_df["graph_idx"].isin(cc)]
            fwy_str = ", ".join(
                f"{f}({c})" for f, c in cc_meta["Fwy"].value_counts().head(3).items()
            ) if not cc_meta.empty else "N/A"
            print(f"  {i:>3}  {size:>5}  {pct:>4.1f}%  {color:<10}  {fwy_str}")

    degrees = [d for _, d in G.degree()]
    print()
    print(f"  Nodi totali   : {G.number_of_nodes()}")
    print(f"  Archi totali  : {G.number_of_edges()}")
    print(f"  Grado medio   : {np.mean(degrees):.2f}")
    print(f"  Grado max     : {max(degrees)}  (nodo {max(G.degree(), key=lambda x: x[1])[0]})")
    print(f"  Densità       : {nx.density(G):.5f}")
    if not stats["is_connected"]:
        print(f"  CC più grande : {stats['sizes'][0]} nodi ({stats['sizes'][0]/G.number_of_nodes()*100:.1f}%)")
        print(f"  CC più piccola: {stats['sizes'][-1]} nodi")
        isolati = sum(1 for s in stats["sizes"] if s == 1)
        if isolati:
            print(f"  Nodi isolati  : {isolati}")
        try:
            gcc = G.subgraph(stats["components"][0])
            print(f"  Diametro CC0  : {nx.diameter(gcc)}")
        except Exception:
            pass
    else:
        try:
            print(f"  Diametro      : {nx.diameter(G)}")
        except Exception:
            pass
    print("═" * 65)
    print()


# ════════════════════════════════════════════════════════════════════════════
# MAPPA FOLIUM
# ════════════════════════════════════════════════════════════════════════════

def build_map(G: nx.Graph,
              coords: dict,
              meta_df: pd.DataFrame,
              stats: dict) -> folium.Map:

    lats = [c[0] for c in coords.values()]
    lons = [c[1] for c in coords.values()]
    center = [np.mean(lats), np.mean(lons)]

    m = folium.Map(
        location=center,
        zoom_start=10,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )
    MiniMap(toggle_display=True, tile_layer="CartoDB dark_matter").add_to(m)

    n_cc       = stats["n_components"]
    node_cc    = stats["node_to_cc"]
    node_color = stats["node_to_color"]

    # ── Archi ──────────────────────────────────────────────────────────────
    edge_group = folium.FeatureGroup(name="🔗 Archi", show=True)
    costs  = [d["weight"] for _, _, d in G.edges(data=True)]
    c_min, c_max = min(costs), max(costs)

    def edge_color(c):
        t = (c - c_min) / (c_max - c_min + 1e-9)
        return f"#{int(220*t):02x}{int(200*(1-t)):02x}44"

    for u, v, d in G.edges(data=True):
        if u not in coords or v not in coords:
            continue
        lat1, lon1 = coords[u]
        lat2, lon2 = coords[v]
        folium.PolyLine(
            locations=[[lat1, lon1], [lat2, lon2]],
            color=edge_color(d["weight"]),
            weight=1.5, opacity=0.6,
            tooltip=f"Arco {u}↔{v} | dist: {d['weight']:.1f}",
        ).add_to(edge_group)
    edge_group.add_to(m)

    # ── Nodi — un layer per componente connessa ────────────────────────────
    meta_idx = meta_df.set_index("graph_idx")
    degree   = dict(G.degree())
    max_deg  = max(degree.values()) if degree else 1

    cc_groups = {}
    for i in range(n_cc):
        size  = stats["sizes"][i]
        label = f"CC{i:02d} · {size} nodi" if n_cc > 1 else "📍 Tutti i sensori"
        cc_groups[i] = folium.FeatureGroup(name=f"📍 {label}", show=True)

    for node_idx in sorted(G.nodes()):
        if node_idx not in coords:
            continue
        lat, lon  = coords[node_idx]
        cc_idx    = node_cc.get(node_idx, 0)
        color     = node_color.get(node_idx, "#00ffcc")
        deg       = degree.get(node_idx, 0)
        radius    = 4 + 5 * (deg / max_deg)

        row = meta_idx.loc[node_idx] if node_idx in meta_idx.index else {}
        def g(attr, default="N/A"):
            if isinstance(row, dict): return row.get(attr, default)
            return getattr(row, attr, default)

        popup_html = (
            f"<b>Sensore PeMS #{g('ID')}</b><br>"
            f"Indice grafo : {node_idx}<br>"
            f"Autostrada   : {g('Fwy')} ({g('Direction')})<br>"
            f"Contea       : {g('County')} — D{g('District')}<br>"
            f"Grado        : {deg}<br>"
            f"Coord        : {lat:.5f}, {lon:.5f}<br>"
            + (f"<span style='color:{color}'>● Componente CC{cc_idx} "
               f"({stats['sizes'][cc_idx]} nodi)</span>"
               if n_cc > 1 else
               f"<span style='color:#5f5'>● Grafo connesso</span>")
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True, fill_color=color, fill_opacity=0.88, weight=0.6,
            tooltip=f"#{g('ID')} · {g('Fwy')} · grado {deg}"
                    + (f" · CC{cc_idx}" if n_cc > 1 else ""),
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(cc_groups[cc_idx])

    for grp in cc_groups.values():
        grp.add_to(m)

    # ── Heatmap densità ────────────────────────────────────────────────────
    heat_grp = folium.FeatureGroup(name="🌡️ Heatmap densità", show=False)
    HeatMap([[coords[n][0], coords[n][1]] for n in G.nodes() if n in coords],
            radius=18, blur=22, min_opacity=0.3).add_to(heat_grp)
    heat_grp.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ── Titolo + legenda ───────────────────────────────────────────────────
    if n_cc > 1:
        badges = "".join(
            f'<span style="background:{CC_PALETTE[i%len(CC_PALETTE)]};'
            f'color:#111;padding:1px 5px;border-radius:3px;margin:2px;'
            f'font-size:10px;display:inline-block;">'
            f'CC{i}: {stats["sizes"][i]}</span>'
            for i in range(min(n_cc, 10))
        )
        extra = f'<span style="font-size:10px;opacity:.7"> +{n_cc-10} altre</span>' if n_cc > 10 else ""
        cc_block = (
            f'<div style="margin-top:6px;font-size:11px;">'
            f'⚠️ <b>{n_cc} componenti sconnesse</b><br>'
            f'{badges}{extra}</div>'
        )
    else:
        cc_block = '<div style="margin-top:5px;font-size:11px;color:#5f5;">✅ Grafo connesso</div>'

    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;top:14px;left:50%;transform:translateX(-50%);
                z-index:9999;background:rgba(8,8,24,0.92);color:#e0f0ff;
                padding:10px 22px;border-radius:9px;
                font-family:'Courier New',monospace;font-size:13px;
                border:1px solid #00ffcc44;pointer-events:none;
                text-align:center;max-width:560px;">
      <b style="font-size:15px;color:#00ffcc;">PEMS08 — Traffic Sensor Graph</b><br>
      <span style="font-size:11px;opacity:.75;">
        170 sensori · San Francisco Bay Area · Calif. · Lug–Ago 2016
      </span><br>
      <span style="font-size:11px;">
        Archi: <span style="color:#00dd28;">■</span> breve &nbsp;→&nbsp;
               <span style="color:#dc3300;">■</span> lungo
      </span>
      {cc_block}
    </div>"""))

    return m


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Controlla file ──────────────────────────────────────────────────
    for fpath in [FILE_CA_META, FILE_EDGES]:
        if not os.path.exists(fpath):
            print(f"[!] File non trovato: {fpath}")
            sys.exit(1)

    # ── 2. Carica ca_meta e filtra i 170 sensori di PEMS08 ────────────────
    print("[1/5] Carico ca_meta.csv e filtro sensori PEMS08 …")
    meta_all = pd.read_csv(FILE_CA_META)

    required = {"ID", "Lat", "Lng", "ID2"}
    missing_cols = required - set(meta_all.columns)
    if missing_cols:
        print(f"  [!] Colonne mancanti in ca_meta.csv: {missing_cols}")
        sys.exit(1)

    meta_df = meta_all[
        meta_all["ID2"].between(PEMS08_ID2_OFFSET,
                                PEMS08_ID2_OFFSET + PEMS08_N_NODES - 1)
    ].copy()
    meta_df["graph_idx"] = (meta_df["ID2"] - PEMS08_ID2_OFFSET).astype(int)

    print(f"      {len(meta_df)} sensori PEMS08 trovati in ca_meta")
    print(f"      Districts: {dict(meta_df['District'].value_counts())}")
    print(f"      County top3: {dict(meta_df['County'].value_counts().head(3))}")

    coords = {
        int(row.graph_idx): (float(row.Lat), float(row.Lng))
        for row in meta_df.itertuples()
    }

    # ── 3. Carica archi ────────────────────────────────────────────────────
    print("[2/5] Carico archi e costruisco il grafo …")
    edges_df = pd.read_csv(FILE_EDGES)
    edges_df.columns = [c.lower().strip() for c in edges_df.columns]
    col_map = {}
    for c in edges_df.columns:
        if c in ("from", "src", "source"): col_map[c] = "from"
        if c in ("to",   "dst", "target"): col_map[c] = "to"
        if c in ("cost", "weight", "distance", "dist"): col_map[c] = "cost"
    edges_df = edges_df.rename(columns=col_map)

    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(int(row["from"]), int(row["to"]), weight=float(row["cost"]))

    print(f"      Nodi: {G.number_of_nodes()}  Archi: {G.number_of_edges()}")

    # ── 4. Analisi componenti connesse ─────────────────────────────────────
    print("[3/5] Analisi componenti connesse …")
    stats = analyze_components(G)
    print_component_report(G, stats, meta_df)

    # ── 5. Genera mappa ────────────────────────────────────────────────────
    print("[4/5] Genero la mappa interattiva …")
    m = build_map(G, coords, meta_df, stats)

    print("[5/5] Salvo la mappa …")
    m.save(OUTPUT_HTML)
    print(f"\n  ✅  Mappa salvata: {OUTPUT_HTML}")
    print(f"      Apri nel browser.\n")


if __name__ == "__main__":
    main()
