from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1) Preprocessing dei metadati: PeMS meta -> location_summary.csv
# ---------------------------------------------------------------------------
def preprocess_metadata(
    meta_file: str,
    output_file: str,
    station_types: Optional[Iterable[str]] = ("ML",),
    default_max_speed: Optional[float] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Converte il file dei metadati PeMS nel formato ``location_summary.csv``.

    Il file di Chicago ha le colonne::

        id, street, length, start_latitude, start_longitude,
        end_latitude, end_longitude, max_speed

    Il file PeMS ``d07_text_meta_*.txt`` e' tab-separated e ha le colonne::

        ID, Fwy, Dir, District, County, City, State_PM, Abs_PM,
        Latitude, Longitude, Length, Type, Lanes, Name, User_ID_1..4

    Mappatura applicata:

    ===================  ==========================================
    Colonna Chicago      Origine PeMS
    ===================  ==========================================
    id                   ID
    street               "{Fwy}-{Dir}"  (es. "5-N")
    length               Length
    start_latitude       Latitude
    start_longitude      Longitude
    end_latitude         Latitude   (vedi NOTA *)
    end_longitude        Longitude  (vedi NOTA *)
    max_speed            non disponibile (vedi NOTA **)
    ===================  ==========================================

    NOTA (*) — coordinate di fine segmento
        Il dataset di Chicago descrive **segmenti** stradali, ognuno con un
        punto di inizio e uno di fine (start/end). I metadati PeMS, invece,
        descrivono **stazioni puntuali** di rilevamento: per ogni stazione c'e'
        una sola coppia (Latitude, Longitude). Non esiste quindi un punto di
        "fine" nei dati grezzi. Qui ``end_*`` viene posto uguale a ``start_*``
        (segmento degenere / punto). Se la pipeline a valle costruisce il grafo
        collegando archi che condividono nodi (vedi
        ``utils.build_edges_with_node_ids_chicago``), questa scelta va rivista:
        servira' una logica ad hoc per LA (p.es. collegare stazioni consecutive
        lungo la stessa autostrada/direzione ordinandole per ``Abs_PM``).

    NOTA (**) — velocita' massima
        I metadati PeMS **non contengono il limite di velocita'**. La colonna
        ``max_speed`` viene riempita con ``default_max_speed`` (default
        ``None`` -> NaN). Se serve un valore reale va recuperato da una fonte
        esterna (es. shapefile/dati Caltrans).

    Parameters
    ----------
    meta_file : path
        File dei metadati PeMS di input (tab-separated).
    output_file : path
        Dove salvare il ``location_summary.csv`` risultante.
    station_types : iterable di str oppure None
        Filtra le stazioni per ``Type``. Default ``("ML",)`` = solo mainline,
        che sono le stazioni tipicamente usate per i grafi di traffico
        (gli altri tipi sono rampe e simili: OR, FR, HV, FF, CD).
        Passare ``None`` per tenere tutte le stazioni.
    default_max_speed : float oppure None
        Valore con cui riempire la colonna ``max_speed`` (non presente nei
        metadati PeMS). Default ``None`` -> NaN.
    save : bool
        Se True salva il CSV su ``output_file``.

    Returns
    -------
    pandas.DataFrame
        Il dataframe con lo schema di Chicago.
    """
    meta_file = Path(meta_file)
    output_file = Path(output_file)

    if not meta_file.exists():
        raise FileNotFoundError(f"File dei metadati PeMS non trovato: {meta_file}")

    # Il file e' tab-separated; alcune colonne (Length, User_ID_*) possono essere vuote.
    raw = pd.read_csv(meta_file, sep="\t", dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]

    print(f"[preprocess_metadata] Lette {len(raw)} stazioni da {meta_file.name}")

    # --- Filtro opzionale per tipo di stazione -----------------------------
    if station_types is not None:
        station_types = list(station_types)
        before = len(raw)
        raw = raw[raw["Type"].isin(station_types)].copy()
        print(
            f"[preprocess_metadata] Filtro Type in {station_types}: "
            f"{before} -> {len(raw)} stazioni"
        )

    # --- Conversioni numeriche --------------------------------------------
    def to_float(series: pd.Series) -> pd.Series:
        # Stringhe vuote -> NaN
        return pd.to_numeric(series.replace("", np.nan), errors="coerce")

    lat = to_float(raw["Latitude"])
    lon = to_float(raw["Longitude"])
    length = to_float(raw["Length"])

    # --- Costruzione del dataframe in formato Chicago ----------------------
    out = pd.DataFrame(
        {
            "id": pd.to_numeric(raw["ID"], errors="coerce").astype("Int64"),
            # "street": combino numero autostrada e direzione, es. "5-N"
            "street": raw["Fwy"].str.strip() + "-" + raw["Dir"].str.strip(),
            "length": length,
            "start_latitude": lat,
            "start_longitude": lon,
            # NOTA (*): nei dati PeMS esiste un solo punto per stazione.
            "end_latitude": lat,
            "end_longitude": lon,
            # NOTA (**): max_speed non e' nei metadati PeMS.
            "max_speed": default_max_speed,
        }
    )

    out = out[CHICAGO_COLUMNS]  # garantisco ordine identico a Chicago

    # --- Diagnostica / avvisi ---------------------------------------------
    n_missing_coord = int(out[["start_latitude", "start_longitude"]].isna().any(axis=1).sum())
    n_missing_len = int(out["length"].isna().sum())
    if n_missing_coord:
        print(f"[preprocess_metadata] ATTENZIONE: {n_missing_coord} stazioni senza coordinate (NaN).")
    if n_missing_len:
        print(f"[preprocess_metadata] ATTENZIONE: {n_missing_len} stazioni senza Length (NaN).")
    print(
        "[preprocess_metadata] NOTA: 'end_latitude'/'end_longitude' = punto della stazione "
        "(i metadati PeMS non hanno punti di fine segmento)."
    )
    print(
        "[preprocess_metadata] NOTA: 'max_speed' assente nei metadati PeMS -> "
        f"riempito con {default_max_speed!r}."
    )

    if save:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_file, index=False)
        print(f"[preprocess_metadata] Salvato: {output_file}  ({len(out)} righe)")

    return out


# ---------------------------------------------------------------------------
# 2) Preprocessing delle misurazioni: file giornalieri -> un CSV per stazione
# ---------------------------------------------------------------------------
def _day_files_sorted(input_dir: Path) -> list[Path]:
    """Ritorna i file giornalieri ``*.txt`` ordinati per data (dal nome file).

    Vengono ignorati i ``.gz`` (copie compresse dello stesso contenuto).
    Il nome ha forma ``d07_text_station_5min_YYYY_MM_DD.txt``.
    """
    files = [Path(p) for p in glob.glob(os.path.join(str(input_dir), "*.txt"))]

    def date_key(p: Path):
        m = re.search(r"(\d{4})_(\d{2})_(\d{2})", p.name)
        return tuple(int(g) for g in m.groups()) if m else (0, 0, 0)

    return sorted(files, key=date_key)


def preprocess_measurements(
    input_dir: str,
    output_dir: str,
    station_types: Optional[Iterable[str]] = ("ML",),
    feature_columns: Iterable[str] = ("avg_speed", "total_flow", "avg_occupancy", "samples", "observed"),
    sites: Optional[Iterable[int]] = None,
    max_days: Optional[int] = None,
    clean_output: bool = True,
) -> None:
    """Riorganizza le misurazioni 5-min PeMS da "un file per giorno" a "un file per stazione".

    Input
    -----
    File giornalieri ``d07_text_station_5min_YYYY_MM_DD.txt`` (tab/CSV PeMS,
    senza header): ogni file contiene le misurazioni di **tutte** le stazioni
    per le 288 finestre di 5 minuti di una giornata.

    Output
    ------
    In ``output_dir`` viene scritto un ``<station_id>.csv`` per stazione, con una
    riga per istante temporale, **ordinata per data** su tutti i giorni
    elaborati. Le colonne sono ``[*feature_columns, "time"]`` (sul modello dei
    file per-stazione di Chicago/New York).

    I file vengono processati un giorno alla volta e le righe vengono
    **accodate** ai CSV di stazione: in questo modo non si tiene mai in memoria
    piu' di un giorno per volta (i file grezzi sono molto grandi, ~176 MB l'uno).

    Parameters
    ----------
    input_dir : path
        Cartella con i file giornalieri ``*.txt``.
    output_dir : path
        Cartella di output (creata se assente).
    station_types : iterable di str oppure None
        Filtra per tipo di stazione (col ``lane_type``). Default ``("ML",)``
        (solo mainline, coerente con :func:`preprocess_metadata`).
        ``None`` = tutte.
    feature_columns : iterable di str
        Quali feature scrivere, scelte tra le chiavi di
        :data:`PEMS_5MIN_COLUMNS` (escluse ``timestamp``/``station`` che sono
        gestite a parte). L'ordine e' rispettato nel CSV.
    sites : iterable di int oppure None
        Se fornito, elabora solo questi ID di stazione.
    max_days : int oppure None
        Se fornito, limita il numero di giorni elaborati (utile per test).
    clean_output : bool
        Se True svuota i ``*.csv`` esistenti in ``output_dir`` prima di iniziare,
        evitando di accodare a dati di run precedenti.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    feature_columns = list(feature_columns)

    # Validazione delle feature richieste
    unknown = [c for c in feature_columns if c not in PEMS_5MIN_COLUMNS]
    if unknown:
        raise ValueError(f"feature_columns sconosciute: {unknown}. Valide: {list(PEMS_5MIN_COLUMNS)}")

    day_files = _day_files_sorted(input_dir)
    if max_days is not None:
        day_files = day_files[:max_days]
    if not day_files:
        raise FileNotFoundError(f"Nessun file giornaliero *.txt trovato in {input_dir}")

    print(f"[preprocess_measurements] {len(day_files)} giorni da elaborare (input: {input_dir.name})")

    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        removed = 0
        for old in output_dir.glob("*.csv"):
            old.unlink()
            removed += 1
        if removed:
            print(f"[preprocess_measurements] Rimossi {removed} CSV preesistenti da {output_dir}")

    # Colonne da leggere (per indice) e relativi nomi
    needed = {"timestamp": PEMS_5MIN_COLUMNS["timestamp"],
              "station": PEMS_5MIN_COLUMNS["station"]}
    if station_types is not None:
        needed["lane_type"] = PEMS_5MIN_COLUMNS["lane_type"]
    for c in feature_columns:
        needed[c] = PEMS_5MIN_COLUMNS[c]

    # IMPORTANTE: con usecols pandas restituisce le colonne in ordine crescente
    # di indice nel file, quindi 'names' va allineato a quell'ordine.
    items = sorted(needed.items(), key=lambda kv: kv[1])
    names = [name for name, _ in items]
    usecols = [idx for _, idx in items]
    numeric_cols = [c for c in feature_columns]  # tutte le feature sono numeriche

    sites = set(int(s) for s in sites) if sites is not None else None
    station_types = list(station_types) if station_types is not None else None

    written_header: set[int] = set()  # stazioni a cui ho gia' scritto l'header in questo run
    total_rows = 0

    for di, day_file in enumerate(day_files, start=1):
        print(f"[preprocess_measurements] ({di}/{len(day_files)}) {day_file.name} ...", flush=True)

        df = pd.read_csv(
            day_file,
            header=None,
            usecols=usecols,
            names=names,
            dtype=str,
            keep_default_na=False,
            engine="c",
        )

        # Filtri
        if station_types is not None:
            df = df[df["lane_type"].isin(station_types)]
            df = df.drop(columns=["lane_type"])
        df["station"] = pd.to_numeric(df["station"], errors="coerce").astype("Int64")
        if sites is not None:
            df = df[df["station"].isin(sites)]
        if df.empty:
            continue

        # Conversioni
        df["time"] = pd.to_datetime(df["timestamp"], format=PEMS_TS_FORMAT, errors="coerce")
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")

        # Ordino per stazione e tempo: nel groupby le righe restano in ordine cronologico
        df = df.sort_values(["station", "time"])

        out_cols = feature_columns + ["time"]
        for station_id, g in df.groupby("station", sort=True):
            sid = int(station_id)
            path = output_dir / f"{sid}.csv"
            first = sid not in written_header
            g[out_cols].to_csv(
                path,
                index=False,
                mode="w" if first else "a",
                header=first,
            )
            written_header.add(sid)
            total_rows += len(g)

    print(
        f"[preprocess_measurements] Completato: {len(written_header)} stazioni, "
        f"{total_rows} righe totali scritte in {output_dir}"
    )


if __name__ == "__main__":
    """
    los_angeles_preprocessing.py
    ============================

    Raccolta di funzioni di preprocessing per il nuovo dataset "losangeles"
    (dati PeMS, District 7 della California).

    Ogni azione di preprocessing e' definita da una o piu' funzioni, e in fondo al
    file c'e' un classico blocco ``if __name__ == "__main__"`` che, in funzione dei
    parametri definiti li', permette di eseguire selettivamente quello che si vuole.

    Funzioni disponibili
    --------------------
    - :func:`preprocess_metadata`
        Trasforma il file dei metadati delle stazioni PeMS
        ``d07_text_meta_2023_12_22.txt`` in un ``location_summary.csv`` con la
        stessa identica struttura del file usato per Chicago
        (``data/raw/chicago/traffic/location_summary.csv``).
    """
    # ---------------------------------------------------------------------------
    # Percorsi di default (relativi alla root del repo EV_GNN)
    # ---------------------------------------------------------------------------
    # Questo file si trova in:  EV_GNN/src/dataset/losangeles/los_angeles_preprocessing.py
    # quindi la root del progetto e' 3 livelli sopra.
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    LA_TRAFFIC_DIR = PROJECT_ROOT / "data" / "raw" / "losangeles" / "traffic" / "Traffic stations"

    DEFAULT_META_FILE = LA_TRAFFIC_DIR / "d07_text_meta_2023_12_22.txt"
    DEFAULT_OUTPUT_FILE = LA_TRAFFIC_DIR / "location_summary.csv"

    # Cartella con i file giornalieri PeMS "station_5min" (un file = un giorno, tutte le stazioni)
    DEFAULT_MEASUREMENTS_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "losangeles" / "traffic" / "Stations 5 minutes traffics"
    # Cartella di output: un .csv per stazione (riga = istante temporale)
    DEFAULT_MEASUREMENTS_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "losangeles" / "traffic" / "traffic_data"

    # Struttura (0-based) del file PeMS "d07_text_station_5min_*.txt".
    # Le prime 12 colonne sono a livello di stazione; seguono i dati per-corsia.
    PEMS_5MIN_COLUMNS = {
        "timestamp": 0,  # data/ora del campione (5 min)
        "station": 1,  # ID stazione
        "district": 2,
        "freeway": 3,
        "direction": 4,
        "lane_type": 5,  # ML, OR, FR, HV, FF, CD
        "station_length": 6,
        "samples": 7,  # n. campioni ricevuti
        "observed": 8,  # % di osservato (vs imputato)
        "total_flow": 9,  # flusso totale (veicoli/5min)
        "avg_occupancy": 10,  # occupazione media
        "avg_speed": 11,  # velocita' media (mph)
    }

    # Formato del timestamp nei file PeMS, es. "05/21/2026 00:00:00"
    PEMS_TS_FORMAT = "%m/%d/%Y %H:%M:%S"

    # Schema (ordine delle colonne) del location_summary.csv di Chicago, da replicare.
    CHICAGO_COLUMNS = ["id",
                       "street",
                       "length",
                       "start_latitude",
                       "start_longitude",
                       "end_latitude",
                       "end_longitude",
                       "max_speed"]

    # ---- Parametri / interruttori delle azioni da eseguire ----------------
    RUN_PREPROCESS_METADATA = False
    RUN_PREPROCESS_MEASUREMENTS = True

    # Tipo di stazione comune alle due azioni (coerenza metadati <-> misurazioni)
    STATION_TYPES = ("ML",)      # None = tutte le stazioni; ("ML",) = solo mainline

    # --- Azione 1: metadati -> location_summary.csv ------------------------
    if RUN_PREPROCESS_METADATA:
        df = preprocess_metadata(
            meta_file=DEFAULT_META_FILE,
            output_file=DEFAULT_OUTPUT_FILE,
            station_types=STATION_TYPES,
            default_max_speed=None,   # nessun limite di velocita' nei metadati PeMS
            save=True,
        )
        print(df.head())

    # --- Azione 2: misurazioni 5-min -> un CSV per stazione ----------------
    if RUN_PREPROCESS_MEASUREMENTS:
        preprocess_measurements(
            input_dir=DEFAULT_MEASUREMENTS_INPUT_DIR,
            output_dir=DEFAULT_MEASUREMENTS_OUTPUT_DIR,
            station_types=STATION_TYPES,
            feature_columns=("avg_speed", "total_flow", "avg_occupancy", "samples", "observed"),
            sites=None,        # es. [715898, 715915] per elaborare solo alcune stazioni
            max_days=None,     # es. 2 per un test rapido sui primi giorni
            clean_output=True,
        )
