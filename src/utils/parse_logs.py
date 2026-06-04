import re
from pathlib import Path
import pandas as pd


# Cattura coppie "chiave: valore". La chiave è una parola (lettere/numeri/underscore),
# il valore è tutto fino al prossimo "qualcosa:" o a fine riga.
KV_PATTERN = re.compile(r'(\w+):\s*(\S+)')


def _convert(value: str):
    """Prova a convertire la stringa in int o float, altrimenti la lascia stringa."""
    # int
    try:
        return int(value)
    except ValueError:
        pass
    # float (gestisce anche notazione scientifica tipo 2.6e-05)
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_line(line: str) -> dict:
    """Trasforma una singola riga di log in un dict {colonna: valore}."""
    line = line.strip()
    if not line:
        return {}
    matches = KV_PATTERN.findall(line)
    return {key: _convert(val) for key, val in matches}


def parse_logs(source) -> pd.DataFrame:
    """
    `source` può essere:
      - un path (str o Path) a un file di log
      - una stringa multiriga di log
      - un iterabile di righe
    """
    # Path su disco
    if isinstance(source, (str, Path)) and Path(source).is_file():
        with open(source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    # Stringa multiriga
    elif isinstance(source, str):
        lines = source.splitlines()
    # Iterabile generico
    else:
        lines = list(source)

    rows = [parse_line(l) for l in lines]
    rows = [r for r in rows if r]  # scarta righe vuote
    df = pd.DataFrame(rows)

    # Mette 'Run' come prima colonna se presente, e ordina per Run
    if 'Run' in df.columns:
        df = df.sort_values('Run').reset_index(drop=True)
        cols = ['Run'] + [c for c in df.columns if c != 'Run']
        df = df[cols]

    return df


if __name__ == '__main__':
    LOG_FILE = "/home/user/Scrivania/PhD/EV/code/EV_GNN_repo_2b/EV_GNN/registry/logs/2026-05-05T17-28-04/log_total.txt"

    df = parse_logs(LOG_FILE)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df)

    # Salva anche in CSV accanto al file di log
    out_csv = Path(LOG_FILE).with_suffix('.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nCSV salvato in: {out_csv}")