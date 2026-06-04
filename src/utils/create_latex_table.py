from pathlib import Path
import pandas as pd


def select_best_rows(df: pd.DataFrame, metric_col: str, mode: str) -> pd.DataFrame:
    """Per ogni modello restituisce la riga con metric_col massima/minima."""
    if mode == "max":
        idx = df.groupby("model")[metric_col].idxmax()
    elif mode == "min":
        idx = df.groupby("model")[metric_col].idxmin()
    else:
        raise ValueError("METRIC_MODE deve essere 'max' o 'min'")
    return df.loc[idx].reset_index(drop=True)


def fmt(mean: float, std: float, decimals: int) -> str:
    """Formatta 'mean ± std' per LaTeX."""
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def build_latex(best: pd.DataFrame, split: str, decimals: int) -> str:
    """Costruisce la tabella LaTeX con colonne MSE, RMSE, MAE."""
    rows = []
    for _, r in best.iterrows():
        rows.append({
            "Model": r["model"],
            "MSE":   fmt(r[f"{split}_mse_mean"],  r[f"{split}_mse_std"],  decimals),
            "RMSE":  fmt(r[f"{split}_rmse_mean"], r[f"{split}_rmse_std"], decimals),
            "MAE":   fmt(r[f"{split}_mae_mean"],  r[f"{split}_mae_std"],  decimals),
        })
    table = pd.DataFrame(rows)

    # to_latex con escape=False per non rompere $...$ e \pm
    latex = table.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(table.columns) - 1),
        caption=f"Risultati su {split} set: per ogni modello la run con "
                f"{split}\\_mse\\_mean {'massimo' if METRIC_MODE == 'max' else 'minimo'}.",
        label=f"tab:{split}_{METRIC_MODE}_per_model",
    )
    return latex


def main():
    df = pd.read_csv(CSV_FILE)

    metric_col = f"{SPLIT}_mse_mean"
    best = select_best_rows(df, metric_col, METRIC_MODE)

    # Ordina i modelli alfabeticamente (puoi cambiare in un ordine fisso se preferisci)
    best = best.sort_values("model").reset_index(drop=True)

    # Applica il fattore moltiplicativo a mean e std delle metriche usate in tabella
    metric_cols = [
        f"{SPLIT}_mse_mean", f"{SPLIT}_mse_std",
        f"{SPLIT}_rmse_mean", f"{SPLIT}_rmse_std",
        f"{SPLIT}_mae_mean", f"{SPLIT}_mae_std",
    ]
    if SCALE_FACTOR != 1:
        best[metric_cols] = best[metric_cols] * SCALE_FACTOR
        print(f"Valori di mean e std moltiplicati per {SCALE_FACTOR}\n")

    print(f"Righe selezionate ({METRIC_MODE} di {metric_col}) per ogni modello:\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(best[["Run", "model", "dataset_name", "emb_dim", "dropout", "batch_size",
                *metric_cols]])

    latex = build_latex(best, SPLIT, DECIMALS)

    out_path = Path(__file__).parent / OUTPUT_TEX
    out_path.write_text(latex, encoding="utf-8")

    print("\n" + "=" * 60)
    print("TABELLA LATEX:")
    print("=" * 60)
    print(latex)
    print(f"Salvata in: {out_path}")


if __name__ == "__main__":
    """
    Per ogni modello estrae la riga con test_mse_mean massimo (o minimo, vedi METRIC_MODE)
    e produce una tabella LaTeX con MSE, RMSE, MAE nel formato mean ± std.
    """

    # === CONFIGURAZIONE ===
    CSV_FILE = "/home/user/Scrivania/PhD/EV/code/EV_GNN_repo_2b/EV_GNN/registry/logs/2026-05-05T17-28-04/log_total.csv"
    METRIC_MODE = "min"  # "max" o "min" — la riga selezionata per modello
    OUTPUT_TEX = "best_per_model.tex"
    SPLIT = "test"  # "test" o "val"
    DECIMALS = 2  # cifre decimali nella tabella
    SCALE_FACTOR = 1e4  # fattore moltiplicativo applicato a mean e std (es. 100, 1000)
    # ======================

    main()