# inference.py
import argparse
import random
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

from src.config import Parameters
from src.dataset.dataset import get_datamodule
from src.utils.utils import get_model

# helper to convert flattened (B*N, T*F) -> (B, N, T, F)
def flat_to_BNTF(flat_tensor, N, T):
    """
    flat_tensor: Tensor with shape (B*N, T*F) or already 4D like (B, N, T, F) or (B, T, N, F)
    N: num_nodes
    T: prediction horizon
    returns: tensor (B, N, T, F)
    """
    if flat_tensor is None:
        return None
    if flat_tensor.dim() == 4:
        # already (B, T, N, F) or (B, N, T, F) — normalize to (B, N, T, F)
        B, A, B2, F = flat_tensor.shape
        # if axis 1 is T, permute to (B, N, T, F)
        if A == T:
            return flat_tensor.permute(0, 2, 1, 3).contiguous()
        return flat_tensor
    # expected shape (B*N, T*F)
    b_n, t_times_f = flat_tensor.shape
    B = int(b_n // N)
    F = int(t_times_f // T)
    return flat_tensor.view(B, N, T, F).contiguous()

def plot_sample_multifeature(history_BN_lf, target_BNTF, pred_BNTF, b_idx, n_idx, run_params, sample_id=None, out_file=None):
    """
    Disegna, in una nuova finestra, un subplot per ciascuna feature:
      - passato (linea continua) sui timestep [0 .. lags-1]
      - futuro target (linea tratteggiata) su [lags .. lags+T-1]
      - futuro predetto (linea continua) su [lags .. lags+T-1]
      - linea verticale nera a x = lags - 0.5
    """
    assert target_BNTF is not None and pred_BNTF is not None, "target/pred non possono essere None"
    B, N, T, F = pred_BNTF.shape
    lags = history_BN_lf.shape[2] if history_BN_lf is not None else 0

    fig_title = f"Sample {sample_id if sample_id is not None else ''}, node {n_idx}"
    fig, axes = plt.subplots(F, 1, figsize=(10, max(3, 2 * F)), sharex=True)
    if F == 1:
        axes = [axes]

    x_past = np.arange(0, lags)
    x_future = np.arange(lags, lags + T)

    # nomi canali/feature (se disponibili)
    try:
        columns_name = (getattr(run_params, "traffic_columns_to_use", []) or []) + \
                       (getattr(run_params, "ev_columns_to_use", []) or [])
        if len(columns_name) != F:
            columns_name = [f"ch{f}" for f in range(F)]
    except Exception:
        columns_name = [f"ch{f}" for f in range(F)]

    for f in range(F):
        # denormalizzazione: target e predetto
        min_val = run_params.min_vals_normalization[f]
        max_val = run_params.max_vals_normalization[f]

        ax = axes[f]
        past = history_BN_lf[b_idx, n_idx, :, f]  # (lags,)
        past_real = past * (max_val - min_val) + min_val
        ax.plot(x_past, past_real, linewidth=1.5, label="past")

        targ = target_BNTF[b_idx, n_idx, :, f]
        pred = pred_BNTF[b_idx, n_idx, :, f]

        targ_real = targ * (max_val - min_val) + min_val
        pred_real = pred * (max_val - min_val) + min_val

        ax.plot(x_future, targ_real, linestyle='--', linewidth=1.5, label="target")
        ax.plot(x_future, pred_real, linestyle='-', linewidth=1.5, label="pred")

        ax.axvline(x=lags - 0.5, color='black', linewidth=1.0)
        ax.set_title(f" {columns_name[f]}", loc='left')
        ax.set_ylabel("value")
        ax.set_xlim(-0.5, lags + T - 0.5)
        ymin = float(min(past_real.min().item(), targ_real.min().item(), pred_real.min().item()))
        ymax = float(max(past_real.max().item(), targ_real.max().item(), pred_real.max().item()))
        ax.set_ylim(ymin, ymax)

        if f == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("time step (relative)")
    fig.suptitle(fig_title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_file:
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        plt.savefig(out_file, dpi=120)
        print(f"Saved plot to {out_file}")
        plt.close(fig)
    else:
        plt.show()

def export_inputs_csv(x_hist_cpu, csv_dir, start_time_str, step_minutes, run_params):
    """
    Salva un CSV per ciascun canale/feature per i dati di input.
    - righe = tutti i timestep concatenati lungo il test set
    - colonne = [timestamp, site_1, ..., site_N]
    """
    os.makedirs(csv_dir, exist_ok=True)
    B, N, lags, F = x_hist_cpu.shape  # (B, N, lags, F)

    # timeline
    total_steps = B * lags
    start_time = pd.to_datetime(start_time_str)

    time_index = pd.date_range(start=start_time, periods=total_steps, freq=pd.Timedelta(minutes=step_minutes))

    # per ogni canale, costruisci (total_steps, N)
    for f in range(F):
        arr = x_hist_cpu[:, :, :, f].cpu().numpy()  # (B, N, lags)
        arr = np.transpose(arr, (0, 2, 1))         # (B, lags, N)
        arr = arr.reshape(-1, N)                   # (B*lags, N)

        # DataFrame
        col_names = ["timestamp"] + [f"site_{i+1}" for i in range(N)]
        df = pd.DataFrame(arr, columns=[f"site_{i+1}" for i in range(N)])
        df.insert(0, "timestamp", time_index)

        # salva
        safe_feat = f"input_feature_{f}"  # opzionale: usa un nome più esplicito se lo hai
        out_path = os.path.join(csv_dir, f"inputs_{safe_feat}.csv")
        df.to_csv(out_path, index=False)
        print(f"[CSV] Saved {out_path} with shape {df.shape}")


def export_predictions_csv(preds_BNTF, csv_dir, start_time_str, step_minutes, run_params, data_type='predictions'):
    """
    Salva un CSV per ciascun canale/feature.
    - righe = tutti i timestep concatenati lungo il test set (ordine dei batch preservato)
    - colonne = [timestamp, site_1, ..., site_N]
    - timestamp parte da start_time_str e avanza di step_minutes
    """
    os.makedirs(csv_dir, exist_ok=True)
    B, N, T, F = preds_BNTF.shape

    # nomi feature per i file
    try:
        feature_names = (getattr(run_params, "traffic_columns_to_use", []) or []) + \
                        (getattr(run_params, "ev_columns_to_use", []) or [])
        if len(feature_names) != F:
            feature_names = [f"ch{f}" for f in range(F)]
    except Exception:
        feature_names = [f"ch{f}" for f in range(F)]

    # timeline
    total_steps = B * T
    start_time = pd.to_datetime(start_time_str)
    time_index = pd.date_range(start=start_time, periods=total_steps, freq=pd.Timedelta(minutes=step_minutes))

    # per ogni canale, costruisci (total_steps, N)
    for f in range(F):
        # shape (B, N, T) -> (B*T, N) preservando l'ordine (batch 0, tutti i T; batch 1, tutti i T; ...)
        # prendiamo i valori così come usciti dal modello (già nella scala normalizzata)
        # se vuoi esportare denormalizzato, abilita la sezione DN più sotto
        arr = preds_BNTF[:, :, :, f].cpu().numpy()  # (B, N, T)
        arr = np.transpose(arr, (0, 2, 1))         # (B, T, N)
        arr = arr.reshape(-1, N)                   # (B*T, N)

        # [OPZIONALE] Denormalizzazione (decommenta se vuoi valori reali)
        # min_val = run_params.min_vals_normalization[f]
        # max_val = run_params.max_vals_normalization[f]
        # arr = arr * (max_val - min_val) + min_val

        # DataFrame
        col_names = ["timestamp"] + [f"site_{i+1}" for i in range(N)]
        df = pd.DataFrame(arr, columns=[f"site_{i+1}" for i in range(N)])
        df.insert(0, "timestamp", time_index)

        # salva
        safe_feat = "".join([c if c.isalnum() or c in "-_." else "_" for c in feature_names[f]])
        out_path = os.path.join(csv_dir, f"{data_type}_{safe_feat}.csv")
        df.to_csv(out_path, index=False)
        print(f"[CSV] Saved {out_path} with shape {df.shape}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='newyork', help="['denmark', 'metr_la', 'newyork', 'chicago']")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", type=str, default='GraphWavenet')
    parser.add_argument("--checkpoint", type=str, default='../checkpoints/ckpt new york/last-v4.ckpt', help="Path to checkpoint file (.ckpt or .pt)")
    parser.add_argument("--output_path", type=str, default='./inference_outputs', help="folder to save preds/targets")
    parser.add_argument("--show", action='store_true', help="Show sample plots (one window per sample)")
    parser.add_argument("--num_samples_to_show", type=int, default=3, help="Numero di sample da mostrare (ogni sample in una nuova finestra)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true", help="Attiva output dettagliato")

    # nuovi argomenti per i CSV finali
    parser.add_argument("--export_csv", default=True, action="store_true", help="Esporta i CSV concatenati per canale")
    # parser.add_argument("--csv_dir", type=str, default="./csv_out", help="cartella per i CSV")
    # parser.add_argument("--start_time", type=str, default="2025-07-17 21:08:00", help="timestamp iniziale del test set")
    # parser.add_argument("--step_minutes", type=float, default=5.0, help="risoluzione temporale (minuti) tra predizioni consecutive")
    args = parser.parse_args()

    # self.num_station = dataset.number_of_station
    # self.run_params.num_nodes = self.num_station
    # self.run_params.traffic_features = dataset.traffic_features
    # self.run_params.ev_features = dataset.ev_features
    # self.run_params.traffic_features_names = dataset.traffic_columns_used_in_data
    # self.run_params.ev_features_names = dataset.ev_columns_used_in_data
    # self.run_params.min_vals_normalization = dataset.min_vals_normalization
    # self.run_params.max_vals_normalization = dataset.max_vals_normalization
    # self.run_params.start_time = dataset.start_time
    # self.run_params.end_time = dataset.end_time
    # self.run_params.traffic_resolution = dataset.traffic_resolution
    # self.run_params.ev_resolution = dataset.ev_resolution

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Parameters and datamodule
    run_params = Parameters(args)
    # opzionale: se questo campo è usato da te per altro, rimuovi/ignora
    run_params.time_series_step = 24

    dataModuleInstance, run_params = get_datamodule(run_params)

    # create dataloaders
    dataModuleInstance.setup(stage='test')
    test_loader = dataModuleInstance.test_dataloader()

    # instantiate model
    model = get_model(run_params)
    device = torch.device('cuda' if torch.cuda.is_available() and getattr(run_params, "accelerator", None) in ['gpu', 'cuda'] else 'cpu')
    model.to(device)
    model.eval()
    save_predictions = False

    # load checkpoint if provided
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location=device)
            state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            model.load_state_dict(state, strict=False)
            print("Checkpoint loaded.")
        else:
            print(f"Checkpoint file not found: {args.checkpoint}. Proceeding with random init.")

    # iterate test set and collect predictions
    preds_list, targets_list, history_list = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.verbose:
                print(f"Test Batch {batch_idx}")

            # forward
            batch.to(device)
            y_pred = model.forward(batch)  # shape (B*N, T*F) o 4D a seconda del modello
            y_pred_cpu = y_pred.detach().cpu()

            # target e history
            y_true = getattr(batch, 'y', None)
            y_true_cpu = y_true.detach().cpu()

            x_hist = getattr(batch, 'x', None)
            x_hist_cpu = x_hist.detach().cpu()

            # converti a (B, N, T, F) / (B, N, lags, F)
            pred_BNTF = flat_to_BNTF(y_pred_cpu, run_params.num_nodes, run_params.prediction_window)
            targ_BNTF = flat_to_BNTF(y_true_cpu, run_params.num_nodes, run_params.prediction_window)
            history_BNlf = torch.reshape(x_hist_cpu, (args.batch_size,run_params.num_nodes,x_hist_cpu.shape[1], x_hist_cpu.shape[2]))

            preds_list.append(pred_BNTF)
            targets_list.append(targ_BNTF)
            history_list.append(history_BNlf)

    preds_all = torch.cat(preds_list, dim=0) if preds_list else None
    targets_all = torch.cat(targets_list, dim=0) if targets_list else None
    history_all = torch.cat(history_list, dim=0) if history_list else None

    os.makedirs(args.output_path, exist_ok=True)

    # opzionale: salva tensor
    if save_predictions and preds_all is not None:
        preds_file = os.path.join(args.output_path, 'preds_BNTF.pt')
        torch.save(preds_all, preds_file)
        print(f"Saved predictions to {preds_file} with shape {tuple(preds_all.shape)}")
        if targets_all is not None:
            targets_file = os.path.join(args.output_path, 'targets_BNTF.pt')
            torch.save(targets_all, targets_file)
            print(f"Saved targets to {targets_file} with shape {tuple(targets_all.shape)}")
        if history_all is not None:
            history_file = os.path.join(args.output_path, 'history_BNlf.pt')
            torch.save(history_all, history_file)
            print(f"Saved history to {history_file} with shape {tuple(history_all.shape)}")

    # ESPORTA CSV concatenati (uno per canale)
    if args.export_csv and preds_all is not None:
        os.makedirs(run_params.csv_out_dir, exist_ok=True)
        export_inputs_csv(
            x_hist_cpu=history_all,
            csv_dir=run_params.csv_out_dir,
            start_time_str=run_params.start_time_test,  # es. "2025-07-17 21:08:00"
            step_minutes=float(run_params.traffic_resolution),  # es. 5.0
            run_params=run_params
        )

        export_predictions_csv(
            preds_BNTF=preds_all,
            csv_dir=run_params.csv_out_dir,
            start_time_str=run_params.start_time_test + timedelta(minutes=run_params.lags * run_params.traffic_resolution),        # es. "2025-07-17 21:08:00"
            step_minutes=float(run_params.traffic_resolution), # es. 5.0
            run_params=run_params,
            data_type='predictions')

        export_predictions_csv(
            preds_BNTF=targets_all,
            csv_dir=run_params.csv_out_dir,
            start_time_str=run_params.start_time_test + timedelta(minutes=run_params.lags * run_params.traffic_resolution),        # es. "2025-07-17 21:08:00"
            step_minutes=float(run_params.traffic_resolution), # es. 5.0
            run_params=run_params,
            data_type='targets')

    # mostra più sample (una finestra per sample)
    if args.show and preds_all is not None and targets_all is not None and history_all is not None:
        B, N, T, F = preds_all.shape
        ns = min(args.num_samples_to_show, B)
        for s in range(ns):
            b_idx = random.randint(0, B - 1)
            n_idx = random.randint(0, N - 1)
            print(f"[{s+1}/{ns}] Plotting sample — sample {b_idx}, node {n_idx}")
            plot_sample_multifeature(history_all, targets_all, preds_all, b_idx, n_idx, run_params, sample_id=s+1, out_file=None)

if __name__ == "__main__":
    main()
