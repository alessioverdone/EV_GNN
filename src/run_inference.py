import argparse
import json
import random
import os
import torch
import numpy as np
import matplotlib

matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta, datetime

from src.config import Parameters
from src.dataset.dataset import get_datamodule
from src.utils.utils import get_model


def save_predictions(history_all,
                     targets_all,
                     preds_all,
                     run_params):
    """
    Function that saves predictions to a csv file

    :param history_all:
    :param targets_all:
    :param preds_all:
    :param run_params:
    :return:
    """
    os.makedirs(run_params.output_path, exist_ok=True)
    os.makedirs(run_params.csv_out_path, exist_ok=True)

    if preds_all == None or targets_all == None or history_all == None:
        raise ValueError("preds_all, targets_all or history_all is None")

    preds_file = os.path.join(run_params.output_path, 'preds_BNTF.pt')
    targets_file = os.path.join(run_params.output_path, 'targets_BNTF.pt')
    history_file = os.path.join(run_params.output_path, 'history_BNlf.pt')

    torch.save(preds_all, preds_file)
    torch.save(targets_all, targets_file)
    torch.save(history_all, history_file)

    print(f"Saved predictions to {preds_file} with shape {tuple(preds_all.shape)}")
    print(f"Saved targets to {targets_file} with shape {tuple(targets_all.shape)}")
    print(f"Saved history to {history_file} with shape {tuple(history_all.shape)}")

    # ESPORTA CSV concatenati (uno per canale)
    if run_params.export_csv:
        export_inputs_csv(
            x_hist_cpu=history_all,
            csv_dir=run_params.csv_out_path,
            start_time_str=run_params.start_time_test,  # es. "2025-07-17 21:08:00"
            step_minutes=float(run_params.traffic_resolution),  # es. 5.0
            run_params=run_params)

        export_predictions_csv(
            preds_BNTF=preds_all,
            csv_dir=run_params.csv_out_path,
            start_time_str=(datetime.fromisoformat(run_params.start_time_test) + timedelta(
                minutes=run_params.lags * run_params.traffic_resolution)).isoformat(),  # es. "2025-07-17 21:08:00"
            step_minutes=float(run_params.traffic_resolution),  # es. 5.0
            run_params=run_params,
            data_type='predictions')

        export_predictions_csv(
            preds_BNTF=targets_all,
            csv_dir=run_params.csv_out_path,
            start_time_str=(datetime.fromisoformat(run_params.start_time_test) + timedelta(
                minutes=run_params.lags * run_params.traffic_resolution)).isoformat(),  # es. "2025-07-17 21:08:00"
            step_minutes=float(run_params.traffic_resolution),  # es. 5.0
            run_params=run_params,
            data_type='targets')

    # Compute test metrics
    save_metrics(preds_all,
                 targets_all,
                 run_params)


def save_metrics(preds_all, targets_all, run_params):
    # Normalized tensors
    diff = preds_all - targets_all
    mse = (diff ** 2).mean()  # MSE
    rmse = torch.sqrt(mse)  # RMSE
    mae = diff.abs().mean()  # MAE

    # MAPE
    eps = 1e-8
    mape = (diff.abs() / (targets_all.abs().clamp(min=eps))).mean() * 100.0

    # R^2
    ss_res = (diff ** 2).sum()
    ss_tot = ((targets_all - targets_all.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot

    # No Normalized tensors
    F = targets_all.shape[-1]
    preds_all_no_norm = torch.zeros_like(preds_all)
    targets_all_no_norm = torch.zeros_like(targets_all)
    for f in range(F):
        # denormalizzazione: target e predetto
        min_val = run_params.min_vals_normalization[f]
        max_val = run_params.max_vals_normalization[f]

        targets_all_no_norm[:,:,:,f] = targets_all[:,:,:,f] * (max_val - min_val) + min_val
        preds_all_no_norm[:,:,:,f] = preds_all[:,:,:,f] * (max_val - min_val) + min_val

    # Compute test metrics
    diff_no_norm = preds_all_no_norm - targets_all_no_norm
    mse_no_norm = (diff_no_norm ** 2).mean()  # MSE
    rmse_no_norm = torch.sqrt(mse_no_norm)  # RMSE
    mae_no_norm = diff_no_norm.abs().mean()  # MAE

    # MAPE
    mape_no_norm = (diff_no_norm.abs() / (targets_all_no_norm.abs().clamp(min=eps))).mean() * 100.0

    # R^2
    ss_res_no_norm = (diff_no_norm ** 2).sum()
    ss_tot_no_norm = ((targets_all_no_norm - targets_all_no_norm.mean()) ** 2).sum()
    r2_no_norm = 1.0 - ss_res_no_norm / ss_tot_no_norm

    metrics = {"mse":float(mse),
               "rmse":float(rmse),
               "mae":float(mae),
               "mape":float(mape),
               "r2":float(r2),
                "mse_no_norm":float(mse_no_norm),
               "rmse_no_norm":float(rmse_no_norm),
               "mae_no_norm":float(mae_no_norm),
               "mape_no_norm":float(mape_no_norm),
               "r2_no_norm":float(r2_no_norm)}

    # Save metrics
    with open(os.path.join(run_params.output_path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)



def show_predictions(history_all,
                     targets_all,
                     preds_all,
                     run_params):
    """
    Function to show the predictions made by the trained model

    :param history_all:
    :param targets_all:
    :param preds_all:
    :param run_params:
    :return:
    """
    # mostra più sample (una finestra per sample)
    B, N, T, F = preds_all.shape
    ns = min(run_params.num_samples_to_show, B)
    for s in range(ns):
        b_idx = random.randint(0, B - 1)
        n_idx = random.randint(0, N - 1)
        print(f"[{s + 1}/{ns}] Plotting sample — sample {b_idx}, node {n_idx}")
        plot_sample_multifeature(history_all,
                                 targets_all,
                                 preds_all,
                                 b_idx,
                                 n_idx,
                                 run_params,
                                 out_file=os.path.join(run_params.output_path, 'out_img.png'),
                                 sample_id=s + 1)


def generate_outputs_of_trained_model(model,
                                      test_loader,
                                      run_params):

    """
    Function to generate outputs of trained model
    :param model:
    :param test_loader:
    :param run_params:
    :return:
    """
    preds_list, targets_list, history_list = [], [], []  # iterate test set and collect predictions

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if run_params.verbose:
                print(f"Test Batch {batch_idx}")

            # forward
            batch.to(run_params.device)
            y_pred = model.forward(batch)  # shape (B*N, T*F) o 4D a seconda del modello
            y_pred_cpu = y_pred.detach().cpu()

            # target e history
            y_true = getattr(batch, 'y', None)
            y_true_cpu = y_true.detach().cpu()

            x_hist = getattr(batch, 'x', None)
            x_hist_cpu = x_hist.detach().cpu()

            # converti a (B, N, T, F) / (B, N, lags, F)
            pred_BNTF = flat_to_bntf(y_pred_cpu, run_params.num_nodes, run_params.prediction_window)
            targ_BNTF = flat_to_bntf(y_true_cpu, run_params.num_nodes, run_params.prediction_window)
            history_BNlf = torch.reshape(x_hist_cpu, (
                run_params.batch_size, run_params.num_nodes, x_hist_cpu.shape[1], x_hist_cpu.shape[2]))

            preds_list.append(pred_BNTF)
            targets_list.append(targ_BNTF)
            history_list.append(history_BNlf)

    preds_all = torch.cat(preds_list, dim=0) if preds_list else None
    targets_all = torch.cat(targets_list, dim=0) if targets_list else None
    history_all = torch.cat(history_list, dim=0) if history_list else None
    return preds_all, targets_all, history_all


def load_trained_model(model,
                       args):
    """
    Function to load trained model
    :param model:
    :param args:
    :return:
    """

    model.to(args.device)
    model.eval()

    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print("Checkpoint loaded.")
    else:
        print(f"Checkpoint file not found: {args.checkpoint}. Proceeding with random init.")
    return model


# helper to convert flattened (B*N, T*F) -> (B, N, T, F)
def flat_to_bntf(flat_tensor,
                 N,
                 T):
    """
    Function to convert flat_tensor to BNTF

    flat_tensor: Tensor with shape (B*N, T*F) or already 4D like (B, N, T, F) or (B, T, N, F)
    N: num_nodes
    T: prediction horizon
    returns: tensor (B, N, T, F)
    """
    if flat_tensor is None:
        raise ValueError('FlatTensor cannot be None.')

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


def plot_sample_multifeature(history_BN_lf,
                             target_BNTF,
                             pred_BNTF,
                             b_idx,
                             n_idx,
                             run_params,
                             sample_id=None,
                             out_file=None):
    """
    Disegna, in una nuova finestra, un subplot per ciascuna feature:
      - passato (linea continua) sui timestep [0 .. lags-1]
      - futuro target (linea tratteggiata) su [lags .. lags+T-1]
      - futuro predetto (linea continua) su [lags .. lags+T-1]
      - linea verticale nera a x = lags - 0.5

    :param history_BN_lf:
    :param target_BNTF:
    :param pred_BNTF:
    :param b_idx:
    :param n_idx:
    :param run_params:
    :param sample_id:
    :param out_file:
    :return:
    """

    B, N, T, F = pred_BNTF.shape
    lags = history_BN_lf.shape[2]

    # Figure
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


def export_inputs_csv(x_hist_cpu,
                      csv_dir,
                      start_time_str,
                      step_minutes,
                      run_params):
    """
    Salva un CSV per ciascun canale/feature per i dati di input.
    - righe = tutti i timestep concatenati lungo il test set
    - colonne = [timestamp, site_1, ..., site_N]

    :param x_hist_cpu:
    :param csv_dir:
    :param start_time_str:
    :param step_minutes:
    :param run_params:
    :return:
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
        arr = np.transpose(arr, (0, 2, 1))  # (B, lags, N)
        arr = arr.reshape(-1, N)  # (B*lags, N)

        # [OPZIONALE] Denormalizzazione (decommenta se vuoi valori reali)
        min_val = run_params.min_vals_normalization[f]
        max_val = run_params.max_vals_normalization[f]
        arr = arr * (max_val - min_val) + min_val

        # DataFrame
        df = pd.DataFrame(arr, columns=[f"site_{i + 1}" for i in range(N)])
        df.insert(0, "timestamp", time_index)

        # salva
        safe_feat = f"input_feature_{f}"  # opzionale: usa un nome più esplicito se lo hai
        out_path = os.path.join(csv_dir, f"inputs_{safe_feat}.csv")
        df.to_csv(out_path, index=False)
        print(f"[CSV] Saved {out_path} with shape {df.shape}")


def export_predictions_csv(preds_BNTF,
                           csv_dir,
                           start_time_str,
                           step_minutes,
                           run_params,
                           data_type='predictions'):
    """
    Salva un CSV per ciascun canale/feature.
    - righe = tutti i timestep concatenati lungo il test set (ordine dei batch preservato)
    - colonne = [timestamp, site_1, ..., site_N]
    - timestamp parte da start_time_str e avanza di step_minutes

    :param preds_BNTF:
    :param csv_dir:
    :param start_time_str:
    :param step_minutes:
    :param run_params:
    :param data_type:
    :return:
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
        arr = np.transpose(arr, (0, 2, 1))  # (B, T, N)
        arr = arr.reshape(-1, N)  # (B*T, N)

        # [OPZIONALE] Denormalizzazione (decommenta se vuoi valori reali)
        min_val = run_params.min_vals_normalization[f]
        max_val = run_params.max_vals_normalization[f]
        arr = arr * (max_val - min_val) + min_val

        # DataFrame
        df = pd.DataFrame(arr, columns=[f"site_{i + 1}" for i in range(N)])
        df.insert(0, "timestamp", time_index)

        # salva
        safe_feat = "".join([c if c.isalnum() or c in "-_." else "_" for c in feature_names[f]])
        out_path = os.path.join(csv_dir, f"{data_type}_{safe_feat}.csv")
        df.to_csv(out_path, index=False)
        print(f"[CSV] Saved {out_path} with shape {df.shape}")


def main():
    parser = argparse.ArgumentParser()
    # Dataset name
    parser.add_argument("--dataset_name",
                        type=str,
                        default='chicago',
                        help="['denmark', 'metr_la', 'newyork', 'chicago']")

    # Batch size
    parser.add_argument("--batch_size",
                        type=int,
                        default=16)

    # Model
    parser.add_argument("--model",
                        type=str,
                        default='GraphWavenet')

    # Checkpoint
    parser.add_argument("--checkpoint",
                        type=str,
                        default='../registry/checkpoints/ckpt_chicago/last.ckpt',
                        help="Path to checkpoint file (.ckpt or .pt)")

    # Output path
    parser.add_argument("--output_path",
                        type=str,
                        default='../registry/inference_outputs',
                        help="folder to save preds/targets")

    # Show
    parser.add_argument("--show",
                        action='store_true',
                        help="Show sample plots (one window per sample)")

    # Num samples to show
    parser.add_argument("--num_samples_to_show",
                        type=int,
                        default=3,
                        help="Numero di sample da mostrare (ogni sample in una nuova finestra)")

    # Seed
    parser.add_argument("--seed",
                        type=int,
                        default=42)

    # Verbose
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        help="Attiva output dettagliato")

    # Export CSV
    parser.add_argument("--export_csv",
                        default=True,
                        action="store_true",
                        help="Esporta i CSV concatenati per canale")

    # Define ID experiment
    parser.add_argument("--id_run",
                        type=str,
                        default="001",
                        help="Experiment ID")

    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Parameters, dataloader and datamodule
    run_params = Parameters(args)
    data_module_instance, run_params = get_datamodule(run_params)
    data_module_instance.setup(stage='test')
    test_loader = data_module_instance.test_dataloader()

    # Instantiate model
    model = get_model(run_params)
    model = load_trained_model(model,
                               run_params)

    # Generate outputs
    preds_all, targets_all, history_all = generate_outputs_of_trained_model(model,
                                                                            test_loader,
                                                                            run_params)

    # Save predictions
    save_predictions(history_all,
                     targets_all,
                     preds_all,
                     run_params)

    # Show predictions
    show_predictions(history_all,
                     targets_all,
                     preds_all,
                     run_params)


if __name__ == "__main__":
    main()
