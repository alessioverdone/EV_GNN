# inference.py
import argparse
import random
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

from src.config import Parameters
from src.dataset.dataset import get_datamodule
from src.utils.utils import get_model

# helper to convert flattened (B*N, T*F) -> (B, N, T, F)
def flat_to_BNTF(flat_tensor, N, T):
    """
    flat_tensor: Tensor with shape (B*N, T*F) or (B*N, T) / (B*N, T) etc.
    N: num_nodes
    T: prediction horizon
    returns: tensor (B, N, T, F)
    """
    if flat_tensor is None:
        return None
    if flat_tensor.dim() == 4:
        # already (B, T, N, F) or (B, N, T, F) — normalize to (B, N, T, F)
        if flat_tensor.shape[1] == T:
            # shape (B, T, N, F) -> permute
            if flat_tensor.shape[2] is not None:
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

    # x-axis per passato e futuro
    x_past = np.arange(0, lags)
    x_future = np.arange(lags, lags + T)

    columns_name = run_params.traffic_columns_to_use + run_params.ev_columns_to_use  # TODO: add dict real name/visualiz. name

    for f in range(F):
        # denormalizzazione: target e predetto
        min_val = run_params.min_vals_normalization[f]
        max_val = run_params.max_vals_normalization[f]

        ax = axes[f]
        # passato
        # if history_BN_lf is not None:
        #     past = history_BN_lf[b_idx, n_idx, :, f]  # (lags,)
        #     past_real = past * (max_val - min_val) + min_val
        #     ax.plot(x_past, past_real, linewidth=1.5, label="past")
        # else:
        #     ax.text(0.5, 0.5, 'No history available', ha='center', va='center', transform=ax.transAxes)
        past = history_BN_lf[b_idx, n_idx, :, f]  # (lags,)
        past_real = past * (max_val - min_val) + min_val
        ax.plot(x_past, past_real, linewidth=1.5, label="past")

        # futuro: target (tratteggiato) e pred (continuo)
        targ = target_BNTF[b_idx, n_idx, :, f]
        pred = pred_BNTF[b_idx, n_idx, :, f]

        # Denormalizza
        targ_real = targ * (max_val - min_val) + min_val
        pred_real = pred * (max_val - min_val) + min_val

        ax.plot(x_future, targ_real, linestyle='--', linewidth=1.5, label="target")
        ax.plot(x_future, pred_real, linestyle='-', linewidth=1.5, label="pred")

        # linea verticale al confine tra passato e futuro
        ax.axvline(x=lags - 0.5, color='black', linewidth=1.0)

        # titolino con id feature
        ax.set_title(f" {columns_name[f]}", loc='left')
        ax.set_ylabel("value")

        # opzionale: limiti x più comodi
        ax.set_xlim(-0.5, lags + T - 0.5)
        ax.set_ylim(min([past_real.min(), targ_real.min(), pred_real.min()]), max([past_real.max(), targ_real.max(), pred_real.max()]))


        # metti la leggenda solo nel primo subplot per non affollare
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
        # nuova finestra per ogni sample
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='newyork', help="['denmark', 'metr_la', 'newyork', 'chicago']")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", type=str, default='GraphWavenet')
    parser.add_argument("--checkpoint", type=str, default='../checkpoints/last-v4.ckpt', help="Path to checkpoint file (.ckpt or .pt)")
    parser.add_argument("--output_path", type=str, default='./inference_outputs', help="folder to save preds/targets")
    parser.add_argument("--show", default=True, action='store_true', help="Show sample plots (one window per sample)")
    parser.add_argument("--num_samples_to_show", type=int, default=3, help="Numero di sample da mostrare (ogni sample in una nuova finestra)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_false", help="Attiva output dettagliato")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Parameters and datamodule
    run_params = Parameters(args)
    run_params.time_series_step = 24

    dataModuleInstance, run_params = get_datamodule(run_params)

    # create dataloaders
    dataModuleInstance.setup(stage='test')
    test_loader = dataModuleInstance.test_dataloader()

    # instantiate model
    model = get_model(run_params)
    device = torch.device('cuda' if torch.cuda.is_available() and run_params.accelerator in ['gpu', 'cuda'] else 'cpu')
    model.to(device)
    model.eval()
    save_predictions = False

    # load checkpoint if provided
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location=device)
            state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            model.load_state_dict(state)

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

    preds_all = torch.cat(preds_list, dim=0)
    targets_all = torch.cat(targets_list, dim=0)
    history_all = torch.cat(history_list, dim=0)

    os.makedirs(args.output_path, exist_ok=True)

    # opzionale: salva
    if save_predictions:
        preds_file = os.path.join(args.output_path, 'preds_BNTF.pt')
        targets_file = os.path.join(args.output_path, 'targets_BNTF.pt')
        history_file = os.path.join(args.output_path, 'history_BNlf.pt')

        if preds_all is not None:
            torch.save(preds_all, preds_file)
            print(f"Saved predictions to {preds_file} with shape {tuple(preds_all.shape)}")
        if targets_all is not None:
            torch.save(targets_all, targets_file)
            print(f"Saved targets to {targets_file} with shape {tuple(targets_all.shape)}")
        if history_all is not None:
            torch.save(history_all, history_file)
            print(f"Saved history to {history_file} with shape {tuple(history_all.shape)}")

    # mostra più sample (una finestra per sample)
    if args.show and preds_all is not None and targets_all is not None:
        B, N, T, F = preds_all.shape
        for s in range(args.num_samples_to_show):
            b_idx = random.randint(0, B - 1)
            n_idx = random.randint(0, N - 1)
            print(f"[{s+1}/{args.num_samples_to_show}] Plotting sample — sample {b_idx}, node {n_idx}")
            plot_sample_multifeature(history_all, targets_all, preds_all, b_idx, n_idx, run_params, sample_id=s+1, out_file=None)

if __name__ == "__main__":
    main()
