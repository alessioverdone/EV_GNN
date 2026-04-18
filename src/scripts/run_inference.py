import argparse
import random
import torch
import numpy as np
import matplotlib

from src.utils.inference_utils import load_trained_model, generate_outputs_of_trained_model, save_predictions, \
    show_predictions
from src.utils.utils import get_model

matplotlib.use("tkagg")

from src.config import Parameters
from src.dataset.dataset import get_datamodule


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
                        default='../registry/checkpoints/ckpt_chicago/id=005_epoch=3-val_mse=0.004710',
                        help="Path to checkpoint file (.ckpt or .pt)")

    # Output path
    parser.add_argument("--output_path",
                        type=str,
                        default='../../registry/inference_outputs',
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
                        default="005",
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
