import argparse
import lightning as pl

from src.config import Parameters
from src.dataset.dataset import get_datamodule
from src.utils.utils import get_model, get_callbacks

# Parser
parser = argparse.ArgumentParser(description="Experiments parameters!")
parser.add_argument("--dataset_name", type=str, default='chicago', help="['denmark', 'metr_la', 'newyork', 'chicago']")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size!")
parser.add_argument("--model", type=str, default='GraphWavenet', help="Select model!")
parser.add_argument("--verbose", "-v", action="store_false", help="Attiva output dettagliato")
args = parser.parse_args()

# Parameters
run_params = Parameters(args)

# Get dataset
dataModuleInstance, run_params = get_datamodule(run_params)

# Import model
model = get_model(run_params)

# Import callbacks
callbacks = get_callbacks(run_params)

# Training
trainer = pl.Trainer(accelerator=run_params.accelerator,
                     log_every_n_steps=run_params.log_every_n_steps,
                     max_epochs=run_params.max_epochs,
                     enable_progress_bar=run_params.enable_progress_bar,
                     enable_model_summary=False,
                     check_val_every_n_epoch=run_params.check_val_every_n_epoch,
                     logger=False,
                     callbacks=[])

# Start training
trainer.fit(model, datamodule=dataModuleInstance)

# Testing
res_test = trainer.test(model, datamodule=dataModuleInstance)
