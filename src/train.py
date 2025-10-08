import argparse
import lightning as pl
import pytorch_lightning as lp
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.config import Parameters
from src.dataset.dataset import get_datamodule
from src.utils.utils import get_model, get_callbacks

# Parser
parser = argparse.ArgumentParser(description="Experiments parameters!")
parser.add_argument("--dataset_name", type=str, default='newyork', help="['denmark', 'metr_la', 'newyork', 'chicago']")
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
callbacks = list()
checkpoint_callback = ModelCheckpoint(
    dirpath='../checkpoints',
    save_last=True,
    filename="{epoch}-{val_mse:.6f}",  # naming
    save_top_k=2,
    verbose=True,
    monitor='val_mse',
    mode='min'
)

early_stop_callback = EarlyStopping(
    monitor='val_mse',
    min_delta=0.00,
    patience=4,
    verbose=False,
    mode='min')

if run_params.save_ckpts:
    callbacks += [checkpoint_callback]

if run_params.early_stopping:
    callbacks += [early_stop_callback]

# Training
trainer = pl.Trainer(accelerator=run_params.accelerator,
                     log_every_n_steps=run_params.log_every_n_steps,
                     max_epochs=run_params.max_epochs,
                     enable_progress_bar=run_params.enable_progress_bar,
                     enable_model_summary=False,
                     check_val_every_n_epoch=run_params.check_val_every_n_epoch,
                     logger=False,
                     callbacks=callbacks)

# Start training
trainer.fit(model, datamodule=dataModuleInstance)

# Testing
res_test = trainer.test(model, datamodule=dataModuleInstance)

# Test on 150 epochs
#        Test metric             DataLoader 0
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#         test_mae           0.007828162983059883
#         test_mape            1.553382396697998
#         test_mse          0.00039691291749477386
#         test_rmse           0.01533879991620779
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────