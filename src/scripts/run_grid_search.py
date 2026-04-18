import os
from datetime import datetime
import sys
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.config import Parameters
from src.dataset.dataset import get_datamodule
from src.utils.utils import get_model, build_combinations, initialize_log_parameters, setup_seed, update_seed_metrics, \
    update_run_metrics


def run_single_seed(combo: dict,
                    run_params: Parameters,
                    cont: int) -> tuple:
    if run_params.reproducible:
        setup_seed(run_params.seed)

    # Dataset
    data_module_instance, run_params = get_datamodule(run_params)

    # Model
    model = get_model(run_params)

    # Callbacks
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_params.dirpath_save_ckpt,
        save_last=True,
        filename=f"grid_{cont:04d}_" + "id=" + run_params.id_run + "_{epoch}-{val_mse:.6f}",
        save_top_k=2,
        verbose=False,
        monitor='val_mse',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_mse',
        min_delta=0.00,
        patience=4,
        verbose=False,
        mode='min'
    )

    if run_params.save_ckpts:
        callbacks.append(checkpoint_callback)
        enable_checkpointing= True
    else:
        enable_checkpointing = False

    if run_params.early_stopping:
        callbacks.append(early_stop_callback)

    # Trainer
    trainer = pl.Trainer(
        accelerator=run_params.accelerator,
        log_every_n_steps=run_params.log_every_n_steps,
        max_epochs=run_params.max_epochs,
        enable_progress_bar=run_params.enable_progress_bar,
        enable_model_summary=False,
        check_val_every_n_epoch=run_params.check_val_every_n_epoch,
        limit_train_batches=run_params.limit_train_batches,
        logger=False,
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing
    )

    # Train
    trainer.fit(model, datamodule=data_module_instance)

    # Test
    res_test = trainer.test(model, datamodule=data_module_instance)
    return trainer, res_test, run_params


def run_single_combination(combo: dict,
                           cont: int,
                           global_config: dict,
                           seed_list: list):
    print(f'\nRun {cont + 1} | Params: {combo}')
    grid_params_dict = initialize_log_parameters(cont, combo)
    val_results, test_results = [], []

    for seed in seed_list:
        # Merge combo with global config
        global_config['seed'] = seed
        full_config = {**global_config, **combo}
        args = DictNamespace(full_config)
        run_params = Parameters(args)

        # Single-run
        train_module, res_test, run_params = run_single_seed(combo, run_params, cont)
        val_results, test_results = update_seed_metrics(train_module.model, res_test, val_results, test_results)

    update_run_metrics(val_results, test_results, grid_params_dict, run_params)



def main():
    search_space = {
        'dataset_name': ['newyork', 'chicago'],
        'emb_dim':[64,32],
        'dropout':[0.2,0.0],
        'batch_size': [16, 32],
        'model': ['GraphWavenet']}

    global_config = {
        'id_run': '004',
        'save_ckpts': False,
        'early_stopping': True,
        'verbose': False,
    }
    seed_list = [654, 897, ]
    log_folder = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    global_config['logs_dir'] = os.path.join(Parameters(DictNamespace(global_config)).logs_dir, log_folder)
    os.makedirs(global_config['logs_dir'], exist_ok=True)
    combinations = build_combinations(search_space)
    print(f'Total combinations: {len(combinations)}')

    for cont, combo in enumerate(combinations):
        # try:
        #     print(f'\nRun {cont + 1}/{len(combinations)}')
        #     run_single_combination(combo, cont, global_config, seed_list)
        # except Exception:
        #     print(f'Error in run {cont + 1}: ', sys.exc_info()[0])
        print(f'\nRun {cont + 1}/{len(combinations)}')
        run_single_combination(combo, cont, global_config, seed_list)


if __name__ == '__main__':
    main()