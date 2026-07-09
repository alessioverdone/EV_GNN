import argparse
import random
import os
import numpy as np
from lightning.pytorch import LightningDataModule
from torch_geometric.loader import DataLoader as DataLoaderPyg
import torch

from src.config import Parameters
from src.dataset.chicago import DatasetChicago
from src.dataset.denmark import Dataset_Denmark
from src.dataset.losangeles import DatasetLosangeles
from src.dataset.newyork import DatasetNewyork


class EVDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.run_params = params

        # Data and metadata path
        data_path = os.path.join(self.run_params.project_path, 'data', 'raw', self.run_params.dataset_name)
        self.run_params.traffic_temporal_data_folder = os.path.join(data_path,  'traffic/traffic_data')
        self.run_params.traffic_metadata_file = os.path.join(data_path, 'traffic/location_summary.csv')
        self.run_params.ev_temporal_data_folder = os.path.join(data_path, 'ev/ev_locations_availability')
        self.run_params.ev_metadata_file = os.path.join(data_path, 'ev/ev_location_metadata.csv')

        # Denmark dataset
        if self.run_params.dataset_name == 'denmark':
            dataset = Dataset_Denmark(self.run_params)

        # Newyork dataset
        elif self.run_params.dataset_name == 'newyork':
            self.run_params.traffic_columns_to_use = ["speed", "travel_time"]
            self.run_params.ev_columns_to_use = ["Available"]
            dataset = DatasetNewyork(self.run_params)

        # Chicago dataset
        elif self.run_params.dataset_name == 'chicago':
            self.run_params.traffic_columns_to_use = ["speed", "length"]
            self.run_params.ev_columns_to_use = ["Available"]
            dataset = DatasetChicago(self.run_params)

        # Chicago dataset
        elif self.run_params.dataset_name == 'losangeles':
            self.run_params.traffic_columns_to_use = ["avg_speed", "total_flow", "avg_occupancy"]
            self.run_params.ev_columns_to_use = ["Available"]
            dataset = DatasetLosangeles(self.run_params)
        else:
            raise ValueError(f'Dataset {self.run_params.dataset_name} not recognized')

        print(f'Dataset starting time: {dataset.start_time}')
        print(f'Dataset end time: {dataset.end_time}')
        print(f'Dataset traffic mean resolution: {dataset.traffic_resolution}')
        print(f'Dataset ev mean resolution: {dataset.ev_resolution}')

        # Update parameters
        self.num_station = dataset.number_of_station
        self.update_params_after_dataset(dataset)

        # Split data
        self.split_temporal_data(dataset, split_data_modality='sequential')

        # Get dataloaders
        self.train_loader = DataLoaderPyg(self.train_data,
                                          batch_size=self.run_params.batch_size,
                                          shuffle=True,
                                          drop_last=True)

        self.val_loader = DataLoaderPyg(self.val_data,
                                        batch_size=self.run_params.batch_size,
                                        drop_last=True)

        self.test_loader = DataLoaderPyg(self.test_data,
                                         batch_size=self.run_params.batch_size,
                                         drop_last=True)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def split_temporal_data(self, dataset, split_data_modality = 'sequential'):
          # ['random', 'sequential']  # TODO: Metti in config
        if split_data_modality == 'random':
            len_dataset = len(dataset)
            train_snapshots = int(self.run_params.train_ratio * len_dataset)
            val_test_snapshots = len_dataset - train_snapshots
            val_snapshots = int(self.run_params.val_test_ratio * val_test_snapshots)
            test_snapshots = len_dataset - train_snapshots - val_snapshots
            self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(dataset, [train_snapshots,
                                                                                                     val_snapshots,
                                                                                                     test_snapshots])  # N, T, F
        elif split_data_modality == 'sequential':
            len_dataset = len(dataset)
            train_snapshots = int(self.run_params.train_ratio * len_dataset)
            val_test_snapshots = len_dataset - train_snapshots
            val_snapshots = int(self.run_params.val_test_ratio * val_test_snapshots)
            test_snapshots = len_dataset - train_snapshots - val_snapshots
            start_point = random.randint(0, len_dataset - test_snapshots)
            test_start = start_point
            test_end = start_point + test_snapshots
            train_start = test_end
            train_end = train_start + train_snapshots
            val_start = train_end
            val_end = val_start + val_snapshots
            dataset_indices = np.arange(len_dataset).tolist() + np.arange(
                len_dataset).tolist()  # 2 times because restart from the beginning
            train_idx = dataset_indices[train_start:train_end]
            val_idx = dataset_indices[val_start:val_end]
            test_idx = dataset_indices[test_start:test_end]
            self.train_data = torch.utils.data.Subset(dataset, train_idx)
            self.val_data = torch.utils.data.Subset(dataset, val_idx)
            self.test_data = torch.utils.data.Subset(dataset, test_idx)

            # Here i'm sure by construction that test doesn't overlap the end of the dataset
            self.run_params.start_time_test = list(self.test_data[0].time_input)[0].isoformat()
            self.run_params.end_time_test = list(self.test_data[len(self.test_data) - 1].time_output)[-1].isoformat()
        else:
            raise ValueError(f'split_data_modality not recognized')

    def update_params_after_dataset(self, dataset : torch.utils.data.Dataset):
        self.run_params.num_nodes = self.num_station
        self.run_params.traffic_features = dataset.traffic_features
        self.run_params.ev_features = dataset.ev_features
        self.run_params.traffic_features_names = dataset.traffic_columns_used_in_data.tolist()
        self.run_params.ev_features_names = dataset.ev_columns_used_in_data.tolist()
        self.run_params.min_vals_normalization = dataset.min_vals_normalization.tolist()
        self.run_params.max_vals_normalization = dataset.max_vals_normalization.tolist()
        self.run_params.start_time = dataset.start_time.isoformat()
        self.run_params.end_time = dataset.end_time.isoformat()
        self.run_params.traffic_resolution = dataset.traffic_resolution
        self.run_params.ev_resolution = dataset.ev_resolution


def get_datamodule(params):
    # Dataset from scratch
    if params.dataset_name in ['denmark', 'newyork', 'chicago', 'losangeles']:
        data_module_instance = EVDataModule(params)
        params = data_module_instance.run_params
    else:
        raise ValueError('Define dataset name correct!')

    return data_module_instance, params


if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser(description="Experiments parameters!")
    parser.add_argument("--dataset_name", type=str, default='newyork',
                        help="['denmark', 'metr_la', 'newyork', 'chicago', 'losangeles]")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size!")
    parser.add_argument("--graph_distance_threshold", type=float, default=50, help="Cluster node thresh.")
    parser.add_argument("--model", type=str, default='GraphWavenet', help="Select model!")
    parser.add_argument("--verbose", "-v", action="store_false", help="Attiva output dettagliato")
    args = parser.parse_args()

    # Parameters
    instance_parameters = Parameters(args)

    # Datamodule
    dm = EVDataModule(instance_parameters)
    print('DM created!') # con 20 graph_distance_threshold dà errore


