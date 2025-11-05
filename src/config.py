import os
import json

class Parameters:
    # Datasets and paths
    save_preprocessed_data = False  # related to traffic_temporal_data_processed
    load_preprocessed_data = False  # related to traffic_temporal_data_processed
    use_traffic_metadata_processed = True
    use_traffic_temporal_data_processed = False
    dataset_name = 'newyork'  # ['METR-LA', 'Electricity']
    project_path = r'/home/alessio/Scrivania/Alessio/EV/EV_GNN_repo/EV_GNN'
    preprocessed_dataset_path = os.path.join(project_path, 'data', 'dev', dataset_name)
    traffic_temporal_data_folder = os.path.join(project_path, 'data', dataset_name, 'traffic/traffic_data')
    traffic_metadata_file = os.path.join(project_path, 'data', dataset_name, 'traffic/processed_newyork_traffic_graph.csv')
    ev_temporal_data_folder =os.path.join(project_path, 'data', dataset_name, 'ev/stations_connectors_counts_data')
    ev_metadata_file = os.path.join(project_path, 'data', dataset_name, 'ev/location_meta_data.csv')
    chkpt_dir = ''
    default_save_tensor_name = '_processed_tensors'

    # Training parameters
    device = 'cuda'
    model = 'GraphWavenet'  # 'gcn', 'gat', 'gcn1d', 'gcn1d-big', 'gConvLSTM', 'gConvGRU', 'DCRNN', 'GraphWavenet', 'AGCRNModel', 'miniLSTM', 'miniGRU'
    lags = 24
    prediction_window = 24
    time_series_step = 4
    batch_size = 64
    train_ratio = 0.7
    val_test_ratio = 0.5
    num_workers = 1
    early_stop_callback_flag = False
    lr = 3e-4
    test_eval = 10
    seed = 42
    id_run = '001'

    # LT Trainer parameters
    accelerator = 'gpu'
    log_every_n_steps = 300
    max_epochs = 150
    enable_progress_bar = True
    check_val_every_n_epoch = 4
    node_features = 24

    # Data params
    num_nodes = 0
    num_of_traffic_nodes_limit = -1  # -1 for all nodes
    num_of_ev_nodes_limit = 200
    traffic_features = 0
    ev_features = 0
    graph_distance_threshold = 10

    # Model parameters
    emb_dim = 32
    dropout = 0.0
    num_layers=2

    # Execution flags
    logging = False
    save_ckpts = True
    save_logs = True
    reproducible = True
    verbose = False
    early_stopping = False

    # Inference parameters
    checkpoint = r''
    output_path = r''
    show = True
    num_samples_to_show = 1
    export_csv = True

    def __init__(self, params=None):
        # Show parser args
        if params.verbose:
            print("Parameters:")
            for name, value in vars(params).items():
                print(f"  {name}: {value}")

        # Copy all default class attributes as istance attribute
        for attr, value in type(self).__dict__.items():
            if not attr.startswith('__') and not callable(value):
                setattr(self, attr, value)

        # If a Namespace is passed, overwrite the values
        if params is not None:
            # If Namespace argparse, convert in dict
            items = (vars(params).items()
                     if not isinstance(params, dict)
                     else params.items())
            for name, val in items:
                # Assign only if attribute exists and val is not None
                if hasattr(self, name) and val is not None:
                    setattr(self, name, val)

        # Actions to execute when instance is created
        if self.num_of_traffic_nodes_limit != -1:
            self.num_nodes = self.num_of_traffic_nodes_limit

        self.preprocessed_data_path = os.path.join(self.project_path, 'dataset', self.dataset_name)
        self.preprocessed_dataset_path = os.path.join(self.project_path, 'data', 'dev', self.dataset_name)
        self.traffic_temporal_data_folder = os.path.join(self.project_path, 'data', self.dataset_name, 'traffic/traffic_data')
        self.traffic_metadata_file = os.path.join(self.project_path, 'data', self.dataset_name,
                                             'traffic/processed_newyork_traffic_graph.csv')
        self.ev_temporal_data_folder = os.path.join(self.project_path, 'data', self.dataset_name, 'ev/stations_connectors_counts_data')
        self.ev_metadata_file = os.path.join(self.project_path, 'data', self.dataset_name, 'ev/location_meta_data.csv')

        self.dirpath_save_ckpt = os.path.join(self.project_path, 'registry', 'checkpoints', f'ckpt_{self.dataset_name}')
        self.dirpath_save_config = os.path.join(self.project_path, 'registry', 'configurations', f'config_{self.id_run}.json')
        os.makedirs(self.dirpath_save_ckpt, exist_ok=True)
        os.makedirs(self.preprocessed_data_path, exist_ok=True)


    def to_json(self):
        """Salva tutte le variabili dell'istanza in un file JSON."""
        # Prendi tutti gli attributi pubblici (evitando metodi e variabili interne)
        data = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('__') and not callable(v)
        }
        # Salva in JSON
        with open(self.dirpath_save_config, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✅ Parametri salvati in {self.dirpath_save_config}")




