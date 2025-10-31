from einops import rearrange
from sklearn.metrics import mean_absolute_error as MAE, root_mean_squared_error, mean_absolute_percentage_error
from torch import optim, nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import functional as F
import torch
from tsl.nn.models import DCRNNModel, GraphWaveNetModel, AGCRNModel
from lightning.pytorch import LightningModule

from src.model.GCN1D import GCN1DConv_big, GCN1DConv
from src.model.GConvRNN import GraphConvRNN_our
from src.model.miniRNN import MultiLayerLSTMParallel, MultiLayerGRUParallel


class TF_model(LightningModule):
    def __init__(self, params):
        super(TF_model, self).__init__()
        hidden_channels = params.emb_dim
        node_features = params.node_features
        num_nodes = params.num_nodes
        self.params = params
        # GCN
        if params.model == 'gcn':
            self.conv1 = GCNConv(node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # GAT
        elif params.model == 'gat':
            self.conv1 = GATConv(node_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
            self.conv3 = GATConv(hidden_channels, hidden_channels)

        # GCN1D
        elif params.model == 'gcn1d':
            self.gcn1d_model = GCN1DConv(node_features, params.prediction_window, 1, 1, hid_dim=hidden_channels,)
        # GCN1D-B
        elif params.model == 'gcn1d-big':
            self.gcn1d_big_model = GCN1DConv_big(node_features, params.prediction_window, 1, 1, hid_dim=hidden_channels,)
        # GCONV-LSTM
        elif  params.model == 'gConvLSTM':
            self.g_conv_lstm = GraphConvRNN_our(1, params.prediction_window, hidden_channels, params, cell_type='LSTM')
        # GCONV-GRU
        elif  params.model == 'gConvGRU':
            self.g_conv_gru = GraphConvRNN_our(1, params.prediction_window, hidden_channels, params, cell_type='GRU')
        # DCRNN
        elif params.model == 'DCRNN':
            self.dcrnn = DCRNNModel(input_size=1,
                                output_size=1,
                                horizon=params.prediction_window,
                                hidden_size=hidden_channels,
                                kernel_size=2,
                                n_layers=2,
                                dropout=0.0,
                                activation='relu')
        # GraphWavenet
        elif params.model == 'GraphWavenet':
            self.graph_wavenet = GraphWaveNetModel(input_size=self.params.traffic_features+self.params.ev_features,
                                                    output_size=self.params.traffic_features+self.params.ev_features,
                                                    horizon=params.prediction_window,
                                                    hidden_size=hidden_channels,
                                                    n_nodes=params.num_nodes,
                                                    dropout=0.3)
        # AGCRN
        elif params.model == 'AGCRNModel':
            self.agcrn = AGCRNModel(input_size=1,
                                    output_size=1,
                                    horizon=params.prediction_window,
                                    hidden_size=hidden_channels,
                                    exog_size=0,
                                    n_nodes=params.num_nodes)

        # miniLSTM
        elif params.model == 'miniLSTM':
            self.miniLSTM = MultiLayerLSTMParallel(input_size=1,
                                                   hidden_size=hidden_channels,
                                                   output_size=params.prediction_window,
                                                   num_layers=params.num_layers,
                                                   seq_len=params.lags)
        # miniGRU
        elif params.model == 'miniGRU':
            self.miniGRU = MultiLayerGRUParallel(input_size=1,
                                                   hidden_size=hidden_channels,
                                                   output_size=params.prediction_window,
                                                   num_layers=params.num_layers,
                                                   seq_len=params.lags)


        self.lin = Linear(hidden_channels, params.prediction_window)
        self.params = params

        # torch lightning specific
        self.automatic_optimization = False
        self.debug = False
        self.params = params
        self.best_mse = 1000
        self.best_mae = 1000

    def forward(self, data):
        x, y, edge_index, edge_weight = (data.x,
                                         data.y[:, :self.params.prediction_window],
                                         data.edge_index,
                                         data.edge_attr)
        if self.params.dataset_name in ['METR-LA', 'solar', 'electricity'] and len(x.shape) == 4:
            if self.params.model not in ['DCRNN', 'GraphWavenet', 'AGCRNModel']:
                x = rearrange(x, 'b t n f -> (b n) (t f)')
            y = rearrange(y, 'b t n f -> (b n) (t f)')    # 1. Obtain node embeddings

        if self.params.model in ['gcn', 'gat']:
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)

            # 2. Apply a final regressor
            x = F.dropout(x, p=self.params.dropout, training=self.training)
            x = self.lin(x)
        elif self.params.model == 'gcn1d':
            x = self.gcn1d_model(x, edge_index)

        elif self.params.model == 'gcn1d-big':
            x = self.gcn1d_big_model(x, edge_index)

        elif self.params.model == 'gConvLSTM':
            x = self.g_conv_lstm(x, edge_index)

        elif self.params.model == 'gConvGRU':
            x = self.g_conv_gru(x, edge_index)

        elif self.params.model == 'DCRNN':
            # torch-spatiotemporal library data format
            if x.shape != 4:
                x = torch.reshape(x.unsqueeze(-1), (self.params.batch_size, self.params.lags, self.params.num_nodes, 1))
                edge_index = edge_index[:, :int(edge_index.shape[1]/self.params.batch_size)]
            x = self.dcrnn(x, edge_index)
            x = rearrange(x, 'b t n f ->  (b n) (t f) ')

        elif self.params.model == 'GraphWavenet':
            # torch-spatiotemporal library data format
            if x.shape != 4:
                # x = torch.reshape(x.unsqueeze(-1), (self.params.batch_size, self.params.lags, self.params.num_nodes, 1))
                # edge_index = edge_index[:, :int(edge_index.shape[1]/self.params.batch_size)]
                x = torch.reshape(x.unsqueeze(-1), (self.params.batch_size,
                                                    self.params.lags,
                                                    self.params.num_nodes,
                                                    self.params.traffic_features+self.params.ev_features))  # newyork era 7
                edge_index = edge_index[:, :int(edge_index.shape[1]/self.params.batch_size)]
            x = self.graph_wavenet(x, edge_index)
            x = rearrange(x, 'b t n f ->  (b n) (t f) ')

        elif self.params.model == 'AGCRNModel':
            # torch-spatiotemporal library data format
            if x.shape != 4:
                x = torch.reshape(x.unsqueeze(-1), (self.params.batch_size, self.params.lags, self.params.num_nodes, 1))
                edge_index = edge_index[:, :int(edge_index.shape[1]/self.params.batch_size)]
            x = self.agcrn(x)
            x = rearrange(x, 'b t n f ->  (b n) (t f) ')

        elif self.params.model == 'miniLSTM':
            x = self.miniLSTM(x)

        elif self.params.model == 'miniGRU':
            x = self.miniGRU(x)
        return x

    def on_validation_epoch_end(self):
        actual_loss = self.trainer.logged_metrics['val_mse'].tolist()
        actual_rmse = self.trainer.logged_metrics['val_rmse'].tolist()
        actual_mae = self.trainer.logged_metrics['val_mae'].tolist()
        actual_mape = self.trainer.logged_metrics['val_mape'].tolist()
        if actual_loss < self.best_mse:
            self.best_mse = actual_loss
            self.best_rmse = actual_rmse
            self.best_mae = actual_mae
            self.best_mape = actual_mape

        sch = self.lr_schedulers()
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_mse"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.lr)
        # Define the scheduler, you can set the patience and other options
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.15)  # , verbose=True

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_mse',  # Metric to monitor for reducing learning rate
                'frequency': 1,  # How often to call the scheduler (after every epoch in this case)
            }
        }

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()

        # Get data from batches
        x, y, edge_index, edge_weight = (train_batch.x,
                                         train_batch.y,
                                         train_batch.edge_index,
                                         train_batch.edge_attr)
        if self.params.dataset_name in ['METR-LA', 'solar', 'electricity'] and len(x.shape) == 4:
            x = rearrange(x, 'b t n f -> (b n) (t f)')
            y = rearrange(y, 'b t n f -> (b n) (t f)')

        y_predicted = self.forward(train_batch)
        loss_forecasting = F.mse_loss(y_predicted, y)
        loss_forecasting.backward()
        train_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        train_rmse = root_mean_squared_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        train_mape = mean_absolute_percentage_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        optimizer.step()

        self.log('train_mse', loss_forecasting.detach(), batch_size=self.params.batch_size, prog_bar=True)
        self.log('train_rmse', train_rmse, batch_size=self.params.batch_size, prog_bar=True)
        self.log('train_mae', train_mae, batch_size=self.params.batch_size, prog_bar=True)
        self.log('train_mape', train_mape, batch_size=self.params.batch_size, prog_bar=True)

    def validation_step(self, val_batch, batch_idx):
        # Get data from batches
        x, y, edge_index, edge_weight = (val_batch.x,
                                         val_batch.y,  #  val_batch.y[:, :self.params.prediction_window],
                                         val_batch.edge_index,
                                         val_batch.edge_attr)
        if self.params.dataset_name in ['METR-LA', 'solar', 'electricity'] and len(x.shape) == 4:
            x = rearrange(x, 'b t n f -> (b n) (t f)')
            y = rearrange(y, 'b t n f -> (b n) (t f)')

        y_predicted = self.forward(val_batch)
        loss_forecasting = F.mse_loss(y_predicted, y)
        val_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        val_rmse = root_mean_squared_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        val_mape = mean_absolute_percentage_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())

        self.log('val_mse', loss_forecasting.detach(), batch_size=self.params.batch_size, prog_bar=True)
        self.log('val_rmse', val_rmse, batch_size=self.params.batch_size, prog_bar=True)
        self.log('val_mae', val_mae, batch_size=self.params.batch_size, prog_bar=True)
        self.log('val_mape', val_mape, batch_size=self.params.batch_size, prog_bar=True)


    def test_step(self, test_batch, batch_idx):
        # Get data from batches
        x, y, edge_index, edge_weight = (test_batch.x,
                                         test_batch.y,
                                         test_batch.edge_index,
                                         test_batch.edge_attr)
        if self.params.dataset_name in ['METR-LA', 'solar', 'electricity'] and len(x.shape) == 4:
            x = rearrange(x, 'b t n f -> (b n) (t f)')
            y = rearrange(y, 'b t n f -> (b n) (t f)')

        y_predicted = self.forward(test_batch)
        loss_forecasting = F.mse_loss(y_predicted, y)
        test_mae = MAE(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        test_rmse = root_mean_squared_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        test_mape = mean_absolute_percentage_error(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
        self.log('test_mse', loss_forecasting.detach().cpu(), batch_size=self.params.batch_size)
        self.log('test_rmse', test_rmse, batch_size=self.params.batch_size)
        self.log('test_mae', test_mae, batch_size=self.params.batch_size)
        self.log('test_mape', test_mape, batch_size=self.params.batch_size)
