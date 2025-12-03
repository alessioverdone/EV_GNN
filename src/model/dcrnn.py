import torch
import lightning as pl
from tsl.nn.models import DCRNNModel

class TSL_model(pl.LightningModule):
    def __init__(self, node_features, out_dim, hidden_channels, params):
        super(TSL_model, self).__init__()
        self.node_features = node_features
        self.out_dim = out_dim
        self.hidden_channels = hidden_channels
        self.params = params

        self.model = DCRNNModel(input_size=self.node_features,
                                output_size=1,
                                horizon=self.out_dim,
                                hidden_size=self.hidden_channels,
                                kernel_size=2,
                                n_layers=2,
                                dropout=0.0,
                                activation='relu')


    def forward(self, x, edge_index, edge_weight=None):
        if self.cell_type == 'LSTM':
            h_0 = torch.zeros(x.shape[0],  self.hidden_channels).to(x.device)
            c_0 = torch.zeros(x.shape[0],  self.hidden_channels).to(x.device)
            h_1 = torch.zeros(x.shape[0], self.hidden_channels).to(x.device)
            c_1 = torch.zeros(x.shape[0], self.hidden_channels).to(x.device)
            # x has shape: [Batch x #Nodes, #Features x #Lags]
            x = torch.reshape(x, (-1, 1, self.params.lags))
            # x has shape: [Batch x #Nodes, #Features, #Lags]
            for i in range(self.params.lags):
                x_t = x[:, :, i]
                h_0, c_0 = self.recurrent(x_t, edge_index, edge_weight, H=h_0, C=c_0)
                h_0 = self.relu(h_0)
                h_1, c_1 = self.recurrent2(h_0, edge_index, edge_weight, H=h_1, C=c_1)
                h_1 = self.tanh(h_1)

        elif self.cell_type == 'GRU':
            h_0 = torch.zeros(x.shape[0], self.hidden_channels).to(x.device)
            h_1 = torch.zeros(x.shape[0], self.hidden_channels).to(x.device)
            # x has shape: [Batch x #Nodes, #Features x #Lags]
            x = torch.reshape(x, (-1, 1, self.params.lags))
            # x has shape: [Batch x #Nodes, #Features, #Lags]
            for i in range(self.params.lags):
                x_t = x[:, :, i]
                h_0 = self.recurrent(x_t, edge_index, edge_weight, H=h_0)
                h_0 = self.relu(h_0)
                h_1 = self.recurrent2(h_0, edge_index, edge_weight, H=h_1)
                h_1 = self.tanh(h_1)
        else:
            raise ValueError("cell_type not defined!")
        h = self.linear(h_1)
        return h

    def __str__(self):
        return "GConvLSTM"