import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch_geometric.nn.inits import zeros
from torch import Tensor
from torch.nn import Parameter, init
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import (get_laplacian,
                                   remove_self_loops)
from typing import Optional
import math


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 padding=0,
                 bias=True, h_conv=0):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = padding

    def forward(self, input):
        input = F.pad(input, (self.__padding * 2, 0))
        return super(CausalConv1d, self).forward(input)


class Conv_base(torch.nn.Module):
    def __init__(self, input_features_cnn, output_features_cnn, in_channels, out_channels, kernel_size, h_conv, params):
        super(Conv_base, self).__init__()
        self.params = params
        self.h_conv = h_conv

        kernel_size = kernel_size  # 7
        input_features_cnn = input_features_cnn  # 32
        self.in_channels = in_channels  # 128
        out_channels = out_channels  # 128
        output_features_cnn = output_features_cnn  # 32
        dilation = 1
        stride = 1
        padding = int(
            (stride * (output_features_cnn - 1) - input_features_cnn + (dilation * (kernel_size - 1) + 1)) / 2)
        self.conv1d_1 = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                     dilation=dilation, stride=stride)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.conv1d_1.weight, a=math.sqrt(5))

        if self.conv1d_1.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.conv1d_1.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.conv1d_1.bias, -bound, bound)

    def forward(self, x_conv):
        h = torch.reshape(x_conv, (x_conv.shape[0], self.in_channels, -1))
        h = self.conv1d_1(h)
        h = torch.reshape(h, (h.shape[0], -1))
        return h


class ChebConv(MessagePassing):
    def __init__(self, input_features_cnn, output_features_cnn, kernel_size, params, in_channels: int,
                 out_channels: int, K: int,
                 normalization: Optional[str] = 'sym', bias: bool = True, h_conv=False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.lins = torch.nn.ModuleList(
            [Conv_base(input_features_cnn, output_features_cnn, in_channels, out_channels, kernel_size, h_conv, params)
             for _ in range(K)])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None,
                 batch: OptTensor = None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        # It causes problems in GNNExplainer evaluation
        # edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
        #                                          fill_value=-1.,
        #                                          num_nodes=num_nodes)
        assert edge_weight is not None

        return edge_index, edge_weight

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: OptTensor = None, batch: OptTensor = None,
                lambda_max: OptTensor = None):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
                                      device=x.device)
        assert lambda_max is not None

        edge_index, norm = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype,
                                         batch=batch)

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')


class GCN1D_Encoder(nn.Module):
    """
    Model for doing forecasting on temporal series data
    Functions:
        __init__
        forward
    """

    def __init__(self, params, last_layer_linear=False):
        super(GCN1D_Encoder, self).__init__()
        self.params = params
        self.in_channels = self.params.NODE_FEATURES * 5
        self.K = self.params.FILTER_SIZE
        self.normalization = "sym"
        self.bias = None
        self.INPUT_LINEAR_LAYERS_UNITS = None
        self.last_layer_linear = last_layer_linear

        # Modello TOP
        self.conv_1 = ChebConv(input_features_cnn=self.params.LAGS,
                               output_features_cnn=32,  # 32
                               kernel_size=5,  # 7
                               params=self.params,
                               in_channels=self.params.INPUT_CHANNELS,
                               out_channels=24,
                               K=self.K,
                               normalization=self.normalization,
                               bias=self.bias, h_conv=0
                               )
        self.b_1 = Parameter(torch.Tensor(1, 32 * 24))

        self.conv_2 = ChebConv(input_features_cnn=32,  # 32
                               output_features_cnn=30,  # 28 #30
                               kernel_size=5,  # 7
                               params=self.params,
                               in_channels=24,
                               out_channels=32,
                               K=self.K,
                               normalization=self.normalization,
                               bias=self.bias, h_conv=1
                               )
        self.b_2 = Parameter(torch.Tensor(1, 32 * 30))
        self.conv_3 = ChebConv(input_features_cnn=30,  # 28 #30
                               output_features_cnn=28,  # 24 #28
                               kernel_size=5,  # 5 #7
                               params=self.params,
                               in_channels=32,
                               out_channels=64,
                               K=self.K,
                               normalization=self.normalization,
                               bias=self.bias, h_conv=1
                               )
        self.b_3 = Parameter(torch.Tensor(1, 64 * 28))
        self.conv_4 = ChebConv(input_features_cnn=28,  # 24 #28
                               output_features_cnn=26,  # 20 #26
                               kernel_size=3,  # 5
                               params=self.params,
                               in_channels=64,
                               out_channels=64,
                               K=self.K,
                               normalization=self.normalization,
                               bias=self.bias, h_conv=1
                               )
        self.b_4 = Parameter(torch.Tensor(1, 64 * 26))
        self.conv_5 = ChebConv(input_features_cnn=26,  # 20  #26
                               output_features_cnn=24,  # 16  #24
                               kernel_size=3,  # 5 #7
                               params=self.params,
                               in_channels=64,
                               out_channels=96,
                               K=self.K,
                               normalization=self.normalization,
                               bias=self.bias, h_conv=1
                               )
        self.b_5 = Parameter(torch.Tensor(1, 96 * 24))
        if self.params.NUM_LAYERS >= 6:
            self.conv_6 = ChebConv(input_features_cnn=24,  # 16   #24
                                   output_features_cnn=22,  # 14  #22
                                   kernel_size=3,  # 3  #7
                                   params=self.params,
                                   in_channels=96,
                                   out_channels=96,
                                   K=self.K,
                                   normalization=self.normalization,
                                   bias=self.bias, h_conv=1
                                   )
            self.b_6 = Parameter(torch.Tensor(1, 96 * 22))
            self.INPUT_LINEAR_LAYERS_UNITS = 96 * 22
        if self.params.NUM_LAYERS >= 7:
            self.conv_7 = ChebConv(input_features_cnn=22,  # 14  #22
                                   output_features_cnn=20,  # 12  #20
                                   kernel_size=3,  # 3  #7
                                   params=self.params,
                                   in_channels=96,
                                   out_channels=128,
                                   K=self.K,
                                   normalization=self.normalization,
                                   bias=self.bias, h_conv=2
                                   )
            self.b_7 = Parameter(torch.Tensor(1, 128 * 20))
            self.INPUT_LINEAR_LAYERS_UNITS = 128 * 20
        if self.params.NUM_LAYERS >= 8:
            self.conv_8 = ChebConv(input_features_cnn=20,  # 14  #22
                                   output_features_cnn=18,  # 12  #20
                                   kernel_size=3,  # 3  #7
                                   params=self.params,
                                   in_channels=128,
                                   out_channels=192,
                                   K=self.K,
                                   normalization=self.normalization,
                                   bias=self.bias, h_conv=2
                                   )
            self.b_8 = Parameter(torch.Tensor(1, 192 * 18))
            self.INPUT_LINEAR_LAYERS_UNITS = 192 * 18
        if self.params.NUM_LAYERS >= 9:
            self.conv_9 = ChebConv(input_features_cnn=18,  # 14  #22
                                   output_features_cnn=16,  # 12  #20
                                   kernel_size=3,  # 3  #7
                                   params=self.params,
                                   in_channels=192,
                                   out_channels=256,
                                   K=self.K,
                                   normalization=self.normalization,
                                   bias=self.bias, h_conv=2
                                   )
            self.b_9 = Parameter(torch.Tensor(1, 256 * 16))
            self.INPUT_LINEAR_LAYERS_UNITS = 256 * 16
        if self.params.NUM_LAYERS >= 10:
            self.conv_10 = ChebConv(input_features_cnn=16,  # 14  #22
                                    output_features_cnn=14,  # 12  #20
                                    kernel_size=3,  # 3  #7
                                    params=self.params,
                                    in_channels=256,
                                    out_channels=320,
                                    K=self.K,
                                    normalization=self.normalization,
                                    bias=self.bias, h_conv=2
                                    )
            self.b_10 = Parameter(torch.Tensor(1, 320 * 14))
            self.INPUT_LINEAR_LAYERS_UNITS = 320 * 14
        if self.params.NUM_LAYERS >= 11:
            self.conv_11 = ChebConv(input_features_cnn=14,  # 14  #22
                                    output_features_cnn=12,  # 12  #20
                                    kernel_size=3,  # 3  #7
                                    params=self.params,
                                    in_channels=320,
                                    out_channels=320,
                                    K=self.K,
                                    normalization=self.normalization,
                                    bias=self.bias, h_conv=2
                                    )
            self.b_11 = Parameter(torch.Tensor(1, 320 * 12))
            self.INPUT_LINEAR_LAYERS_UNITS = 320 * 12

        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(self.INPUT_LINEAR_LAYERS_UNITS,
                                      self.params.PREDICTION_WINDOW)
        # self.linear_time = torch.nn.Linear(self.params.LAGS,
        #                                    self.params.OUTPUT_MLP_DIMENSION_FORECASTING)

        self.params = params

        zeros(self.b_1)
        zeros(self.b_2)
        zeros(self.b_3)
        zeros(self.b_4)
        zeros(self.b_5)
        if self.params.NUM_LAYERS >= 6:
            zeros(self.b_6)
        if self.params.NUM_LAYERS >= 7:
            zeros(self.b_7)
        if self.params.NUM_LAYERS >= 8:
            zeros(self.b_8)
        if self.params.NUM_LAYERS >= 9:
            zeros(self.b_9)
        if self.params.NUM_LAYERS >= 10:
            zeros(self.b_10)
        if self.params.NUM_LAYERS >= 11:
            zeros(self.b_11)

    def forward(self, x, edge_index, edge_weight):
        h_x = self.relu(self.conv_1(x, edge_index, edge_weight) + self.b_1)
        h = self.relu(self.conv_2(h_x, edge_index, edge_weight) + self.b_2)
        h = self.relu(self.conv_3(h, edge_index, edge_weight) + self.b_3)
        h = self.relu(self.conv_4(h, edge_index, edge_weight) + self.b_4)  # + h_x
        h = self.relu(self.conv_5(h, edge_index, edge_weight) + self.b_5)
        if self.params.NUM_LAYERS >= 6:
            h = self.relu(self.conv_6(h, edge_index, edge_weight) + self.b_6)  # + h_x
        if self.params.NUM_LAYERS >= 7:
            h = self.relu(self.conv_7(h, edge_index, edge_weight) + self.b_7)
        if self.params.NUM_LAYERS >= 8:
            h = self.relu(self.conv_8(h, edge_index, edge_weight) + self.b_8)
        if self.params.NUM_LAYERS >= 9:
            h = self.relu(self.conv_9(h, edge_index, edge_weight) + self.b_9)
        if self.params.NUM_LAYERS >= 10:
            h = self.relu(self.conv_10(h, edge_index, edge_weight) + self.b_10)
        if self.params.NUM_LAYERS >= 11:
            h = self.relu(self.conv_11(h, edge_index, edge_weight) + self.b_11)
        h = self.linear(h)
        return h


class GCN1DConv(nn.Module):
    def __init__(self, in_dim, out_dim, in_ch, out_ch, hid_dim=128, hid_ch=8, k=5, activation=False, normalization="sym", bias=None):
        super(GCN1DConv, self).__init__()
        self.block_1 = ChebConv(input_features_cnn=in_dim,  # 14  #22
                              output_features_cnn=hid_dim,  # 12  #20
                              kernel_size=k,  # 3  #7
                              params=None,
                              in_channels=in_ch,
                              out_channels=hid_ch,
                              K=3,
                              normalization=normalization,
                              bias=bias)
        self.bias_1 = Parameter(torch.Tensor(1, hid_ch * hid_dim))
        zeros(self.bias_1)

        self.block_2 = ChebConv(input_features_cnn=hid_dim,  # 14  #22
                              output_features_cnn=out_dim,  # 12  #20
                              kernel_size=k,  # 3  #7
                              params=None,
                              in_channels=hid_ch,
                              out_channels=out_ch,
                              K=3,
                              normalization=normalization,
                              bias=bias)
        self.bias_2 = Parameter(torch.Tensor(1, out_ch * out_dim))
        zeros(self.bias_2)
        self.activation = activation
        if self.activation:
            self.relu = nn.ReLU()
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        if len(x.shape) == 3:
            x = torch.reshape(x, (-1, x.shape[2]))
        h = self.block_1(x, edge_index, edge_weight)
        h += self.bias_1
        h = self.relu(h)
        if self.activation: h = self.relu(h)
        h = self.block_2(h, edge_index, edge_weight)
        h += self.bias_2
        if self.activation: h = self.relu(h)
        return h

class GCN1DConv_big(nn.Module):
    def __init__(self, in_dim, out_dim, in_ch, out_ch, hid_dim=128, hid_ch=8, k=3, activation=False, normalization="sym", bias=None):
        super(GCN1DConv_big, self).__init__()
        self.K = k
        self.normalization = "sym"
        self.bias = None
        self.INPUT_LINEAR_LAYERS_UNITS = None
        self.NUM_LAYERS = 6


        # Modello TOP
        self.conv_1 = ChebConv(input_features_cnn=in_dim,
                               output_features_cnn=32,  # 32
                               kernel_size=5,  # 7
                               params=None,
                               in_channels=in_ch,
                               out_channels=24,
                               K=self.K,
                               normalization=self.normalization,
                               bias=self.bias, h_conv=0
                               )
        self.b_1 = Parameter(torch.Tensor(1, 32 * 24))

        self.conv_2 = ChebConv(input_features_cnn=32,  # 32
                               output_features_cnn=30,  # 28 #30
                               kernel_size=5,  # 7
                               params=None,
                               in_channels=24,
                               out_channels=32,
                               K=self.K,
                               normalization=self.normalization,
                               bias=self.bias, h_conv=1
                               )
        self.b_2 = Parameter(torch.Tensor(1, 32 * 30))
        self.conv_3 = ChebConv(input_features_cnn=30,  # 28 #30
                               output_features_cnn=28,  # 24 #28
                               kernel_size=5,  # 5 #7
                               params=None,
                               in_channels=32,
                               out_channels=64,
                               K=self.K,
                               normalization=self.normalization,
                               bias=self.bias, h_conv=1
                               )
        self.b_3 = Parameter(torch.Tensor(1, 64 * 28))
        self.conv_4 = ChebConv(input_features_cnn=28,  # 24 #28
                               output_features_cnn=26,  # 20 #26
                               kernel_size=3,  # 5
                               params=None,
                               in_channels=64,
                               out_channels=64,
                               K=self.K,
                               normalization=self.normalization,
                               bias=self.bias, h_conv=1
                               )
        self.b_4 = Parameter(torch.Tensor(1, 64 * 26))
        self.conv_5 = ChebConv(input_features_cnn=26,  # 20  #26
                               output_features_cnn=24,  # 16  #24
                               kernel_size=3,  # 5 #7
                               params=None,
                               in_channels=64,
                               out_channels=96,
                               K=self.K,
                               normalization=self.normalization,
                               bias=self.bias, h_conv=1
                               )
        self.b_5 = Parameter(torch.Tensor(1, 96 * 24))
        if self.NUM_LAYERS >= 6:
            self.conv_6 = ChebConv(input_features_cnn=24,  # 16   #24
                                   output_features_cnn=22,  # 14  #22
                                   kernel_size=3,  # 3  #7
                                   params=None,
                                   in_channels=96,
                                   out_channels=96,
                                   K=self.K,
                                   normalization=self.normalization,
                                   bias=self.bias, h_conv=1
                                   )
            self.b_6 = Parameter(torch.Tensor(1, 96 * 22))
            self.INPUT_LINEAR_LAYERS_UNITS = 96 * 22
        if self.NUM_LAYERS >= 7:
            self.conv_7 = ChebConv(input_features_cnn=22,  # 14  #22
                                   output_features_cnn=20,  # 12  #20
                                   kernel_size=3,  # 3  #7
                                   params=None,
                                   in_channels=96,
                                   out_channels=128,
                                   K=self.K,
                                   normalization=self.normalization,
                                   bias=self.bias, h_conv=2
                                   )
            self.b_7 = Parameter(torch.Tensor(1, 128 * 20))
            self.INPUT_LINEAR_LAYERS_UNITS = 128 * 20
        if self.NUM_LAYERS >= 8:
            self.conv_8 = ChebConv(input_features_cnn=20,  # 14  #22
                                   output_features_cnn=18,  # 12  #20
                                   kernel_size=3,  # 3  #7
                                   params=None,
                                   in_channels=128,
                                   out_channels=192,
                                   K=self.K,
                                   normalization=self.normalization,
                                   bias=self.bias, h_conv=2
                                   )
            self.b_8 = Parameter(torch.Tensor(1, 192 * 18))
            self.INPUT_LINEAR_LAYERS_UNITS = 192 * 18
        if self.NUM_LAYERS >= 9:
            self.conv_9 = ChebConv(input_features_cnn=18,  # 14  #22
                                   output_features_cnn=16,  # 12  #20
                                   kernel_size=3,  # 3  #7
                                   params=None,
                                   in_channels=192,
                                   out_channels=256,
                                   K=self.K,
                                   normalization=self.normalization,
                                   bias=self.bias, h_conv=2
                                   )
            self.b_9 = Parameter(torch.Tensor(1, 256 * 16))
            self.INPUT_LINEAR_LAYERS_UNITS = 256 * 16
        if self.NUM_LAYERS >= 10:
            self.conv_10 = ChebConv(input_features_cnn=16,  # 14  #22
                                    output_features_cnn=14,  # 12  #20
                                    kernel_size=3,  # 3  #7
                                    params=None,
                                    in_channels=256,
                                    out_channels=320,
                                    K=self.K,
                                    normalization=self.normalization,
                                    bias=self.bias, h_conv=2
                                    )
            self.b_10 = Parameter(torch.Tensor(1, 320 * 14))
            self.INPUT_LINEAR_LAYERS_UNITS = 320 * 14
        if self.NUM_LAYERS >= 11:
            self.conv_11 = ChebConv(input_features_cnn=14,  # 14  #22
                                    output_features_cnn=12,  # 12  #20
                                    kernel_size=3,  # 3  #7
                                    params=None,
                                    in_channels=320,
                                    out_channels=320,
                                    K=self.K,
                                    normalization=self.normalization,
                                    bias=self.bias, h_conv=2
                                    )
            self.b_11 = Parameter(torch.Tensor(1, 320 * 12))
            self.INPUT_LINEAR_LAYERS_UNITS = 320 * 12

        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(self.INPUT_LINEAR_LAYERS_UNITS,
                                      out_dim)
        # self.linear_time = torch.nn.Linear(self.params.LAGS,
        #                                    self.params.OUTPUT_MLP_DIMENSION_FORECASTING)


        zeros(self.b_1)
        zeros(self.b_2)
        zeros(self.b_3)
        zeros(self.b_4)
        zeros(self.b_5)
        if self.NUM_LAYERS >= 6:
            zeros(self.b_6)
        if self.NUM_LAYERS >= 7:
            zeros(self.b_7)
        if self.NUM_LAYERS >= 8:
            zeros(self.b_8)
        if self.NUM_LAYERS >= 9:
            zeros(self.b_9)
        if self.NUM_LAYERS >= 10:
            zeros(self.b_10)
        if self.NUM_LAYERS >= 11:
            zeros(self.b_11)


    def forward(self, x, edge_index, edge_weight=None):
        h_x = self.relu(self.conv_1(x, edge_index, edge_weight) + self.b_1)
        h = self.relu(self.conv_2(h_x, edge_index, edge_weight) + self.b_2)
        h = self.relu(self.conv_3(h, edge_index, edge_weight) + self.b_3)
        h = self.relu(self.conv_4(h, edge_index, edge_weight) + self.b_4)  # + h_x
        h = self.relu(self.conv_5(h, edge_index, edge_weight) + self.b_5)
        if self.NUM_LAYERS >= 6:
            h = self.relu(self.conv_6(h, edge_index, edge_weight) + self.b_6)  # + h_x
        if self.NUM_LAYERS >= 7:
            h = self.relu(self.conv_7(h, edge_index, edge_weight) + self.b_7)
        if self.NUM_LAYERS >= 8:
            h = self.relu(self.conv_8(h, edge_index, edge_weight) + self.b_8)
        if self.NUM_LAYERS >= 9:
            h = self.relu(self.conv_9(h, edge_index, edge_weight) + self.b_9)
        if self.NUM_LAYERS >= 10:
            h = self.relu(self.conv_10(h, edge_index, edge_weight) + self.b_10)
        if self.NUM_LAYERS >= 11:
            h = self.relu(self.conv_11(h, edge_index, edge_weight) + self.b_11)
        h = self.linear(h)
        return h
