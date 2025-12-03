import torch
import torch.nn as nn
import torch.nn.functional as F


def parallel_scan_log(log_coeffs, log_values):
    # log_coeffs: (batch_size, seq_len, input_size)
    # log_values: (batch_size, seq_len + 1, input_size)
    a_star = F.pad(torch.cumsum(log_coeffs, dim=-1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)


def g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))




class MiniLSTMParallelCell(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MiniLSTMParallelCell, self).__init__()
        self.linear_i = nn.Linear(dim_in, dim_out)
        self.linear_f = nn.Linear(dim_in, dim_out)
        self.linear_h = nn.Linear(dim_in, dim_out)

    def forward(self, x, h_0):
        # x: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)
        diff = F.softplus(-self.linear_f(x)) - F.softplus(-self.linear_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = log_g(h_0)
        log_tilde_h = log_g(self.linear_h(x))
        h = parallel_scan_log(log_f,
                              torch.cat([log_h_0, log_i + log_tilde_h], dim=1))
        return h


# Modello LSTM con più strati utilizzando la MiniLSTMParallelCell
class MultiLayerLSTMParallel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len):
        super(MultiLayerLSTMParallel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Creazione di più strati LSTM
        self.layers = nn.ModuleList(
            [MiniLSTMParallelCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_size*seq_len, output_size)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        x = x.unsqueeze(-1)
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, 1, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        # Passaggio attraverso i vari strati LSTM
        for i, layer in enumerate(self.layers):
            # Passiamo l'intera sequenza x al layer, che elabora tutta la sequenza in parallelo
            h_0_n = F.sigmoid(layer(x, h[i]))
            h_1_n = h_0_n[:,1:,:]
            h[i] = h_1_n
            x = h_1_n  # L'intera sequenza di hidden states viene passata al prossimo livello
        x = torch.reshape(x, (batch_size, -1))
        res = self.output_layer(x)
        return res

class MiniGRUParallelCell(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MiniGRUParallelCell, self).__init__()
        self.linear_i = nn.Linear(dim_in, dim_out)
        self.linear_f = nn.Linear(dim_in, dim_out)
        self.linear_h = nn.Linear(dim_in, dim_out)
        self.linear_z = nn.Linear(dim_in, dim_out)

    def forward(self, x, h_0):
        k = self.linear_z(x)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h_0 = log_g(h_0)
        log_tilde_h = log_g(self.linear_h(x))
        h = parallel_scan_log(log_coeffs,
                              torch.cat([log_h_0, log_z + log_tilde_h], dim=1))
        return h

class MultiLayerGRUParallel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len):
        super(MultiLayerGRUParallel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Creazione di più strati LSTM
        self.layers = nn.ModuleList(
            [MiniGRUParallelCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_size*seq_len, output_size)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        x = x.unsqueeze(-1)
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, 1, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        # Passaggio attraverso i vari strati LSTM
        for i, layer in enumerate(self.layers):
            # Passiamo l'intera sequenza x al layer, che elabora tutta la sequenza in parallelo
            h_0_n = F.sigmoid(layer(x, h[i]))
            h_1_n = h_0_n[:,1:,:]
            h[i] = h_1_n
            x = h_1_n  # L'intera sequenza di hidden states viene passata al prossimo livello
        x = torch.reshape(x, (batch_size, -1))
        res = self.output_layer(x)
        return res


if __name__ == '__main__':
    # Parametri del modello
    input_size = 10
    hidden_size = 20
    num_layers = 3

    # Istanziamento e prova del modello
    model = MultiLayerLSTMParallel(input_size, hidden_size, num_layers)
    input_tensor = torch.randn(5, 15, input_size)  # (batch_size, sequence_length, input_size)

    output = model(input_tensor)
    print(output.shape)  # Dovrebbe essere (batch_size, seq_len, hidden_size)

