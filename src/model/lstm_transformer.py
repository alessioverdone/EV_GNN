import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ---------------------------------------------------------------------------
# Standard LSTM
# ---------------------------------------------------------------------------

class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len, horizon, dropout=0.0):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.output_layer = nn.Linear(hidden_size * seq_len, output_size * horizon)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        # x: (B*N, T, F)
        out, _ = self.lstm(x)           # (B*N, T, hidden_size)
        return self.output_layer(out.reshape(out.size(0), -1))


# ---------------------------------------------------------------------------
# Temporal Transformer
# ---------------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, d_ff,
                 output_size, seq_len, horizon, dropout=0.1):
        super().__init__()
        d_model = max((d_model // nhead) * nhead, nhead)
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.output_layer = nn.Linear(d_model * seq_len, output_size * horizon)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        # x: (B*N, T, F)
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x)             # (B*N, T, d_model)
        return self.output_layer(x.reshape(x.size(0), -1))


# ---------------------------------------------------------------------------
# Informer  (Zhou et al. 2021) — ProbSparse attention + distilling
# ---------------------------------------------------------------------------

class ProbAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        H, d = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)   # (B, H, L, d)
        K = self.k_proj(x).view(B, L, H, d).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, d).transpose(1, 2)

        # Sample k keys to measure query sparsity
        sample_k = min(max(int(self.factor * math.log(L + 1)), 1), L)
        n_top    = min(max(int(self.factor * math.log(L + 1)), 1), L)

        idx = torch.randint(L, (L, sample_k), device=x.device)
        K_samp = K[:, :, idx.reshape(-1)].reshape(B, H, L, sample_k, d)
        QK_samp = torch.einsum('bhqd,bhqsd->bhqs', Q, K_samp)  # (B, H, L, sample_k)
        M = QK_samp.max(-1).values - QK_samp.mean(-1)           # (B, H, L)
        top_idx = M.topk(n_top, dim=-1).indices                  # (B, H, n_top)

        brange = torch.arange(B, device=x.device)[:, None, None]
        hrange = torch.arange(H, device=x.device)[None, :, None]

        Q_sparse = Q[brange, hrange, top_idx]                    # (B, H, n_top, d)
        scores   = torch.einsum('bhqd,bhkd->bhqk', Q_sparse, K) * self.scale
        attn     = self.dropout(scores.softmax(-1))
        sparse_out = torch.einsum('bhqk,bhkd->bhqd', attn, V)   # (B, H, n_top, d)

        # Lazy approximation: non-top queries get mean(V)
        context = V.mean(2, keepdim=True).expand(B, H, L, d).clone()
        context[brange, hrange, top_idx] = sparse_out

        return self.out_proj(context.transpose(1, 2).reshape(B, L, -1))


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, factor=5):
        super().__init__()
        self.attn   = ProbAttention(d_model, n_heads, factor, dropout)
        self.ff1    = nn.Linear(d_model, d_ff)
        self.ff2    = nn.Linear(d_ff, d_model)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.distill = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        # Pre-norm attention
        x = x + self.dropout(self.attn(self.norm1(x)))
        # Pre-norm FFN
        x = x + self.dropout(self.ff2(self.dropout(F.gelu(self.ff1(self.norm2(x))))))
        # Distilling halves sequence length; skip if already length 1
        if x.size(1) > 1:
            x = self.distill(x.transpose(1, 2)).transpose(1, 2)
        return x


class InformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, n_heads, num_layers, d_ff,
                 output_size, seq_len, horizon, dropout=0.1, factor=5):
        super().__init__()
        d_model = max((d_model // n_heads) * n_heads, n_heads)
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        self.layers     = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout, factor)
            for _ in range(num_layers)
        ])
        # Track sequence length after each distilling step
        final_len = seq_len
        for _ in range(num_layers):
            if final_len <= 1:
                break
            final_len = final_len // 2
        self.output_layer = nn.Linear(d_model * max(final_len, 1), output_size * horizon)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        # x: (B*N, T, F)
        x = self.pos_enc(self.input_proj(x))
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x.reshape(x.size(0), -1))
