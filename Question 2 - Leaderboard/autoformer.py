"""Compact Autoformer (Wu et al., 2021, "Autoformer: Decomposition Transformers with
Auto-Correlation for Long-Term Series Forecasting").

Written from scratch following the paper (sections 3.1-3.2) and the structure of the authors'
reference code (github.com/thuml/Autoformer: layers/Autoformer_EncDec.py,
layers/AutoCorrelation.py, models/Autoformer.py). Differences from the reference:

* Delay aggregation always selects top-k delays per example (the reference shares delays
  across the batch during training) and retrieves v[(t - tau) mod L], the same convention as
  `aggregate_delays` in Task 1, so that R(tau) = sum_t q_t k_(t-tau) and the value read back
  comes from the key position that scored.
* Covariates ("marks") are arbitrary real-valued vectors embedded by a linear layer, instead of
  calendar features, because this series has no timestamps.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class MovingAvg(nn.Module):
    """Centered moving average with replicate padding, applied along time of [B, L, C]."""

    def __init__(self, kernel):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        self.kernel = kernel

    def forward(self, x):
        half = self.kernel // 2
        padded = torch.cat([x[:, :1].expand(-1, half, -1), x,
                            x[:, -1:].expand(-1, half, -1)], dim=1)
        return F.avg_pool1d(padded.transpose(1, 2), self.kernel, stride=1).transpose(1, 2)


class SeriesDecomp(nn.Module):
    """x -> (seasonal = x - trend, trend)."""

    def __init__(self, kernel):
        super().__init__()
        self.moving_avg = MovingAvg(kernel)

    def forward(self, x):
        trend = self.moving_avg(x)
        return x - trend, trend


class SeasonalLayerNorm(nn.Module):
    """LayerNorm followed by removal of the per-sequence mean (reference: my_Layernorm)."""

    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        x = self.norm(x)
        return x - x.mean(dim=1, keepdim=True)


class AutoCorrelation(nn.Module):
    """Period-based dependency discovery (FFT) and time-delay aggregation."""

    def __init__(self, factor=1, dropout=0.0):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):                 # [B, L, H, E], [B, S, H, E], [B, S, H, E]
        B, L, H, E = q.shape
        S = k.shape[1]
        if L > S:                               # pad keys/values with zeros up to L
            pad = torch.zeros(B, L - S, H, E, device=q.device, dtype=q.dtype)
            k, v = torch.cat([k, pad], 1), torch.cat([v, pad], 1)
        else:                                   # truncate to the query length
            k, v = k[:, :L], v[:, :L]
        q, k, v = (t.permute(0, 2, 3, 1) for t in (q, k, v))  # [B, H, E, L]
        corr = torch.fft.irfft(torch.fft.rfft(q, dim=-1) * torch.fft.rfft(k, dim=-1).conj(),
                               n=L, dim=-1)     # R(tau) = sum_t q_t k_(t-tau)
        score = corr.mean(dim=(1, 2))           # [B, L], shared across heads and channels
        top_k = max(1, int(self.factor * math.log(L)))
        selected, delays = score[:, 1:].topk(top_k, dim=-1)   # ignore tau = 0
        delays = delays + 1
        weights = self.dropout(selected.softmax(-1))         # [B, K]
        positions = torch.arange(L, device=q.device)
        source = (positions.view(1, 1, L) - delays.unsqueeze(-1)) % L  # [B, K, L]
        gathered = torch.gather(v.unsqueeze(1).expand(-1, top_k, -1, -1, -1), -1,
                                source.view(B, top_k, 1, 1, L).expand(-1, -1, H, E, -1))
        out = (weights.view(B, top_k, 1, 1, 1) * gathered).sum(1)  # [B, H, E, L]
        return out.permute(0, 3, 1, 2)          # [B, L, H, E]


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d, heads, factor, dropout):
        super().__init__()
        self.heads = heads
        self.correlation = AutoCorrelation(factor, dropout)
        self.query, self.key, self.value, self.out = (nn.Linear(d, d) for _ in range(4))

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        S = keys.shape[1]
        H = self.heads
        q = self.query(queries).view(B, L, H, -1)
        k = self.key(keys).view(B, S, H, -1)
        v = self.value(values).view(B, S, H, -1)
        return self.out(self.correlation(q, k, v).reshape(B, L, -1))


def feed_forward(d, d_ff, dropout):
    return nn.Sequential(nn.Conv1d(d, d_ff, 1, bias=False), nn.GELU(), nn.Dropout(dropout),
                         nn.Conv1d(d_ff, d, 1, bias=False))


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, d_ff, kernel, factor, dropout):
        super().__init__()
        self.attention = AutoCorrelationLayer(d, heads, factor, dropout)
        self.ff = feed_forward(d, d_ff, dropout)
        self.decomp1, self.decomp2 = SeriesDecomp(kernel), SeriesDecomp(kernel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.decomp1(x + self.dropout(self.attention(x, x, x)))
        y = self.dropout(self.ff(x.transpose(1, 2)).transpose(1, 2))
        x, _ = self.decomp2(x + y)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, d_ff, kernel, factor, dropout, c_out):
        super().__init__()
        self.self_attention = AutoCorrelationLayer(d, heads, factor, dropout)
        self.cross_attention = AutoCorrelationLayer(d, heads, factor, dropout)
        self.ff = feed_forward(d, d_ff, dropout)
        self.decomp1, self.decomp2, self.decomp3 = (SeriesDecomp(kernel) for _ in range(3))
        self.dropout = nn.Dropout(dropout)
        self.trend_projection = nn.Conv1d(d, c_out, 3, padding=1, padding_mode="circular",
                                          bias=False)

    def forward(self, x, cross):
        x, trend1 = self.decomp1(x + self.dropout(self.self_attention(x, x, x)))
        x, trend2 = self.decomp2(x + self.dropout(self.cross_attention(x, cross, cross)))
        y = self.dropout(self.ff(x.transpose(1, 2)).transpose(1, 2))
        x, trend3 = self.decomp3(x + y)
        trend = self.trend_projection((trend1 + trend2 + trend3).transpose(1, 2))
        return x, trend.transpose(1, 2)


class DataEmbedding(nn.Module):
    """Value embedding (circular conv) + linear covariate embedding; no positional encoding."""

    def __init__(self, c_in, n_marks, d, dropout):
        super().__init__()
        self.value = nn.Conv1d(c_in, d, 3, padding=1, padding_mode="circular", bias=False)
        self.marks = nn.Linear(n_marks, d, bias=False) if n_marks else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, marks):
        out = self.value(x.transpose(1, 2)).transpose(1, 2)
        if self.marks is not None:
            out = out + self.marks(marks)
        return self.dropout(out)


class Autoformer(nn.Module):
    def __init__(self, seq_len, label_len, pred_len, n_marks, c_in=1, d=32, heads=4,
                 d_ff=64, e_layers=1, d_layers=1, kernel=25, factor=1, dropout=0.1):
        super().__init__()
        self.label_len, self.pred_len = label_len, pred_len
        self.decomp = SeriesDecomp(kernel)
        self.enc_embedding = DataEmbedding(c_in, n_marks, d, dropout)
        self.dec_embedding = DataEmbedding(c_in, n_marks, d, dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d, heads, d_ff, kernel, factor, dropout)
                                      for _ in range(e_layers)])
        self.enc_norm = SeasonalLayerNorm(d)
        self.decoder = nn.ModuleList([DecoderLayer(d, heads, d_ff, kernel, factor, dropout, c_in)
                                      for _ in range(d_layers)])
        self.dec_norm = SeasonalLayerNorm(d)
        self.projection = nn.Linear(d, c_in)

    def forward(self, x_enc, marks_enc, marks_dec):
        # x_enc [B, seq_len, c]; marks_enc [B, seq_len, m]; marks_dec [B, label+pred, m]
        mean = x_enc.mean(1, keepdim=True).expand(-1, self.pred_len, -1)
        zeros = torch.zeros_like(mean)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend = torch.cat([trend_init[:, -self.label_len:], mean], 1)
        seasonal = torch.cat([seasonal_init[:, -self.label_len:], zeros], 1)

        enc = self.enc_embedding(x_enc, marks_enc)
        for layer in self.encoder:
            enc = layer(enc)
        enc = self.enc_norm(enc)

        dec = self.dec_embedding(seasonal, marks_dec)
        for layer in self.decoder:
            dec, residual_trend = layer(dec, enc)
            trend = trend + residual_trend
        seasonal_out = self.projection(self.dec_norm(dec))
        return (trend + seasonal_out)[:, -self.pred_len:]
