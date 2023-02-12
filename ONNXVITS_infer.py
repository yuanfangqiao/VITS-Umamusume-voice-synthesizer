import torch
import commons
import models

import math
from torch import nn
from torch.nn import functional as F

import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      emotion_embedding):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.emotion_embedding = emotion_embedding
    
    if self.n_vocab!=0:
      self.emb = nn.Embedding(n_vocab, hidden_channels)
      if emotion_embedding:
        self.emo_proj = nn.Linear(1024, hidden_channels)
      nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, emotion_embedding=None):
    if self.n_vocab!=0:
      x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    if emotion_embedding is not None:
      print("emotion added")
      x = x + self.emo_proj(emotion_embedding.unsqueeze(1))
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask

class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask

class SynthesizerTrn(models.SynthesizerTrn):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_speakers=0,
    gin_channels=0,
    use_sdp=True,
    emotion_embedding=False,
    **kwargs):

    super().__init__(    
      n_vocab,
      spec_channels,
      segment_size,
      inter_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      resblock, 
      resblock_kernel_sizes, 
      resblock_dilation_sizes, 
      upsample_rates, 
      upsample_initial_channel, 
      upsample_kernel_sizes,
      n_speakers=n_speakers,
      gin_channels=gin_channels,
      use_sdp=use_sdp,
      **kwargs
    )
    self.enc_p = TextEncoder(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        emotion_embedding)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)

  def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None, emotion_embedding=None):
    from ONNXVITS_utils import runonnx

    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, emotion_embedding)

    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    #logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    logw = runonnx("ONNX_net/dp.onnx", x=x.numpy(), x_mask=x_mask.numpy(), g=g.numpy())
    logw = torch.from_numpy(logw[0])

    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    
    #z = self.flow(z_p, y_mask, g=g, reverse=True)
    z = runonnx("ONNX_net/flow.onnx", z_p=z_p.numpy(), y_mask=y_mask.numpy(), g=g.numpy())
    z = torch.from_numpy(z[0])

    #o = self.dec((z * y_mask)[:,:,:max_len], g=g)
    o = runonnx("ONNX_net/dec.onnx", z_in=(z * y_mask)[:,:,:max_len].numpy(), g=g.numpy())
    o = torch.from_numpy(o[0])

    return o, attn, y_mask, (z, z_p, m_p, logs_p)

  def predict_duration(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None,
              emotion_embedding=None):
    from ONNXVITS_utils import runonnx

    #x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    x, m_p, logs_p, x_mask = runonnx("ONNX_net/enc_p.onnx", x=x.numpy(), x_lengths=x_lengths.numpy())
    x = torch.from_numpy(x)
    m_p = torch.from_numpy(m_p)
    logs_p = torch.from_numpy(logs_p)
    x_mask = torch.from_numpy(x_mask)

    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    #logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    logw = runonnx("ONNX_net/dp.onnx", x=x.numpy(), x_mask=x_mask.numpy(), g=g.numpy())
    logw = torch.from_numpy(logw[0])

    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    return list(w_ceil.squeeze())

  def infer_with_duration(self, x, x_lengths, w_ceil, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None,
              emotion_embedding=None):
    from ONNXVITS_utils import runonnx

    #x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    x, m_p, logs_p, x_mask = runonnx("ONNX_net/enc_p.onnx", x=x.numpy(), x_lengths=x_lengths.numpy())
    x = torch.from_numpy(x)
    m_p = torch.from_numpy(m_p)
    logs_p = torch.from_numpy(logs_p)
    x_mask = torch.from_numpy(x_mask)

    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None
    assert len(w_ceil) == x.shape[2]
    w_ceil = torch.FloatTensor(w_ceil).reshape(1, 1, -1)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    
    #z = self.flow(z_p, y_mask, g=g, reverse=True)
    z = runonnx("ONNX_net/flow.onnx", z_p=z_p.numpy(), y_mask=y_mask.numpy(), g=g.numpy())
    z = torch.from_numpy(z[0])

    #o = self.dec((z * y_mask)[:,:,:max_len], g=g)
    o = runonnx("ONNX_net/dec.onnx", z_in=(z * y_mask)[:,:,:max_len].numpy(), g=g.numpy())
    o = torch.from_numpy(o[0])

    return o, attn, y_mask, (z, z_p, m_p, logs_p)

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    from ONNXVITS_utils import runonnx
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    # z_p = self.flow(z, y_mask, g=g_src)
    z_p = runonnx("ONNX_net/flow.onnx", z_p=z.numpy(), y_mask=y_mask.numpy(), g=g_src.numpy())
    z_p = torch.from_numpy(z_p[0])
    # z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    z_hat = runonnx("ONNX_net/flow.onnx", z_p=z_p.numpy(), y_mask=y_mask.numpy(), g=g_tgt.numpy())
    z_hat = torch.from_numpy(z_hat[0])
    # o_hat = self.dec(z_hat * y_mask, g=g_tgt)
    o_hat = runonnx("ONNX_net/dec.onnx", z_in=(z_hat * y_mask).numpy(), g=g_tgt.numpy())
    o_hat = torch.from_numpy(o_hat[0])
    return o_hat, y_mask, (z, z_p, z_hat)