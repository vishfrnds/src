import math
from typing import Optional

import tinygrad.nn as nn
from src.layer.nonlinear import NonLinear
from src.logger import logger
from tinygrad.tensor import Tensor


class MLP:
  def __init__(self, input_dim: int, output_dim: int):
    hidden_dim = input_dim * 2
    self.linear1 = nn.Linear(input_dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, output_dim)

  def __call__(self, x: Tensor):
    x = (self.linear1(x)).gelu().dropout(0.1)
    return self.linear2(x)

class AttentionLayer:
  def __init__(self, input_dim: int, heads: int):
    self.input_dim = input_dim
    self.heads = heads

    self.query = nn.Linear(input_dim, input_dim * heads)
    self.key = nn.Linear(input_dim, input_dim * heads)
    self.value = nn.Linear(input_dim, input_dim * heads)
    self.output = nn.Linear(input_dim * heads, input_dim)

  def __call__(self, query_tokens: Tensor, key_tokens: Tensor, key_mask: Optional[Tensor] = None):
    batch_size, q_seq_len, _ = query_tokens.shape
    batch_size, k_seq_len, _ = key_tokens.shape

    # Linear projections and split into heads
    q = self.query(query_tokens).reshape(batch_size, q_seq_len, self.heads, self.input_dim).transpose(1, 2).layernorm(axis=-1)
    k = self.key(key_tokens).reshape(batch_size, k_seq_len, self.heads, self.input_dim).transpose(1, 2).layernorm(axis=-1)
    v = self.value(key_tokens).reshape(batch_size, k_seq_len, self.heads, self.input_dim).transpose(1, 2).layernorm(axis=-1)

    # Attention for each head
    attention = (q.matmul(k.transpose(-2, -1)) / math.sqrt(self.input_dim))
    if key_mask is not None:
      key_mask = key_mask.reshape(batch_size, 1, 1, k_seq_len)  # batch, 1 (heads), 1 (seq_len), seq_len
      attention = attention + key_mask.where(0, -float('inf'))
    attention = attention.softmax(axis=-1).dropout(0.1)

    v = attention.matmul(v)

    # Combine heads
    v = v.transpose(1, 2).reshape(batch_size, q_seq_len, self.input_dim * self.heads).dropout(0.05)

    return NonLinear.apply(self.output(v))


class AttentionLayers:
  def __init__(self, input_dim: int, heads: int, num_layers: int = 1):
    self.input_dim = input_dim
    self.heads = heads
    self.num_layers = num_layers

    self.query_cross_attention = [AttentionLayer(input_dim, heads) for _ in range(num_layers)]
    # self.key_cross_attention = [AttentionLayer(input_dim, heads) for _ in range(num_layers - 1)]
    self.query_self_attention = [AttentionLayer(input_dim, heads) for _ in range(num_layers - 1)]
    # self.key_self_attention = [AttentionLayer(input_dim, heads) for _ in range(num_layers - 1)]

  def __call__(self, query_tokens: Tensor, key_tokens: Tensor, key_mask: Optional[Tensor] = None, query_mask: Optional[Tensor] = None, name: str = ''):
    for i in range(self.num_layers):
      # if i > 0:
      #   key_tokens = (key_tokens + self.key_self_attention[i - 1](key_tokens, key_tokens, key_mask=key_mask)) / 2
      #   key_tokens = self.key_cross_attention[i - 1](key_tokens, query_tokens, key_mask=query_mask)
      query_tokens = self.query_cross_attention[i](query_tokens, key_tokens, key_mask=key_mask)
      if not Tensor.training:
        logger.add(f'out/{name}/query_tokens_inter_{i}', query_tokens)
      if i < self.num_layers - 1:
        query_tokens = query_tokens + self.query_self_attention[i](query_tokens, query_tokens, key_mask=query_mask).contiguous()

      if not Tensor.training:
        logger.add(f'out/{name}/key_tokens_{i}', key_tokens)
        logger.add(f'out/{name}/query_tokens_{i}', query_tokens)
    return query_tokens