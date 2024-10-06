from typing import Union

from regex import P

import src.layers.rope as rope
from tinygrad.nn import Linear
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor


class SelfAttention:
  # Causal attention
  # supports grouped key, value heads
  # supports cache

  # TODO: flash attention

  def __init__(self, dim: int, n_heads: int, n_kv_heads: int) -> None:
    self.n_heads: int = n_heads  # 8
    self.n_kv_heads: int = n_kv_heads
    self.head_dim: int = dim // n_heads  # 64/8 = 8
    self.n_rep: int = self.n_heads // self.n_kv_heads  # 2

    self.wq = Linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wk = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = Linear(self.n_heads * self.head_dim, dim, bias=False)
    self.max_context = 5000

  @staticmethod
  def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

    # return x.reshape(bs, seqlen, n_kv_heads, 1, head_dim).expand(bs, seqlen, n_kv_heads, n_rep, head_dim).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, use_cache=True) -> Tensor:
      x = x.half()
      xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
      xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
      xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
      xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

      xq, xk = rope.apply_rotary_emb(xq, xk, freqs_cis)
      bsz, seqlen, n_heads, head_dim = xq.shape

      # create kv cache
      if use_cache:
        if not hasattr(self, "cache_kv"):
          self.cache_kv = Tensor.zeros(2, bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype).contiguous().realize()
        # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded

      print(self.cache_kv.dtype, x.dtype, xk.dtype, xv.dtype, self.wv.weight.dtype, self.wk.weight.dtype)
      self.cache_kv.shrink((None, None, (start_pos, start_pos + seqlen), None, None)).assign(Tensor.stack(xk, xv)).realize()
      keys = self.cache_kv[0].shrink((None, (0, start_pos + seqlen), None, None)) if start_pos > 0 else xk
      values = self.cache_kv[1].shrink((None, (0, start_pos + seqlen), None, None)) if start_pos > 0 else xv

      # update the cache
      # we can not update with cache = ... As this does not work in jit mode hence need to introduce max_context

      keys, values = self.repeat_kv(keys, self.n_rep), self.repeat_kv(values, self.n_rep)

      xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
      mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=x.dtype).triu(start_pos + 1).realize() if seqlen > 1 else None
      attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2).reshape(bsz, seqlen, -1)
      return self.wo(attn)
