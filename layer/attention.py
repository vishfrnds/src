from src.layer.position_embedding import PositionEmbedding
from tinygrad.nn import Linear
from tinygrad.tensor import Tensor


class SelfAttention:
  # Causal attention
  # supports grouped key, value heads
  # supports cache

  # TODO: flash attention

  def __init__(self, dim: int, n_heads: int, n_kv_heads: int, attention_bias: bool) -> None:
    self.n_heads: int = n_heads  # 8
    self.n_kv_heads: int = n_kv_heads
    self.head_dim: int = dim // n_heads  # 64/8 = 8
    self.n_rep: int = self.n_heads // self.n_kv_heads  # 2

    self.wq = Linear(dim, self.n_heads * self.head_dim, bias=attention_bias)
    self.wk = Linear(dim, self.n_kv_heads * self.head_dim, bias=attention_bias)
    self.wv = Linear(dim, self.n_kv_heads * self.head_dim, bias=attention_bias)
    self.wo = Linear(self.n_heads * self.head_dim, dim, bias=False)
    self.max_context = 5000
    self.cache_kv = None

  def create_cache(self, bsz: int):
    self.cache_kv = Tensor.zeros(2, bsz, self.max_context, self.n_kv_heads, self.head_dim, requires_grad=False).contiguous().realize()

  @staticmethod
  def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
      return x
    return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

  def __call__(self, x: Tensor, start_pos: int, freq_cis: Tensor) -> Tensor:
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

    bsz, seq_len, n_heads, head_dim = xq.shape

    xq, xk = PositionEmbedding.apply_rotary_emb(freq_cis, xq, xk)

    # create kv cache
      # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded

    # print(self.cache_kv.dtype, x.dtype, xk.dtype, xv.dtype, self.wv.weight.dtype, self.wk.weight.dtype)
    # print(self.cache_kv.device, x.device, xk.device, xv.device, self.wv.weight.device, self.wk.weight.device)
    self.cache_kv.shrink((None, None, (start_pos, start_pos + seq_len), None, None)).assign(Tensor.stack(xk, xv)).realize()
    keys = self.cache_kv[0].shrink((None, (0, start_pos + seq_len), None, None))
    values = self.cache_kv[1].shrink((None, (0, start_pos + seq_len), None, None))

    # update the cache
    # we can not update with cache = ... As this does not work in jit mode hence need to introduce max_context

    keys, values = self.repeat_kv(keys, self.n_rep), self.repeat_kv(values, self.n_rep)

    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    mask = Tensor.full((1, 1, seq_len, start_pos + seq_len), float("-inf"), dtype=x.dtype).triu(start_pos + 1).realize() if seq_len > 1 else None
    attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2).reshape(bsz, seq_len, -1)
    return self.wo(attn)
