# code sources:
# https://github.com/karpathy/llama2.c
# https://github.com/tinygrad/tinygrad/blob/master/examples/llama.py

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from src.layer.attention import SelfAttention
from src.layer.position_embedding import PositionEmbedding
from tinygrad import Variable, nn
from tinygrad.engine.jit import TinyJit
from tinygrad.nn import Linear
from tinygrad.tensor import Tensor


class ActivationEnum(Enum):
  SILU = lambda x: x.silu()
  GELU = lambda x: x.gelu()


@dataclass
class TransformerConfig:

  dim: int
  hidden_dim: int
  n_heads: int
  n_layers: int
  norm_eps: float
  vocab_size: int
  n_kv_heads: int
  rope_theta: float
  max_seq_len: int
  use_scaled_rope: bool
  tie_embeddings: bool
  attention_bias: bool
  activation: Callable[[Tensor], Tensor] = ActivationEnum.GELU


class Transformer:

  def __init__(self, config: TransformerConfig):
    self.config = config
    self.layers = [TransformerBlock(config.dim, config.n_heads, n_kv_heads=config.n_kv_heads, norm_eps=config.norm_eps,
                                    hidden_dim=config.hidden_dim, attention_bias=config.attention_bias, activation=config.activation) for _ in range(config.n_layers)]
    self.norm = nn.RMSNorm(config.dim, config.norm_eps)
    print('vocab_size', config.vocab_size, 'dim', config.dim)
    self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
    if not config.tie_embeddings:
      self.output = nn.Linear(config.dim, config.vocab_size, bias=False)  # weight of shape: vocab_size, dim
    self.max_context = config.max_seq_len
    self.selector = Explorer()
    head_dim = self.config.dim // self.config.n_heads

    self.positional_embedding = PositionEmbedding(head_dim=head_dim, max_seq_len=self.config.max_seq_len, rope_theta=self.config.rope_theta, use_scaled_rope=self.config.use_scaled_rope)

  @TinyJit
  def predict_one(self, tokens: Tensor, start_pos: Variable) -> Tensor:
    assert tokens.shape[1] == 1, f"tokens shape {tokens.shape}"
    return self.predict(tokens, start_pos, 1)

  def predict(self, tokens: Tensor, start_pos: Variable, seqlen: int) -> Tensor:
    h = self.tok_embeddings(tokens)
    freq_cis = self.positional_embedding.get_freqs_cis(start_pos, seqlen)
    for layer in self.layers:
      h = layer(h, start_pos, freq_cis)
    h = self.norm(h)
    if self.config.tie_embeddings:
      logits = h.matmul(self.tok_embeddings.weight.T)
    else:
      logits = self.output(self.norm(h))
    return self.selector(logits[:, -1]).realize()

  def initialize_cache_if_needed(self, bsz: int):
    if self.layers[0].attention.cache_kv is None:
      for layer in self.layers:
        layer.attention.create_cache(bsz)

  def __call__(self, tokens: Tensor, start_pos: int) -> Tensor:
    bsz, seqlen = tokens.shape
    self.initialize_cache_if_needed(bsz)
    if seqlen == 1:
      return self.predict_one(tokens, Variable("start_pos", 0, self.max_context).bind(start_pos))
    else:
      return self.predict(tokens, start_pos, seqlen)
    # todo make sure all chache are inialized so jit at 0 works
    # todo if seqlen varaible is costly to jit we should divide seqlen as sum of closest power of 2 so less kernels needs to be memorized
    # check if cache are inialized
    # return self.predict(tokens, Variable("start_pos", 0, self.max_context).bind(start_pos), Variable("seqlen", 1, self.max_context).bind(seqlen))
    return self.predict(tokens, Variable("start_pos", 0, self.max_context).bind(start_pos))

class FeedForward:
  def __init__(self, dim: int, hidden_dim: int, activation: Callable[[Tensor], Tensor]) -> None:
    self.w1: nn.Linear = Linear(dim, hidden_dim, bias=False)
    self.w2 = Linear(hidden_dim, dim, bias=False)
    self.w3 = Linear(dim, hidden_dim, bias=False)
    self.dim: int = dim
    self.hidden_dim = hidden_dim
    self.activation = activation

  def __call__(self, x: Tensor) -> Tensor:
    return self.w2(self.activation(self.w1(x)) * self.w3(x))


class TransformerBlock:
  def __init__(self, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int, norm_eps: float, attention_bias: bool, activation: Callable[[Tensor], Tensor]):
    self.attention = SelfAttention(dim, n_heads, n_kv_heads, attention_bias)
    self.feed_forward = FeedForward(dim, hidden_dim, activation)
    self.attention_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)

  def __call__(self, x: Tensor, start_pos: int, freq_cis: Tensor):
    h = x + self.attention(self.attention_norm(x), start_pos, freq_cis)
    return h + self.feed_forward(self.ffn_norm(h))


selector: Callable[[Tensor], Tensor] = lambda x: x.argmax(-1, keepdim=True)


class MaxLogitsSelector:
  def __call__(self, logits: Tensor) -> Tensor:
    return logits.argmax(-1, keepdim=True)


class Explorer:
  def __call__(self, logits: Tensor) -> Tensor:
    # prob = -logits.softmax(axis=-1)
    # shape: bsz, vocab_size
    # cpu = prob.numpy()
    # sort rows
    # Sort each row and get the indices
    # sorted_indices = np.argsort(cpu, axis=-1)

    # Use the indices to get the sorted matrix
    # sorted_matrix = np.take_along_axis(cpu, sorted_indices, axis=-1)
    # print(-sorted_matrix[0, :20])

    return logits.argmax(-1, keepdim=True)

