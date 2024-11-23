import random
from typing import List, Optional, Tuple

import tinygrad.nn as nn
from src.engine import Engine
from src.layers import (
    MLP,
    AttentionLayer,
    AttentionLayers,
    GradientChoker,
    NonLinear,
    SelfAttention,
)
from src.loss import log_loss
from tinygrad.dtype import dtypes
from tinygrad.nn import optim
from tinygrad.tensor import Tensor

MAX_NUMBER = 200
DIMENSIONS = 16


class SuperpositionAttention:
  # mixes quantified things 1 single superposition.
  def __init__(self, input_dim: int, heads: int):
    self.input_dim = input_dim
    self.heads = heads
    self.query = Tensor.normal(1, input_dim)
    self.key_self_attention = SelfAttention(input_dim, heads)
    self.cross_attention = AttentionLayer(input_dim, heads)

  def __call__(self, positions: Tensor, key_tokens: Tensor, key_mask: Optional[Tensor] = None, name: str = ''):
    # mix keys
    tokens = key_tokens.shape[1]
    positions = positions[:tokens, :].unsqueeze(0).repeat(key_tokens.shape[0], 1, 1)

    key_tokens, positions = self.key_self_attention(tokens=key_tokens, positions=positions, key_mask=key_mask)
    # cross attention
    return self.cross_attention(query_tokens=self.query.unsqueeze(0).repeat(positions.shape[0], 1, 1), key_tokens=key_tokens, key_mask=key_mask)


class QuantizationAttention:
  # assuming lenght of input is single superposition vector which needs to factorized into n output_factors
  def __init__(self, input_dim: int, heads: int, output_factors: int):
    self.input_dim = input_dim
    self.heads = heads
    self.query = Tensor.normal(output_factors, input_dim)
    self.cross_attention = AttentionLayer(input_dim, heads)
    self.query_self_attention = SelfAttention(input_dim, heads)

  def __call__(self, positions: Tensor, key_tokens: Tensor, key_mask: Optional[Tensor] = None, name: str = ''):
    # cross attention
    query_tokens = self.cross_attention(query_tokens=self.query.unsqueeze(0).repeat(positions.shape[0], 1, 1), key_tokens=key_tokens, key_mask=key_mask)
    # mix query
    query_tokens, positions = self.query_self_attention(tokens=query_tokens, positions=positions, key_mask=key_mask)
    return query_tokens


class QueryAttention:
  def __init__(self, input_dim: int, heads: int, output_factors: int):
    # quanta keys to superposition query
    # superposition keys to quant decoded keys
    self.input_dim = input_dim
    self.heads = heads
    self.superposition_attention = SuperpositionAttention(input_dim, heads)
    self.quantization_attention = QuantizationAttention(input_dim, heads, output_factors)

  def encode(self, positions: Tensor, key_tokens: Tensor, key_mask: Optional[Tensor] = None, name: str = ''):
    return self.superposition_attention(positions, key_tokens, key_mask=key_mask, name=name)

  def decode(self, positions: Tensor, key_tokens: Tensor, key_mask: Optional[Tensor] = None, name: str = ''):
    return self.quantization_attention(positions, key_tokens, key_mask=key_mask, name=name)


class NumberEmbedding:

  def __init__(self):
    # encodes -MAX to MAX
    self.vocab_size = MAX_NUMBER * 2 + 1
    self.number_embedding = nn.Embedding(10, DIMENSIONS)
    self.sign_embedding = nn.Embedding(2, DIMENSIONS)

    # self.embedding_encoder = MLP(4 * DIMENSIONS, DIMENSIONS)
    # self.embedding_decoder = MLP(DIMENSIONS, 4 * DIMENSIONS)
    self.embedding_attention_coder = QueryAttention(DIMENSIONS, 4, 4)  # If tokens are fixed mlp is simplified attention where positions are needed
    # self.decoder_query = Tensor.normal(4, DIMENSIONS)

    # add_sub_loss
    self.decoder_a_minus_b = MLP(2 * DIMENSIONS, DIMENSIONS)
    self.decoder_a_plus_b = MLP(2 * DIMENSIONS, DIMENSIONS)

    # min_max_loss
    self.min_max_query = Tensor.normal(2, DIMENSIONS)
    self.min_max_attention = AttentionLayers(DIMENSIONS, 1, 2)

    # sort_loss
    # TODO: This should be length 1 and rest should be position embedding
    self.sort_query = Tensor.normal(20, DIMENSIONS)
    self.sort_attention = AttentionLayers(DIMENSIONS, 2, 4)


  def correct_embedding(self, positions: Tensor) -> Tensor:
    # sign, lsb to msb should be representation of a number
    # or if we use msb to lsb then we should have ability to flow both ways

    # assume number to be less then 3 digits
    sign = self.sign_embedding(positions.lt(0).where(0, 1))
    positions = positions.abs().cast(dtypes.int32)
    embedding = Tensor.stack(sign,
                            self.number_embedding(positions - (positions // 10) * 10),
                            self.number_embedding(positions // 10 - (positions // 100) * 10),
                            self.number_embedding(positions // 100 - (positions // 1000) * 10),
                            dim=-2)
    batch, tokens, _, _ = embedding.shape
    embedding = embedding.reshape(batch * tokens, embedding.shape[-2], DIMENSIONS)
    embedding = self.embedding_attention_coder.encode(positions=self.number_embedding.weight, key_tokens=embedding)
    embedding = embedding.reshape(batch, tokens, DIMENSIONS)
    return embedding

  def __call__(self, positions: Tensor) -> Tuple[Tensor, List[Tensor]]:
    estimate_emb = self.correct_embedding(positions)
    estimate_emb = GradientChoker.apply(estimate_emb, level=4)
    return estimate_emb, self.loss(estimate_emb, positions)

  def loss(self, emb: Tensor, positions: Tensor) -> List[Tensor]:
    reverse_emb = GradientChoker.apply(self.embedding.weight.transpose(), level=8)
    return self.add_sub_loss(emb, positions, reverse_emb) + self.min_max_loss(emb, positions, reverse_emb) + self.sort_loss(emb, positions, reverse_emb)

  def add_sub_loss(self, emb: Tensor, positions: Tensor, reverse_emb: Tensor) -> List[Tensor]:
    emb_i = emb.unsqueeze(1).repeat(1, positions.shape[1], 1, 1)  # t1, t2, t3, t1, t2, t3, t1, t2, t3
    emb_j = emb.unsqueeze(2).repeat(1, 1, positions.shape[1], 1)  # t1, t1, t1, t2, t2, t2, t3, t3, t3
    corr_i_minus_j = positions.unsqueeze(2) - positions.unsqueeze(1) + MAX_NUMBER
    corr_j_minus_i = positions.unsqueeze(1) - positions.unsqueeze(2) + MAX_NUMBER
    corr_i_plus_j = positions.unsqueeze(2) + positions.unsqueeze(1) - MAX_NUMBER
    corr_j_plus_i = positions.unsqueeze(1) + positions.unsqueeze(2) - MAX_NUMBER
    i_minus_j = NonLinear.apply(self.decoder_a_minus_b(Tensor.cat(emb_j, emb_i, dim=-1))).matmul(reverse_emb)
    j_minus_i = NonLinear.apply(self.decoder_a_minus_b(Tensor.cat(emb_i, emb_j, dim=-1))).matmul(reverse_emb)
    i_plus_j = NonLinear.apply(self.decoder_a_plus_b(Tensor.cat(emb_j, emb_i, dim=-1))).matmul(reverse_emb)
    j_plus_i = NonLinear.apply(self.decoder_a_plus_b(Tensor.cat(emb_i, emb_j, dim=-1))).matmul(reverse_emb)
    if not Tensor.training:
      for i in range(1):
        for j in range(1, 2):
          print(f'{positions[0, i].item() - MAX_NUMBER},{positions[0, j].item() - MAX_NUMBER}:{i_minus_j[0, i, j].argmax().item() - MAX_NUMBER}@{corr_i_minus_j[0, i, j].item() - MAX_NUMBER} {j_minus_i[0, i, j].argmax().item() - MAX_NUMBER}@{corr_j_minus_i[0, i, j].item() - MAX_NUMBER} {i_plus_j[0, i, j].argmax().item() - MAX_NUMBER}@{corr_i_plus_j[0, i, j].item() - MAX_NUMBER} {j_plus_i[0, i, j].argmax().item() - MAX_NUMBER}@{corr_j_plus_i[0, i, j].item() - MAX_NUMBER}')

    def get_loss(logits, positions: Tensor, print_str: str) -> Tensor:
      mask = (positions < self.vocab_size).mul(positions >= 0)
      return log_loss(logits=logits, classes=positions, mask=mask, print_str=print_str)
    return [get_loss(i_minus_j, corr_i_minus_j, 'i_minus_j'),
            get_loss(j_minus_i, corr_j_minus_i, 'j_minus_i'),
            get_loss(i_plus_j, corr_i_plus_j, 'i_plus_j'),
            get_loss(j_plus_i, corr_j_plus_i, 'j_plus_i')]

  def min_max_loss(self, emb: Tensor, positions: Tensor, reverse_emb: Tensor) -> List[Tensor]:
    min = positions.min(axis=-1)
    max = positions.max(axis=-1)
    logits = self.min_max_attention(key_tokens=emb, query_tokens=self.min_max_query.unsqueeze(0).repeat(emb.shape[0], 1, 1), name='min_max').matmul(reverse_emb)
    if not Tensor.training:
      print(f'min:{logits[0, 0, :].argmax(axis=-1).item()}@{min[0].numpy()} max:{logits[0, 1, :].argmax(axis=-1).item()}@{max[0].numpy()} {positions[0, :].numpy()}')
    return [log_loss(logits=logits[:, 0, :], classes=min, print_str='min'), log_loss(logits=logits[:, 1, :], classes=max, print_str='max')]

  def sort_loss(self, emb: Tensor, positions: Tensor, reverse_emb: Tensor) -> List[Tensor]:
    assert positions.shape[-1] <= self.sort_query.shape[0], f"positions shape {positions.shape} must be less than sort_query shape {self.sort_query.shape}"
    # query_mask = Tensor.ones_like(positions).pad((None, (0, self.sort_query.shape[0] - positions.shape[-1])))
    logits = self.sort_attention(key_tokens=emb, query_tokens=self.sort_query.shrink(((0, positions.shape[-1]), None)).unsqueeze(0).repeat(emb.shape[0], 1, 1), name='sort').matmul(reverse_emb)
    # assuming positions is always sorted
    if not Tensor.training:
      print(f'sort:{logits[0, :].argmax(axis=-1).numpy()} {positions[0, :].numpy()}')
    return [log_loss(logits=logits, classes=positions, print_str='sort')]


class NumberEngine(Engine):
  def __init__(self):
    super().__init__(model_class=NumberEmbedding, optimizer_class=optim.Adam)

  def single_batch(self, batch_size: int, i: int):
    if i % 3 == 0:
      start = [random.randint(-40, 30) for _ in range(batch_size)]
      return Tensor([[x for x in range(s, s + 10)] for s in start])
    else:
      return Tensor([sorted([random.randint(-40, 40) for _ in range(10)]) for _ in range(batch_size)])


if __name__ == "__main__":
  engine = NumberEngine()
  engine()
