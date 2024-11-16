import argparse
import json
import math
import os
import random
import sys
import time
import traceback
import unittest
from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import tinygrad.nn as nn
from src.layer.engine import Engine
from src.layer.logger import logger
from tinygrad import Variable
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.nn import Linear, optim
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.tensor import Function, Tensor

MAX_NUMBER = 200
DIMENSIONS = 16
class NumberEmbedding:

  def __init__(self):
    # encodes -MAX to MAX
    self.vocab_size = MAX_NUMBER * 2 + 1
    self.embedding = nn.Embedding(self.vocab_size, DIMENSIONS)

    # add_sub_loss
    self.decoder_a_minus_b = MLP(2 * DIMENSIONS, DIMENSIONS)
    self.decoder_a_plus_b = MLP(2 * DIMENSIONS, DIMENSIONS)

    # min_max_loss
    self.min_max_query = Tensor.normal(2, DIMENSIONS)
    self.min_max_attention = AttentionLayers(DIMENSIONS, 1, 2)

    # sort_loss
    self.sort_query = Tensor.normal(Embedding.MAX_GRID_SIZE, DIMENSIONS)
    self.sort_attention = AttentionLayers(DIMENSIONS, 2,4)

  def __call__(self, positions: Tensor) -> Tuple[Tensor, List[Tensor]]:
    positions = positions + self.MAX_NUMBER
    emb = self.embedding(positions)
    emb = GradientChoker.apply(emb, level=4)
    return emb, self.loss(emb, positions)

  def loss(self, emb: Tensor, positions: Tensor) -> List[Tensor]:
    reverse_emb = GradientChoker.apply(self.embedding.weight.transpose(), level=8)
    return self.add_sub_loss(emb, positions, reverse_emb) + self.min_max_loss(emb, positions, reverse_emb) + self.sort_loss(emb, positions, reverse_emb)


  def add_sub_loss(self, emb: Tensor, positions: Tensor, reverse_emb: Tensor) -> List[Tensor]:
    emb_i = emb.unsqueeze(1).repeat(1, positions.shape[1], 1, 1)  # t1, t2, t3, t1, t2, t3, t1, t2, t3
    emb_j = emb.unsqueeze(2).repeat(1, 1, positions.shape[1], 1)  # t1, t1, t1, t2, t2, t2, t3, t3, t3
    corr_i_minus_j = positions.unsqueeze(2) - positions.unsqueeze(1) + self.MAX_NUMBER
    corr_j_minus_i = positions.unsqueeze(1) - positions.unsqueeze(2) + self.MAX_NUMBER
    corr_i_plus_j = positions.unsqueeze(2) + positions.unsqueeze(1) - self.MAX_NUMBER
    corr_j_plus_i = positions.unsqueeze(1) + positions.unsqueeze(2) - self.MAX_NUMBER
    i_minus_j = NonLinear.apply(self.decoder_a_minus_b(Tensor.cat(emb_j, emb_i, dim=-1))).matmul(reverse_emb)
    j_minus_i = NonLinear.apply(self.decoder_a_minus_b(Tensor.cat(emb_i, emb_j, dim=-1))).matmul(reverse_emb)
    i_plus_j = NonLinear.apply(self.decoder_a_plus_b(Tensor.cat(emb_j, emb_i, dim=-1))).matmul(reverse_emb)
    j_plus_i = NonLinear.apply(self.decoder_a_plus_b(Tensor.cat(emb_i, emb_j, dim=-1))).matmul(reverse_emb)
    if not Tensor.training:
      for i in range(1):
        for j in range(1, 2):
          print(f'{positions[0, i].item() - self.MAX_NUMBER},{positions[0, j].item() - self.MAX_NUMBER}:{i_minus_j[0, i, j].argmax().item() - self.MAX_NUMBER}@{corr_i_minus_j[0, i, j].item() - self.MAX_NUMBER} {j_minus_i[0, i, j].argmax().item() - self.MAX_NUMBER}@{ corr_j_minus_i[0, i, j].item() - self.MAX_NUMBER} {i_plus_j[0, i, j].argmax().item() - self.MAX_NUMBER}@{corr_i_plus_j[0, i, j].item() - self.MAX_NUMBER} {j_plus_i[0, i, j].argmax().item() - self.MAX_NUMBER}@{corr_j_plus_i[0, i, j].item() - self.MAX_NUMBER}')
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
