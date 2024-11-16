from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generator, List, Union

from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from models.layers.transformer import Transformer


class Speaker(Enum):
    USER = 0
    ASSISTANT = 1
    SEARCH = 2


class ConversationTree:
  # TODO: may also store model cache as heavy-light decompositions

  @dataclass
  class Node:
    parent: int
    children: List[int]
    data: List[int]
    speaker: Speaker

  def __init__(self):
    self.trees: List[ConversationTree.Node] = []
    self.current_node_id: int = -1

  def add_node(self, data: List[int], speaker: Speaker):
    self.trees.append(ConversationTree.Node(self.current_node_id, [], data, speaker))
    child_id = len(self.trees) - 1
    self.trees[self.current_node_id].children.append(child_id)
    self.current_node_id = child_id
    return child_id


class Tokenizer:
  @abstractmethod
  def encode(self, inp: str) -> List[int]:
    pass

  @abstractmethod
  def decode(self, token: int) -> Union[str, None]:
    pass


class LanguageModel:  # this is generic language model, and leaves individual models to be implemented

  def __init__(self, model: Transformer, tokenizer: Tokenizer):
    self.model = model
    self.start_pos: int = 0
    self.conversation = ConversationTree()
    self.tokenizer = tokenizer
    Tensor.no_grad = True

  def process(self, inp: str) -> Generator[str, None, None]:
    tokens = self.tokenizer.encode(inp)
    self.conversation.add_node(tokens, Speaker.USER)
    x = Tensor([tokens], dtype=dtypes.int32)
    length = len(tokens)
    for _ in range(500):
      x = self.model(x, self.start_pos)
      print(x)
      self.start_pos += length
      length = 1
      op: int = int(x.item())
      self.conversation.add_node([op], Speaker.ASSISTANT)
      str_op = self.tokenizer.decode([op])
      print(str_op)
      if str_op:
        yield str_op
      else:
        break
