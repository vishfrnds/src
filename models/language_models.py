from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from time import time
from typing import Generator, List, Union

from src.layer.transformer import Transformer
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


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

  def __init__(self):
    self.trees: List[ConversationTree.Node] = []
    self.current_node_id: int = -1

  def add_node(self, data: List[int], speaker: Speaker):
    self.trees.append(ConversationTree.Node(self.current_node_id, [], data, speaker))
    child_id = len(self.trees) - 1
    self.trees[self.current_node_id].children.append(child_id)
    self.current_node_id = child_id
    return child_id


class BaseTokenizer:
  def __init__(self, dir_path: str):
    self.dir_path = dir_path

  @abstractmethod
  def input_to_tokens(self, inp: str) -> List[int]:
    raise NotImplementedError

  @abstractmethod
  def token_to_string(self, token: int) -> Union[str, None]:
    raise NotImplementedError


class LanguageModel:  # this is generic language model, and leaves individual models to be implemented

  def __init__(self, model: Transformer, tokenizer: BaseTokenizer):
    self.model = model
    self.start_pos: int = 0
    self.conversation = ConversationTree()
    self.tokenizer = tokenizer
    Tensor.no_grad = True

  def process(self, inp: str) -> Generator[str, None, None]:
    tokens = self.tokenizer.input_to_tokens(inp)
    # self.conversation.add_node(tokens, Speaker.USER)
    x = Tensor([tokens], dtype=dtypes.int32)
    x = self.model(x, self.start_pos)
    self.start_pos += len(tokens)
    while True:
      op: List[int] = x.tolist()[0]
      x = x[:, -1:]
      self.start_pos += len(op)
      # print('op', op)
      # self.conversation.add_node([op], Speaker.ASSISTANT)
      str_op = ''.join([self.tokenizer.token_to_string(o) for o in op])
      if str_op:
        yield str_op
      else:
        break
      # print(f'x {x.tolist()} start_pos {self.start_pos}')
      x = self.model(x, self.start_pos)
      start = time()

  def run(self):
    print("Welcome to the chat bot! Press Ctrl+Enter to send a message.")
    while True:
      try:
        user_input = ""
        while True:
          line = input()
          if line == "":
            break
          user_input += line + "\n"
        
        print("User: " + user_input)
        bot_output = self.process(user_input)
        print("Assistant: ", end="")
        for output in bot_output:
          print(output, end="", flush=True)
        print()
      except KeyboardInterrupt:
        print("\nExiting chat bot. Goodbye!")
        break
