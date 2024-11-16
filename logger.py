import time
from typing import Dict, List

import numpy as np
from tensorboard.summary import Writer

from tinygrad.tensor import Tensor


class Logger:
  # TODO: This does not work with jit
  # Each class can also return a log, it will be map of name, tensor
  def __init__(self):
    self.writer = Writer(f"runs/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    self.logs: Dict[str, int] = {}
    self.tensor_logs: List[Tensor] = []


  def log_state_dict(self, state_dict: Dict[str, Tensor]):
    for name, tensor in state_dict.items():
      if tensor.requires_grad:
        self.log_layer_stats(tensor, name)

  def log_layer_stats(self, tensor: Tensor, name: str):
    assert name not in self.logs or self.logs[name] == len(self.tensor_logs), f"Layer {name} not logged"

    # Log to tensorboard
    # Use a common tag prefix to group related metrics in one graph
    # tag_prefix = f'{name}/stats'
    self.logs[name] = len(self.tensor_logs)
    self.tensor_logs.extend([tensor.max(), tensor.min(), tensor.mean(), tensor.std()])

  def add_loss(self, name: str, loss: Tensor, accuracy: Tensor):
    name = f'z/{name}'
    assert name not in self.logs or self.logs[name] == len(self.tensor_logs), f"Layer {name} not logged"
    # self.writer.add_scalar(f'{name}/accuracy', accuracy, self.cur_key)
    # self.writer.add_scalar(f'{name}/loss', loss, self.cur_key)
    self.logs[name] = len(self.tensor_logs)
    self.tensor_logs.extend([loss, loss, accuracy, accuracy])

  def add(self, name: str, tensor: Tensor):
    self.log_layer_stats(tensor, name)

  def realize(self) -> Tensor:
    return Tensor.stack(*self.tensor_logs).realize()

  def print_stats(self, stats: np.ndarray, index: int):
    for name, tensor in sorted(self.logs.items(), key=lambda x: x[0]):
      print(f'{name}:\t{[f"{x:.3f}" for x in stats[tensor:tensor+4]]}')
      self.writer.add_scalar(f'{name}/max', stats[tensor], index)
      self.writer.add_scalar(f'{name}/min', stats[tensor+1], index)
      self.writer.add_scalar(f'{name}/mean', stats[tensor+2], index)
      self.writer.add_scalar(f'{name}/std', stats[tensor+3], index)
    self.writer.flush()
    print()
    self.tensor_logs = []


logger = Logger()
