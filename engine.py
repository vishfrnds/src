import argparse
import os
import sys
from dataclasses import dataclass
from time import time

from src.logger import logger
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.tensor import Tensor


class Engine:
  @dataclass
  class Config:
    lr: float = 0.001
    batch_size: int = 8

  def parse_args(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=2)
    parser.add_argument('--save', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('-f', type=str)  # for kaggle to work
    return parser.parse_args(sys.argv[1:])

  def __init__(self, model_class: type, optimizer_class: type):
    self.args = self.parse_args()
    self.config = Engine.Config()
    if Device.DEFAULT == 'CUDA' or Device.DEFAULT == 'CLOUD':
      self.config.batch_size = 512
      print('using GPU')
    self.loss = float('inf')
    self.model = model_class()
    loss_to_filename = {int(f.split('_')[2].split('.')[0]): f for f in os.listdir()
                        if f.startswith(f'model_{self.args.version}_') and f.endswith('.pt')}
    if loss_to_filename and self.args.load:
      min_loss = min(loss_to_filename.keys())
      state_dict = safe_load(loss_to_filename[min_loss])
      # update state_dict keys
      model_state_dict = {k: {} for k in get_state_dict(self.model).keys()}
      for k in model_state_dict.keys():
        # exact match
        split = k.split('.')
        for i in range(len(split)):
          if '.'.join(split[i:]) in state_dict:
            cur = model_state_dict[k]
            if len(cur) == 0 or cur[0] > i:
              model_state_dict[k] = (i, '.'.join(split[i:]))
      for k, v in model_state_dict.items():
        if len(v) > 0:
          state_dict[k] = state_dict.pop(v[1])
      load_state_dict(self.model, state_dict, strict=False)
      print('loaded model', loss_to_filename[min_loss])
    else:
      print('no model to load')
    self.state_dict = get_state_dict(self.model)
    self.optimizer = optimizer_class(list(self.state_dict.values()), lr=self.config.lr)

  @TinyJit
  def train(self, input: Tensor) -> Tensor:
    _, losses = self.model(input)
    loss = Tensor.stack(*losses).mean()
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.realize()

  @TinyJit
  def evaluate(self, input: Tensor) -> Tensor:
    self.model(input)
    return logger.realize()

  def single_batch(self, batch_size: int, i: int):
    raise NotImplementedError

  def next_batch(self, batch_size: int = 16):
    Tensor.manual_seed(69)
    i = 0
    while True:
      yield self.single_batch(batch_size, i)
      i += 1

  def __call__(self):

    start = time()
    with Tensor.train():
      print(f'training with {self.config.batch_size} batches')
      i = 0
      for batch in self.next_batch():
        if i % 100 == 0:  # try test first to make sure model is loaded
          end = time()
          print(f'TRAIN: {(end - start)/100.0:.4f} per batch')
          start = end
          with Tensor.test(), Tensor.train(False):
            logger.log_state_dict(self.state_dict)
            stats = self.evaluate(input=batch)
            logger.print_stats(stats.numpy(), i)
            end = time()
            print(f'EVAL: time:{end - start:.1f}')
            start = end
          
        loss: float = self.train(input=batch).item()
        end = time()
        print(f'loss:{loss:.4f}', end=' | ', flush=True)
        if i > 10 and loss < self.loss * 0.8 and self.args.save:
          self.loss = loss
          l = int(self.loss * 100)
          print('saving model', l)
          save_start = time()
          safe_save(get_state_dict(self.model), f'model_{self.args.version}_{l}.pt')
          save_end = time()
          print(f'SAVE: time:{save_end - save_start:.1f}')
          start += save_end - save_start
        i += 1