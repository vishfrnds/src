from typing import Optional, cast

from src.logger import logger
from tinygrad.tensor import Tensor


def log_loss(logits: Tensor, classes: Tensor, mask: Optional[Tensor] = None, print_str: str = '') -> Tensor:
  assert logits.shape[:-1] == classes.shape, f"logits shape {logits.shape} does not match classes shape {classes.shape}"
  if mask is not None:
    assert len(mask.shape) == len(classes.shape), f"mask shape {mask.shape} does not match classes shape {classes.shape}"
    for i, j in zip(mask.shape, classes.shape):
      assert i == j or i == 1, f"mask shape {mask.shape} does not match classes shape {classes.shape}"
  num_classes = cast(int, logits.shape[-1])


  # incorrect = (logits.detach().argmax(axis=-1) - classes).reshape(-1, 1).abs() + 1.1
  # incorrect = incorrect.log()
  classes = classes.reshape(-1).one_hot(num_classes=num_classes)
  logits = logits.reshape(-1, num_classes)
  loss = -logits.log_softmax(axis=-1)
  # .mul(classes).sum(axis=-1)
  if mask is not None:
    classes = classes.mul(mask.reshape(-1, 1))
  loss = loss.mul(classes)
  loss = loss.sum(axis=0) / (classes.sum(axis=0) + 1e-5)
  num_non_zero = (classes.sum(axis=0) > 0).sum()
  loss = loss.sum() / (num_non_zero + 1e-5)


  if not Tensor.training:
    correct = classes.argmax(axis=-1) == logits.argmax(axis=-1)
    if mask is not None:
      correct = correct.mul(mask.reshape(*correct.shape)).int().sum().div(mask.sum())
    else:
      correct = correct.int().mean()
    logger.add_loss(print_str, loss, accuracy=correct)
  return loss
