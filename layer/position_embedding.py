import math
from typing import Optional, Tuple

from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


class PositionEmbedding:
  @staticmethod
  def apply_scaling(freqs: Tensor) -> Tensor:
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    print('combined', freqs.shape)
    for freq in freqs:
      wavelen = 2 * math.pi / freq.item()
      if wavelen < high_freq_wavelen:
        print('low', freq.item())
        new_freqs.append(freq.item())
      elif wavelen > low_freq_wavelen:
        print('high', freq.item() / scale_factor)
        new_freqs.append(freq.item() / scale_factor)
      else:
        assert low_freq_wavelen != high_freq_wavelen
        smooth = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        new_freqs.append((1 - smooth) * freq.item() / scale_factor + smooth * freq.item())
        print('smooth', new_freqs[-1])
    print('new_freqs', len(new_freqs), new_freqs)
    return Tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

  @staticmethod
  def precompute_freqs_cis(head_dim: int, end: int, theta: float, use_scaled: bool, dtype=dtypes.float32) -> Tensor:
    freqs = 1.0 / (theta ** (Tensor.arange(0, head_dim, 2, dtype=dtype)[:(head_dim // 2)] / head_dim))
    if use_scaled:
      freqs = PositionEmbedding.apply_scaling(freqs)
    freqs = Tensor.arange(end, dtype=dtype).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
    return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, head_dim // 2, 2)

  def __init__(self, head_dim: int, max_seq_len: int, rope_theta: float, use_scaled_rope: bool):
    self.freqs_cis = PositionEmbedding.precompute_freqs_cis(head_dim, max_seq_len * 2, rope_theta, use_scaled_rope).contiguous()
    self.freqs_cis.requires_grad = False
    # self.cache: Optional[Tuple[int, int, Tensor]] = None

  # def __call__(self, start_pos: int, seq_len: int) -> Tensor:
  #   return Tensor.arange(start_pos, start_pos + seq_len)

  @staticmethod
  def complex_mul(A: Tensor, c: Tensor, d: Tensor) -> Tensor:
    # (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
    a, b = A[:, :, :, 0:1, :], A[:, :, :, 1:2, :]
    ro = a * c - b * d
    co = a * d + b * c
    return ro.cat(co, dim=-1)

  def get_freqs_cis(self, start_pos: int, seq_len: int) -> Tensor:
    return self.freqs_cis.shrink((None, (start_pos, start_pos + seq_len), None, None, None))

  @staticmethod
  # There two options to rotate indexes we can rotate:
  # [0, 1, 2, ..., d/2] [d/2+1, d/2+2, ..., d]
  # or [0, 2, 4, ..., d] [1, 3, 5, ..., d-1]
  # qwen, llama is doing first
  # Todo parametrize this for different models
  def apply_rotary_emb(freqs_cis: Tensor, xq: Tensor, xk: Tensor) -> Tuple[Tensor, Tensor]:
    dtype = xq.dtype
    # assert freqs_cis.shape[1] == xq.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
    xq = xq.reshape(*xq.shape[0:-1], 2, -1)
    xk = xk.reshape(*xk.shape[0:-1], 2, -1)
    assert len(xq.shape) == 5 and len(xk.shape) == 5 and len(freqs_cis.shape) == 5
    cos, sin = freqs_cis[:, :xq.shape[1], :, :, 0:1], freqs_cis[:, :xq.shape[1], :, :, 1:2]
    cos, sin = cos.transpose(-1, -2), sin.transpose(-1, -2)
    xq_out = PositionEmbedding.complex_mul(xq, cos, sin)
    xk_out = PositionEmbedding.complex_mul(xk, cos, sin)
    return xq_out.flatten(3).cast(dtype), xk_out.flatten(3).cast(dtype)
