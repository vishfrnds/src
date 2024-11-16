import subprocess
import unittest

from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


class cpu_float16_test(unittest.TestCase):
  def test_half_clang(self):
    Tensor.manual_seed(42)
    t = Tensor.randn(1, dtype=dtypes.float32, device="CLANG")
    t = t.realize()
    print(t)
    t = t.cast(dtypes.float16).realize()
    print(t, t.item())

  # Maybe my installation is problem, llvm is not supporting half precision, but clang is.
  def test_half_llvm(self):
    dangerous_code = "import tinygrad; tinygrad.Tensor.manual_seed(42); t = tinygrad.Tensor.randn(1, dtype=tinygrad.dtype.dtypes.float32, device='LLVM'); t = t.realize(); t = t.cast(tinygrad.dtype.dtypes.float16).realize()"
    try:
      subprocess.run(["python3", "-c", dangerous_code], check=True)
    except subprocess.CalledProcessError as e:
      assert e.returncode == -11, "Expected segmentation fault"
    else:
      assert False, "Expected segmentation fault"


if __name__ == "__main__":
  unittest.main()
