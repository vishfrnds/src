import os
from enum import Enum
from pathlib import Path
from typing import Dict

from huggingface_hub import snapshot_download
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_load
from tinygrad.tensor import Tensor

from models.layers.transformer import TransformerConfig


class ModelConfig:
  def __init__(self, hub_name: str, config: TransformerConfig):
    self.hub_name = hub_name
    self.config = config

  def get_auth_token(self):
    auth_token = os.getenv("HF_API_KEY")
    assert auth_token is not None, "Please set HF_API_KEY environment variable"
    return auth_token

  def get_dir_path(self) -> str:
    return './weights/' + self.hub_name

  def get_tokenizer_path(self):
    return self.get_dir_path() + '/original/tokenizer.model'

  def download_model(self) -> None:
    snapshot_download(repo_id=self.hub_name, token=self.get_auth_token(), local_dir=self.get_dir_path(), ignore_patterns=['*.pth'])
    print("Model downloaded successfully.")

  def convert_from_huggingface(self, weights: Dict[str, Tensor], n_heads: int, n_kv_heads: int, n_layers: int) -> Dict[str, Tensor]:
    print("moving tensors to device: ", Device.DEFAULT)
    print(list(Device.get_available_devices()))

    def permute(v: Tensor, n_heads: int):
      return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1]).transpose(1, 2).reshape(*v.shape[:2])

    keymap = {
      "model.embed_tokens.weight": "tok_embeddings.weight",
      **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight" for l in range(n_layers)},
      **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attention.w{x}.weight" for x in ["q", "k", "v", "o"] for l in range(n_layers)},
      **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in range(n_layers)},
      **{f"model.layers.{l}.mlp.{x}_proj.weight": f"layers.{l}.feed_forward.w{y}.weight" for x, y in {"gate": "1", "down": "2", "up": "3"}.items() for l in range(n_layers)},
      "model.norm.weight": "norm.weight",
      "lm_head.weight": "output.weight",
    }
    sd = {}
    for k, v in weights.items():
      if ".rotary_emb." in k:
        continue
      # TODO: This should work with LLVM, but it doesn't, see test in cpu_half_test.py.
      # check if gpu is available
      if Device.DEFAULT not in ["CUDA", "NV"]:
        v = v.to("CLANG").realize()
        if v.dtype == dtypes.bfloat16:
          # v = v.llvm_bf16_cast(dtypes.float32).realize()
          v = v.bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1 << 16).bitcast(dtypes.float32).cast(dtypes.float16).realize()

      if "model.layers" in k:
        if "q_proj" in k:
          v = permute(v, n_heads)
        elif "k_proj" in k:
          v = permute(v, n_kv_heads)
      sd[keymap[k]] = v
    return sd

  def fix_data_type(self, weights: Dict[str, Tensor]) -> Dict[str, Tensor]:
      # TODO: This should work with LLVM, but it doesn't, see test in cpu_half_test.py.
      # check if gpu is available
    for k, v in weights.items():
      if Device.DEFAULT not in ["CUDA", "NV"]:
        v = v.to("CLANG").realize()
        if v.dtype == dtypes.bfloat16:
          # v = v.llvm_bf16_cast(dtypes.float32).realize()
          v = v.bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1 << 16).bitcast(dtypes.float32).cast(dtypes.float16).realize()

    return weights


  def load_model(self) -> Dict[str, Tensor]:
    dir = Path(self.get_dir_path())
    if not dir.exists():
      self.download_model()
    files = list(dir.glob('*.safetensors'))
    assert len(files) > 0, f"No .safetensors files found in {dir}"
    weights = {k: v for file in files for k, v in safe_load(str(file)).items()}
    weights = self.fix_data_type(weights)
    weights = self.convert_from_huggingface(
        weights, self.config.n_heads, self.config.n_kv_heads, self.config.n_layers)
    return weights


class ModelEnum(Enum):
  LLAMA_8B = ModelConfig(hub_name="meta-llama/Llama-3.1-8B-Instruct",
                         config=TransformerConfig(dim=4096, hidden_dim=14336, n_heads=32, n_layers=32, norm_eps=1e-5, vocab_size=128256, n_kv_heads=8, rope_theta=500000.0, max_seq_len=4096, use_scaled_rope=False))
  LLAMA_1B = ModelConfig(hub_name="meta-llama/Llama-3.2-1B-Instruct",
                         config=TransformerConfig(dim=2048, hidden_dim=8192, n_heads=32, n_layers=16, norm_eps=1e-5, vocab_size=128256, n_kv_heads=8, rope_theta=500000.0, max_seq_len=4096, use_scaled_rope=True))
