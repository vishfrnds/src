import argparse
import os
import traceback
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

from huggingface_hub import snapshot_download
from tinygrad.runtime.ops_mcloud import MCloudDevice

from src.layer.transformer import ActivationEnum, Transformer, TransformerConfig
from src.models.language_models import BaseTokenizer, LanguageModel
from src.models.tokenizers_impl import LlamaTokenizer3_1, QwenTokenizer
from tinygrad.device import Buffer, Device
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import prod
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load
from tinygrad.tensor import Tensor


class ModelConfig:
  def __init__(self, hub_name: str, config: TransformerConfig, tokenizer: type[BaseTokenizer]):
    self.hub_name = hub_name
    self.config = config
    self.tokenizer: type[BaseTokenizer] = tokenizer

  def create_language_model(self) -> LanguageModel:
    model = self.load_model()
    # need to create tokenizer once model is downloaded as it downloads the tokenizer too
    print('tokenizer', self.tokenizer)
    tokenizer = self.tokenizer(self.get_dir_path())
    return LanguageModel(model, tokenizer)

  def get_auth_token(self):
    auth_token = os.getenv("HF_API_KEY")
    assert auth_token is not None, "Please set HF_API_KEY environment variable"
    return auth_token

  def get_dir_path(self) -> str:
    return './weights/' + self.hub_name

  def get_tokenizer_path(self):
    return self.get_dir_path() + '/original/tokenizer.model'

  def get_model_weights_on_disk(self) -> Dict[str, Tensor]:
    dir = Path(self.get_dir_path())
    if not dir.exists():
      snapshot_download(repo_id=self.hub_name, token=self.get_auth_token(), local_dir=self.get_dir_path(), ignore_patterns=['*.pth'])
    print("Model downloaded successfully.")
    # This creates a tensor clang for metadata of file
    weights = {k: v for file in Path(self.get_dir_path()).glob('*.safetensors') for k, v in safe_load(str(file)).items()}
    assert weights, f"No .safetensors files found in {self.get_dir_path()}"
    weights = self.convert_from_huggingface(self.fix_data_type(weights),
                                            self.config.n_heads, self.config.n_kv_heads, self.config.n_layers)

    # for v in weights.values(): assert v.device.startswith("DISK"), f"Model weights are not on disk, found {v.device}"
    return weights
  

  def convert_from_huggingface(self, weights: Dict[str, Tensor], n_heads: int, n_kv_heads: int, n_layers: int) -> Dict[str, Tensor]:

    keymap = {
      "model.embed_tokens.weight": "tok_embeddings.weight",
      **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight" for l in range(n_layers)},
      **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attention.w{x}.weight" for x in ["q", "k", "v", "o"] for l in range(n_layers)},
      **{f"model.layers.{l}.self_attn.{x}_proj.bias": f"layers.{l}.attention.w{x}.bias" for x in ["q", "k", "v", "o"] for l in range(n_layers)},
      **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in range(n_layers)},
      **{f"model.layers.{l}.mlp.{x}_proj.weight": f"layers.{l}.feed_forward.w{y}.weight" for x, y in {"gate": "1", "down": "2", "up": "3"}.items() for l in range(n_layers)},
      "model.norm.weight": "norm.weight",
      "lm_head.weight": "output.weight",
    }
    sd = {}
    for k, v in weights.items():
      if ".rotary_emb." in k:
        continue
      assert v.dtype != dtypes.bfloat16 or v.device != 'CLANG', f"bfloat16 found in {k} on device {v.device}"
      sd[keymap[k]] = v
    return sd

  def convert_to_float16(self, v: Tensor) -> Tensor:
    return v.to('CLANG').bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1 << 16).bitcast(dtypes.float32).cast(dtypes.float16).to(Device.DEFAULT)

  def fix_data_type(self, weights: Dict[str, Tensor]) -> Dict[str, Tensor]:
      # TODO: This should work with LLVM, but it doesn't, see test in cpu_half_test.py.
      # check if gpu is available
    # for k, v in weights.items():
      # weights[k] = v.to("CLANG").float().realize()
      # weights[k] = v.float().realize()
      # if Device.DEFAULT == 'CLANG':
      #   print('moving to cpu', k)
      #   v = v.to("CLANG").realize()
      #   if v.dtype == dtypes.bfloat16:
      #     # v = v.llvm_bf16_cast(dtypes.float16).realize()
          # weights[k] = v.to('CLANG').bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1 << 16).bitcast(dtypes.float32).cast(dtypes.float16)

    return weights


  def load_model(self) -> Transformer:
    if Device.DEFAULT == 'MCLOUD':
      from typing import cast
      device = cast(MCloudDevice, Device['MCLOUD'])
      print('VISH1332', device.loaded_models)
      # model_weights = device.loaded_models[self.hub_name]
      # print('VISH1332', model_weights)
      return None
    else:
      weights = self.get_model_weights_on_disk()
    # if Device.DEFAULT == 'MCLOUD':
    #   t = Tensor([1])
    #   print('a', t)
    #   t = t.realize()
    #   print('b', t)
    #   print('t', t.numpy())
    #   print('weights in memory', len(MCloudDevice.server_memory))
    #   for k, v in weights.items():
    #     v = self.convert_to_float16(v)
    #     hash = h(bytes(v.data()))
    #     buffer = Buffer(device='MCLOUD', size=prod(v.shape), dtype=v.dtype)
    #     buffer._buf = MCloudDevice.server_memory[hash]
    #     empty_lazybuffer = LazyBuffer.metaop(Ops.EMPTY, v.shape, v.dtype, "MCLOUD")
    #     empty_lazybuffer.buffer = buffer
    #     del empty_lazybuffer.srcs
    #     weights[k] = Tensor(empty_lazybuffer)
    # else:
      for k, v in weights.items():
        weights[k] = self.convert_to_float16(v)

    transformer = Transformer(self.config)
    model_state_dict = get_state_dict(transformer)
    model_keys = [k for k, v in model_state_dict.items() if v.requires_grad is not False]  # it has 3 states t, f, none
    assert not (extra := [k for k in weights if k not in model_keys]), f"Extra keys: {extra}"
    assert not (missing := [k for k in model_keys if k not in weights]), f"Missing keys: {missing}"
    #load_state_dict(transformer, weights, strict=False, consume=True)
    for k in model_keys:
      if weights[k].shape != model_state_dict[k].shape:
        raise ValueError(f'Shape mismatch in layer `{k}`: Expected shape {weights[k].shape}, but found {model_state_dict[k].shape} in state dict.')
      model_state_dict[k].replace(weights[k].to(model_state_dict[k].device)).realize()
      del weights[k]
    print('model loaded')
    return transformer


class ModelEnum(Enum):
  LLAMA_8B = ModelConfig(
    hub_name="meta-llama/Llama-3.1-8B-Instruct",
    config=TransformerConfig(
      dim=4096,
      hidden_dim=14336,
      n_heads=32,
      n_layers=32,
      norm_eps=1e-5,
      vocab_size=128256,
      n_kv_heads=8,
      rope_theta=500000.0,
      max_seq_len=4096,
      use_scaled_rope=False,
      tie_embeddings=False,
      attention_bias=False,
    ),
    tokenizer=LlamaTokenizer3_1
  )
  LLAMA_1B = ModelConfig(
    hub_name="meta-llama/Llama-3.2-1B-Instruct",
    config=TransformerConfig(
      dim=2048,
      hidden_dim=8192,
      n_heads=32,
      n_layers=16,
      norm_eps=1e-5,
      vocab_size=128256,
      n_kv_heads=8,
      rope_theta=500000.0,
      max_seq_len=8192,
      use_scaled_rope=True,
      tie_embeddings=True,
      attention_bias=False,
      activation=ActivationEnum.SILU
    ),
    tokenizer=LlamaTokenizer3_1
  )
  QWEN_0_5B = ModelConfig(
    hub_name="Qwen/Qwen2.5-Coder-0.5B",
    config=TransformerConfig(
      dim=896,
      hidden_dim=4864,
      n_heads=14,
      n_kv_heads=2,
      n_layers=24,
      norm_eps=1e-6,
      vocab_size=151936,
      rope_theta=1000000.0,
      max_seq_len=32768,
      use_scaled_rope=False,
      tie_embeddings=True,
      attention_bias=True,
      activation=ActivationEnum.SILU
    ),
    tokenizer=QwenTokenizer
  )

  QWEN_7B_INSTRUCT = ModelConfig(
    hub_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    config=TransformerConfig(
      dim=3584,
      hidden_dim=18944,
      n_heads=28,
      n_kv_heads=4,
      n_layers=28,
      norm_eps=1e-6,
      vocab_size=152064,
      rope_theta=1000000.0,
      max_seq_len=32768,
      use_scaled_rope=False,
      tie_embeddings=False,
      attention_bias=True,
      activation=ActivationEnum.SILU
    ),
    tokenizer=QwenTokenizer
  )


if __name__ == '__main__':
  try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=[e.name for e in ModelEnum], default='QWEN_0_5B')
    args = parser.parse_args()
    model_enum = ModelEnum[args.model]
    print(f"Using model: {model_enum}")
    model_config = model_enum.value
    language_model = model_config.create_language_model()
    language_model.run()
  except Exception as e:
    print(e)
    traceback.print_exc()
