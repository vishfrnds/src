import json
from os import getenv
from pathlib import Path
from typing import Dict

from model_config import ModelConfig
from model_config import config_llama3_chat_7b as model_config
from tinygrad import Device, Tensor, dtypes
from tinygrad.nn.state import safe_load

# **** helper functions ****


def convert_from_huggingface(weights: Dict[str, Tensor], n_heads: int, n_kv_heads: int, n_layers: int):
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
    v = v.to(Device.DEFAULT)
    if "model.layers" in k:
      if "q_proj" in k:
        v = permute(v, n_heads)
      elif "k_proj" in k:
        v = permute(v, n_kv_heads)
    sd[keymap[k]] = v
  return sd


def load_model(model_config: ModelConfig):
  # model = Transformer(**model_config)
  dir = model_config.get_dir_path()
  index = dir + '/model.safetensors.index.json'
  # check if file exists
  assert Path(index).exists(), f"{index} does not exist"
  with open(index) as fp:
    weight_map = json.load(fp)['weight_map']
    parts: dict[str, Tensor] = {n: safe_load(dir + '/' + n) for n in set(weight_map.values())}
    print(parts.values())
    weights = {k: parts[n][k] for k, n in weight_map.items()}
    weights = convert_from_huggingface(weights, model_config.n_heads, model_config.n_kv_heads, model_config.n_layers)


load_model(model_config)
