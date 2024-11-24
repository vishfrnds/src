import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

import tinygrad
from src.models.hub.model_config import ModelEnum
from src.models.qwen import Qwen
from tinygrad.dtype import dtypes


class TestQwenModel(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    print('Device.DEFAULT', tinygrad.Device.DEFAULT)
    tinygrad.dtypes.default_float = dtypes.float32
    # tinygrad.Device.DEFAULT = "CLANG"
    input_text = "#write a quick sort algorithm"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
    cls.hub_model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", device_map="cpu", torch_dtype=torch.float32).eval()
    cls.hub_model_inputs = tokenizer([input_text], return_tensors="pt").to('cpu').input_ids

    # Load tinygrad implmentation
    qwen = Qwen(ModelEnum.QWEN_0_5B.value)
    cls.tiny_model = qwen.model
    cls.tiny_input = tinygrad.Tensor([qwen.tokenizer.encode(input_text)])
    cls.tiny_model.initialize_cache_if_needed(bsz=1)

  @staticmethod
  def unnest(nested_list):
    """Recursively flattens a nested list of lists into a single flat list"""
    flattened = []
    if isinstance(nested_list, list):
      for item in nested_list:
        flattened.extend(TestQwenModel.unnest(item))
    else:
      flattened.append(nested_list)
    return flattened

  def assert_close(self, tiny, hub: torch.Tensor, name: str):
    # tiny = tiny.cast(dtypes.float32)
    # assert same shape
    self.assertEqual(tiny.shape, hub.shape, f"Shapes don't match {tiny.shape} {hub.shape}")
    tiny, hub = self.unnest(tiny.tolist()), self.unnest(hub.tolist())
    mismatch = 0
    # assert avg_diff < 1e-4, f'avg diff {avg_diff}'
    for a, b in zip(tiny, hub):
      if abs(a - b) / (abs(b) + 1e-4) > 0.2 and abs(a - b) > 1e-2:
        mismatch += 1
        if mismatch < 10:
          print(f'{name} mismatch {a} {b}, diff {abs(a - b)}, ratio {abs(a - b) / (2 * (abs(a) + abs(b) + 1e-4))}')
    avg_diff = sum(abs(tiny - hub) for tiny, hub in zip(tiny, hub)) / len(tiny)
    if mismatch > 0:
      print(f'{name}: avg diff {avg_diff}, mismatch {mismatch}')
    assert mismatch < 0.01 * len(tiny), f'{mismatch} mismatches'

  def test_model_outputs(self):
    hub_hidden_states = self.hub_model(self.hub_model_inputs, output_hidden_states=True).hidden_states
    hub = self.hub_model.model.embed_tokens(self.hub_model_inputs)
    # self.assert_close(hub, hub_hidden_states[0], 'embed_tokens')
    tiny = self.tiny_model.tok_embeddings(self.tiny_input)
    self.assert_close(tiny, hub_hidden_states[0], 'tok_embeddings')

    self.assert_close(self.tiny_model.layers[0].attention.wq.weight, self.hub_model.model.layers[0].self_attn.q_proj.weight, 'q_proj')
    self.assert_close(self.tiny_model.layers[0].attention.wq.bias, self.hub_model.model.layers[0].self_attn.q_proj.bias, 'q_proj_bias')
    self.assert_close(self.tiny_model.layers[0].attention.wk.weight, self.hub_model.model.layers[0].self_attn.k_proj.weight, 'k_proj')
    self.assert_close(self.tiny_model.layers[0].attention.wk.bias, self.hub_model.model.layers[0].self_attn.k_proj.bias, 'k_proj_bias')
    self.assert_close(self.tiny_model.layers[0].attention.wv.weight, self.hub_model.model.layers[0].self_attn.v_proj.weight, 'v_proj')

    tiny_query = self.tiny_model.layers[0].attention.wq(tiny).reshape(tiny.shape[0], tiny.shape[1], self.tiny_model.config.n_heads, self.tiny_model.layers[0].attention.head_dim)
    hub_query = self.hub_model.model.layers[0].self_attn.q_proj(hub).view(hub.shape[0], hub.shape[1], self.hub_model.config.num_attention_heads, self.hub_model.model.layers[0].self_attn.head_dim)
    self.assert_close(tiny_query, hub_query, 'q_proj')
    tiny_key = self.tiny_model.layers[0].attention.wk(tiny).reshape(tiny.shape[0], tiny.shape[1], self.tiny_model.config.n_kv_heads, self.tiny_model.layers[0].attention.head_dim)
    hub_key = self.hub_model.model.layers[0].self_attn.k_proj(hub).view(hub.shape[0], hub.shape[1], self.hub_model.config.num_key_value_heads, self.hub_model.model.layers[0].self_attn.head_dim)
    self.assert_close(tiny_key, hub_key, 'k_proj')

    tiny_freqs_cis = self.tiny_model.positional_embedding.get_freqs_cis(0, hub.shape[1])
    tiny_query, tiny_key = self.tiny_model.positional_embedding.apply_rotary_emb(tiny_freqs_cis, tiny_query, tiny_key)
    tiny_query, tiny_key = tiny_query.transpose(1, 2), tiny_key.transpose(1, 2)

    cos, sin = self.hub_model.model.rotary_emb(hub, torch.arange(hub.shape[1]).unsqueeze(0))
    hub_query, hub_key = apply_rotary_pos_emb(hub_query.transpose(1, 2), hub_key.transpose(1, 2), cos, sin)
    self.assert_close(tiny_query, hub_query, 'rotary_emb_query')
    self.assert_close(tiny_key, hub_key, 'rotary_emb_key')

    for layer_idx, layer in enumerate(self.tiny_model.layers):
      tiny = layer(tiny, 0, tiny_freqs_cis)
      if layer_idx == len(self.tiny_model.layers) - 1:
        tiny = self.tiny_model.norm(tiny)
      self.assert_close(tiny, hub_hidden_states[layer_idx + 1], f'layer_{layer_idx}')
    print(f'output_len {len(hub_hidden_states)}')

  def test_second_call(self):
    eos_token_ids = [151664, 151662, 151659, 151660, 151661, 151662, 151663, 151664, 151645, 151643]
    hub_generate_ids = self.hub_model.generate(self.hub_model_inputs, max_length=11, eos_token_id=eos_token_ids, do_sample=False)
    tiny_output = []
    tiny_output.append(self.tiny_model(self.tiny_input, 0))
    start_pos = self.tiny_input.shape[1]
    assert tiny_output[-1].item() == hub_generate_ids[0][start_pos].item(), f'first_output {tiny_output[0].item()} {hub_generate_ids[0][start_pos].item()}'
    for _ in range(10):
      start_pos += 1
      tiny_output.append(self.tiny_model(tiny_output[-1], start_pos))
      assert tiny_output[-1].item() == hub_generate_ids[0][start_pos].item(), f'first_output {tiny_output[0].item()} {hub_generate_ids[0][start_pos].item()}'



if __name__ == '__main__':
  unittest.main()