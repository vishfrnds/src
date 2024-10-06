import os
from dataclasses import dataclass

from huggingface_hub import snapshot_download
from models.layers.transformer import Transformer


@dataclass
class ModelConfig:
  hub_name: str
  config: Transformer.Config

  def get_auth_token(self):
    auth_token = os.getenv("HF_API_KEY")
    assert auth_token is not None, "Please set HF_API_KEY environment variable"
    return auth_token

  def get_dir_path(self) -> str:
    return './weights/' + self.hub_name

  def get_tokenizer_path(self):
    return self.get_dir_path() + '/original/tokenizer.model'

  def download_model(self):
    snapshot_download(repo_id=self.hub_name, token=self.get_auth_token(), local_dir=self.get_dir_path(), ignore_patterns=['*.pth'])
    print("Model downloaded successfully.")


config_llama3_chat_7b: ModelConfig = ModelConfig(hub_name="meta-llama/Llama-3.1-8B-Instruct",
                                                 config=Transformer.Config(dim=4096, hidden_dim=1436,
                                                                           n_heads=32, n_layers=32, norm_eps=1e-5, vocab_size=128256, n_kv_heads=8, rope_theta=0.01, max_seq_len=4096))
