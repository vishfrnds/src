import os
from dataclasses import dataclass

from huggingface_hub import snapshot_download


@dataclass
class ModelConfig:
  hub_name: str
  dim: int
  hidden_dim: int
  n_heads: int
  n_layers: int
  norm_eps: float
  vocab_size: int
  n_kv_heads: int
  rope_theta: float
  max_seq_len: int

  def get_auth_token(self):
    auth_token = os.getenv("HF_API_KEY")
    assert auth_token is not None, "Please set HF_API_KEY environment variable"
    return auth_token

  def get_dir_path(self):
    return './weights/' + 'a' + self.hub_name

  def get_tokenizer_path(self):
    return self.get_dir_path() + '/original/tokenizer.model'

  def download_model(self):
    snapshot_download(repo_id=self.hub_name, token=self.get_auth_token(), local_dir=self.get_dir_path(), ignore_patterns=['*.pth'])
    print("Model downloaded successfully.")


config_llama3_chat_7b: ModelConfig = ModelConfig(hub_name="meta-llama/Llama-3.1-8B-Instruct", dim=4096, hidden_dim=1436,
                                                 n_heads=32, n_layers=32, norm_eps=1e-5, vocab_size=128256, n_kv_heads=32, rope_theta=0.01, max_seq_len=4096)


if __name__ == "__main__":
  print("Downloading model...")
  download_model(config_llama3_chat_7b)
