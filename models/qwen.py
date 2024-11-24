import argparse
import traceback

from transformers import AutoTokenizer

from src.models.hub.model_config import ModelConfig, ModelEnum
from src.models.language_models import LanguageModel, Tokenizer


class QwenTokenizer(Tokenizer):

  def __init__(self, hub_config: ModelConfig):
    self.special_tokens = {
      "<|fim_prefix|>": 151659,
      "<|fim_middle|>": 151660,
      "<|fim_suffix|>": 151661,
      "<|fim_pad|>": 151662,
      "<|repo_name|>": 151663,
      "<|file_sep|>": 151664,
      "<|im_start|>": 151644,
      "<|im_end|>": 151645
    }
    self.tokenizer = AutoTokenizer.from_pretrained(hub_config.hub_name)
    # assert self.tokenizer.vocab_size == hub_config.config.vocab_size, f"Tokenizer vocab size {self.tokenizer.vocab_size} does not match model vocab size {hub_config.config.vocab_size}"

  def encode(self, inp: str):
    return self.tokenizer.encode(inp)

  def decode(self, token: int):
    return self.tokenizer.decode(token)


class Qwen(LanguageModel):
  def __init__(self, hub_model: ModelConfig):
    # need to create tokenizer once model is downloaded as it downloads the tokenizer too
    tokenizer = QwenTokenizer(hub_model)
    super().__init__(hub_model.load_model(), tokenizer)

  def process(self):
    input_text = "#write a quick sort algorithm"
    output_stream = super().process(input_text)
    for op in output_stream:
      print(op, end='', flush=True)


if __name__ == '__main__':
  try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=ModelEnum, default=ModelEnum.QWEN_0_5B, choices=list(ModelEnum))
    args = parser.parse_args()
    Qwen(args.model.value).process()
  except Exception as e:
    print(e)
    traceback.print_exc()