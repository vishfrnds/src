# code to read safe tensors into llama

import argparse
import sys

from tinygrad.nn.state import load_state_dict

from models.hub.model_config import ModelConfig, ModelEnum
from models.hub.tokenizer import LlamaTokenizer3_1
from models.language_models import LanguageModel
from models.layers.transformer import Transformer

sys.settrace(None)


class Llama(LanguageModel):
  def __init__(self, hub_model: ModelConfig):
    model = Transformer(hub_model.config)
    load_state_dict(model, hub_model.load_model(), strict=False, consume=True)
    # need to create tokenizer once model is downloaded as it downloads the tokenizer too
    tokenizer = LlamaTokenizer3_1(hub_model.get_tokenizer_path())
    super().__init__(model, tokenizer)

  def process(self):
    input = 'Hello, how are you?'
    output_stream = super().process(input)
    print(''.join(output_stream))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--size', type=ModelEnum, default=ModelEnum.LLAMA_1B, choices=list(ModelEnum))
  args = parser.parse_args()
  Llama(args.size.value).process()
