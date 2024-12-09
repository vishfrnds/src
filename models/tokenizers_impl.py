# Naming this file tokenizers.py clashes with transformer library.
import os
from tiktoken.load import load_tiktoken_bpe
import tiktoken
from typing import (
    AbstractSet,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
)
from pathlib import Path
from transformers import AutoTokenizer

from src.models.language_models import BaseTokenizer


class QwenTokenizer(BaseTokenizer):

  def __init__(self, dir_path: str):
    super().__init__(dir_path)
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
    self.tokenizer = AutoTokenizer.from_pretrained(self.dir_path)
  
  def test_run(self):
    input_text = "#write a quick sort algorithm"
    print(self.tokenizer.encode(input_text))

  def input_to_tokens(self, inp: str) -> List[int]:
    print('encoding inp', inp)
    return self.tokenizer.encode(inp)


  def token_to_string(self, token: int) -> Union[str, None]:
    eos_token_ids = [151664, 151662, 151659, 151660, 151661, 151662, 151663, 151664, 151645, 151643]
    return self.tokenizer.decode(token)


# Can be improved a lot by reading those tokenizer files instead of hardcoding values here.


class LlamaTokenizer3:
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

  def __init__(self, model_path: str):
    mergeable_ranks = load_tiktoken_bpe(model_path)
    self.num_base_tokens = len(mergeable_ranks)
    special_tokens = [
      "<|begin_of_text|>",
      "<|end_of_text|>",
      "<|reserved_special_token_0|>",
      "<|reserved_special_token_1|>",
      "<|reserved_special_token_2|>",
      "<|reserved_special_token_3|>",
      "<|start_header_id|>",
      "<|end_header_id|>",
      "<|reserved_special_token_4|>",
      "<|eot_id|>",
    ] + [
      f"<|reserved_special_token_{i}|>"
      for i in range(5, 256 - 5)
    ]
    self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}

    self.model = tiktoken.Encoding(name=model_path, pat_str=self.pat_str, mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens)

  @property
  def bos_id(self): return self.special_tokens["<|begin_of_text|>"]
  @property
  def stop_tokens(self): return {self.special_tokens["<|end_of_text|>"], self.special_tokens["<|eot_id|>"]}

  def decode(self, toks): return self.model.decode([t for t in toks if t < self.num_base_tokens])

  def encode(self, text, allow_special=False):
    return self.model.encode(text, allowed_special="all" if allow_special else set(), disallowed_special=set())


# Llama 3.1 Tokenizer (with tiktoken)


# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


# https://github.com/karpathy/nano-llama31/blob/master/tokenizer.py
class LlamaTokenizer3_1(BaseTokenizer):
  """ Converts List[int] <-> str """

  special_tokens: Dict[str, int]
  num_reserved_special_tokens = 256
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

  def __init__(self, model_path):
    assert os.path.isfile(model_path), model_path
    mergeable_ranks = load_tiktoken_bpe(model_path)
    num_base_tokens = len(mergeable_ranks)
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|finetune_right_pad_id|>",
        "<|step_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",  # end of message
        "<|eot_id|>",  # end of turn
        "<|python_tag|>",
    ]
    reserved_tokens = [
        f"<|reserved_special_token_{2 + i}|>"
        for i in range(self.num_reserved_special_tokens - len(special_tokens))
    ]
    special_tokens = special_tokens + reserved_tokens

    self.special_tokens = {
        token: num_base_tokens + i for i, token in enumerate(special_tokens)
    }
    self.model = tiktoken.Encoding(
        name=Path(model_path).name,
        pat_str=self.pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=self.special_tokens,
    )

    self.n_words: int = num_base_tokens + len(special_tokens)
    self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
    self.eos_id: int = self.special_tokens["<|end_of_text|>"]
    self.eot_id: int = self.special_tokens["<|eot_id|>"]
    self.eom_id: int = self.special_tokens["<|eom_id|>"]
    self.python_tag_id = self.special_tokens["<|python_tag|>"]
    self.pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]
    self.stop_tokens = [
        # AK: I changed the tokens around here to be for the base model
        # as I understand the sequence is <BOS>content<EOS><BOS>content<EOS>...
        self.special_tokens["<|begin_of_text|>"],
        self.special_tokens["<|end_of_text|>"],
    ]

  def encode(
      self,
      s: str,
      *,
      bos: bool = True,
      eos: bool = False,
      allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
      disallowed_special: Union[Literal["all"], Collection[str]] = (),
  ) -> List[int]:
    """
    Encodes a string into a list of token IDs.

    Args:
        s (str): The input string to be encoded.
        bos (bool): Whether to prepend the beginning-of-sequence token.
        eos (bool): Whether to append the end-of-sequence token.
        allowed_tokens ("all"|set[str]): allowed special tokens in string
        disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

    Returns:
        list[int]: A list of token IDs.

    By default, setting disallowed_special=() encodes a string by ignoring
    special tokens. Specifically:
    - Setting `disallowed_special` to () will cause all text corresponding
      to special tokens to be encoded as natural text (insteading of raising
      an error).
    - Setting `allowed_special` to "all" will treat all text corresponding
      to special tokens to be encoded as special tokens.
    """
    if allowed_special is None:
      allowed_special = set()
    assert type(s) is str

    substrs = (
        substr
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
        for substr in self._split_whitespaces_or_nonwhitespaces(
            s[i: i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
        )
    )
    t: List[int] = []
    for substr in substrs:
      t.extend(
          self.model.encode(
              substr,
              allowed_special=allowed_special,
              disallowed_special=disallowed_special,
          )
      )
    if bos:
      t.insert(0, self.bos_id)
    if eos:
      t.append(self.eos_id)
    return t

  def decode(self, t: int) -> str:
    # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
    return self.model.decode([t])

  @staticmethod
  def _split_whitespaces_or_nonwhitespaces(
      s: str, max_consecutive_slice_len: int
  ) -> Iterator[str]:
    """
    Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
    consecutive whitespaces or consecutive non-whitespaces.
    """
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0

    for i in range(len(s)):
      is_now_space = s[i].isspace()

      if current_slice_is_space ^ is_now_space:
        current_slice_len = 1
        current_slice_is_space = is_now_space
      else:
        current_slice_len += 1
        if current_slice_len > max_consecutive_slice_len:
          yield s[slice_start:i]
          slice_start = i
          current_slice_len = 1
    yield s[slice_start:]
