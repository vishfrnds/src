import unittest
from transformers import AutoTokenizer
from hub import config_llama3_chat_7b as model
from tokenizer import LlamaTokenizer3_1


class TestTokenizers(unittest.TestCase):
  # TODO: Add tests for
  # Special tokens, and hindi script
  # Add speed tests of hf and tiktoken tokenizer.

  def setUp(self):
    # Initialize your custom tokenizer
    self.custom_tokenizer = LlamaTokenizer3_1(model.get_tokenizer_path())

    # Initialize the Hugging Face tokenizer
    self.hf_tokenizer = AutoTokenizer.from_pretrained(model.get_dir_path(), token=model.get_auth_token()
                                                      )

    # Define a sample text
    self.sample_text = "Hello, how are you?"

  def test_tokenizers_produce_same_results(self):
    # Tokenize using your custom tokenizer
    custom_tokens = self.custom_tokenizer.encode(self.sample_text, bos=True, eos=False)

    # Tokenize using the Hugging Face tokenizer
    hf_tokens = self.hf_tokenizer.encode(self.sample_text)

    # Compare the results
    self.assertEqual(custom_tokens, hf_tokens, "The tokenizers produce different results.")


if __name__ == '__main__':
  unittest.main()
