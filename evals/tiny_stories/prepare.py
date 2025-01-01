"""
Downloads and tokenizes the TinyStories dataset.
- The download is from HuggingFace datasets.
- The tokenization is Llama 3.1 Tokenizer (with tiktoken).

"""
import os
import glob
import json
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from safetensors.torch import save_file, load_file
import numpy as np
from tqdm import tqdm
import os
import glob
import json
import requests
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from src.models.tokenizers_impl import LlamaTokenizer3_1


def get_story_iterator(safetensor_path, batch_size=2):
  tensors = load_file(safetensor_path)
  values = tensors["values"]
  offsets = tensors["offsets"]
  print(offsets)
  num_stories = len(offsets) - 1
  
  for start_idx in range(0, num_stories, batch_size):
    end_idx = min(start_idx + batch_size, num_stories)
    batch_offsets = offsets[start_idx:end_idx + 1]
    batch_lengths = batch_offsets[1:] - batch_offsets[:-1]
    max_len = batch_lengths.max()
    
    # Create padded batch
    batch = torch.zeros((end_idx - start_idx, max_len), dtype=torch.int32)
    
    # Fill each story into the batch
    for i, (start, length) in enumerate(zip(batch_offsets[:-1], batch_lengths)):
      batch[i, :length] = values[start:start + length]
    print('yield')
    
    yield batch, batch_lengths

def process_shard_chunk(shard_filename, chunk_start, chunk_size, tokenizer_path):
  """Process a specific chunk of a shard"""
  tokenizer = LlamaTokenizer3_1(tokenizer_path)
  def encode(x):
    return tokenizer.encode(x, bos=True, eos=True)

  story_tokens = []
  with open(shard_filename, "r") as f:
    data = json.load(f)
  
  # Only process the specified chunk
  chunk_end = min(chunk_start + chunk_size, len(data))
  chunk_data = data[chunk_start:chunk_end]
  
  for example in chunk_data:
    text = example["story"].strip()
    tokens = encode(text)
    story_tokens.append(tokens)
  
  return story_tokens

def write_datafile(filename, toks):
  """Saves token data as a .bin file"""
  assert len(toks) < 2**31, "token count too large"
  header = np.zeros(256, dtype=np.int32)
  header[0] = 20240801 # magic
  header[1] = 7 # version
  header[2] = len(toks)
  toks_np = np.array(toks, dtype=np.uint32)
  print(f"writing {len(toks):,} tokens to {filename}")
  with open(filename, "wb") as f:
    f.write(header.tobytes())
    f.write(toks_np.tobytes())
  del toks_np
  gc.collect()

class TokenWriter:
  def __init__(self, output_dir, max_stories_per_file=500000):
    self.output_dir = output_dir
    self.max_stories_per_file = max_stories_per_file
    self.stories = []
    self.file_counter = 1
  
  def add_tokens(self, story_tokens):
    self.stories.append(torch.tensor(story_tokens, dtype=torch.int32))
    if len(self.stories) >= self.max_stories_per_file:
      self._write_file()
  
  def _write_file(self):
    if not self.stories:
      return
      
    filename = os.path.join(
      self.output_dir,
      f"TinyStories_{self.file_counter}.safetensor"
    )

    values = torch.cat([torch.tensor(s, dtype=torch.int32) for s in self.stories])
    offsets = torch.zeros(len(self.stories) + 1, dtype=torch.int64)
    lengths = torch.tensor([len(s) for s in self.stories], dtype=torch.int32)
    torch.cumsum(lengths, 0, out=offsets[1:])

    tensors = {
        "values": values,  # All tokens concatenated
        "offsets": offsets  # Start position of each story
    }
    save_file(tensors, filename)
    print(f"Saved {len(self.stories)} stories to {filename}")
    self.file_counter += 1
    self.stories = []
    gc.collect()
  
  def finish(self):
    if self.stories:
      self._write_file()

def tokenize(tokenizer_path, data_cache_dir, stories_per_chunk=1000):
  """Memory-efficient tokenization that processes data in small chunks"""
  data_dir = os.path.join(data_cache_dir, "TinyStories_all_data")
  shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
  
  token_writer = TokenWriter(data_cache_dir)
  for shard_filename in shard_filenames:
    
    # Get the number of stories in this shard
    print(shard_filename)
    with open(shard_filename, "r") as f:
      num_stories = len(json.load(f))
    
    # Process the shard in chunks
    with ProcessPoolExecutor(max_workers=3) as executor:
      futures = []
      for chunk_start in range(0, num_stories, stories_per_chunk):
        future = executor.submit(
          process_shard_chunk,
          shard_filename,
          chunk_start,
          stories_per_chunk,
          tokenizer_path
        )
        futures.append(future)
      
      # Process chunks as they complete
      for future in as_completed(futures):
        try:
          chunk_story_tokens = future.result()
          for story_tokens in chunk_story_tokens:
            token_writer.add_tokens(story_tokens)
          del chunk_story_tokens
          gc.collect()
        except Exception as e:
          print(f"Error processing chunk: {e}")
    
    # Force garbage collection after each shard
    gc.collect()
  
    # Write any remaining tokens and report total
  token_writer.finish()
  gc.collect()


def download_file(url: str, fname: str, chunk_size=1024):
  """Helper function to download a file from a given url"""
  resp = requests.get(url, stream=True)
  total = int(resp.headers.get("content-length", 0))
  with open(fname, "wb") as file, tqdm(
    desc=fname,
    total=total,
    unit="iB",
    unit_scale=True,
    unit_divisor=1024,
  ) as bar:
    for data in resp.iter_content(chunk_size=chunk_size):
      size = file.write(data)
      bar.update(size)

def download():
  """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
  os.makedirs(DATA_CACHE_DIR, exist_ok=True)

  # download the TinyStories dataset, unless it's already downloaded
  data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
  data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
  if not os.path.exists(data_filename):
    print(f"Downloading {data_url} to {data_filename}...")
    download_file(data_url, data_filename)
  else:
    print(f"{data_filename} already exists, skipping download...")

  # unpack the tar.gz file into all the data shards (json files)
  data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
  if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    print(f"Unpacking {data_filename}...")
    os.system(f"tar -xzf {data_filename} -C {data_dir}")
  else:
    print(f"{data_dir} already exists, skipping unpacking...")

  # print a single example just for debugging and such
  shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
  print("Download done.")
  print(f"Number of shards: {len(shard_filenames)}")
  # with open(shard_filenames[0], "r") as f:
  #   data = json.load(f)
  # print(f"Example story:\n{data[0]}")

if __name__ == "__main__":
  DATA_CACHE_DIR = "weights/data/tiny_stories"
  TOKENIZER_PATH = "weights/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model"
  
  # Process 1000 stories at a time - adjust this based on your memory constraints
  STORIES_PER_CHUNK = 1000
  
  download() # Assuming this function exists as before
  tokenize(TOKENIZER_PATH, DATA_CACHE_DIR, STORIES_PER_CHUNK)

