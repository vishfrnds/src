from src.models.model_config import ModelEnum
from src.models.model_config import ModelEnum
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import os
import torch
from src.evals.tiny_stories.prepare import get_story_iterator

class Llama:
  def __init__(self) -> None:
    self.tokenizer = AutoTokenizer.from_pretrained(ModelEnum.LLAMA_1B.value.get_dir_path())
    self.hub_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(ModelEnum.LLAMA_1B.value.get_dir_path(), device_map="cpu", torch_dtype=torch.bfloat16).eval()
    self.loss_fn = CrossEntropyLoss(reduction='none')


  def run(self):
    print("Welcome to the chat bot! Press Ctrl+Enter to send a message.")
    while True:
      try:
        user_input = ""
        while True:
          line = input()
          if line == "":
            break
          user_input += line + "\n"
        
        print("User: " + user_input)
        hub_model_inputs = self.tokenizer([user_input], return_tensors="pt").to('cpu')
        generated_ids = self.hub_model.generate(**hub_model_inputs)
        bot_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Assistant: ", end="")
        for output in bot_output:
          print(output, end="", flush=True)
        print()
      except KeyboardInterrupt:
        print("\nExiting chat bot. Goodbye!")
        break

  def calculate_loss(self, input_ids: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Calculate cross entropy loss for each position in the sequence."""
    with torch.no_grad():
      # Forward pass through model
      print(f'Input IDs: {input_ids}')
      outputs = self.hub_model(input_ids)
      print(f'Outputs: {outputs}')
      logits = outputs.logits
      
      # Shift sequences for language modeling
      shift_logits = logits[..., :-1, :].contiguous()
      shift_labels = input_ids[..., 1:].to(torch.int64).contiguous()
      
      # Calculate loss
      loss = self.loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
      )
      
      return loss.view(shift_labels.size())
  # def get_batch(self, file: str):
  #   # We recreate np.memmap every batch to avoid a memory leak, as per
  #   # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
  #   data = np.memmap(file, dtype=np.uint16, mode='r')
  #   ix = torch.randint(len(data) - block_size, (batch_size,))
  #   x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
  #   y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
  #   if device_type == 'cuda':
  #     # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
  #     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
  #   else:
  #     x, y = x.to(device), y.to(device)
  #   return x, y
  #

  def evaluate_file(self) -> dict:
    """Evaluate cross entropy on a single tokenized file."""
    # Load tokenized data
    total_loss = 0
    total_tokens = 0
    
    # Process in batches
    for x, lens in get_story_iterator("weights/tiny_stories/TinyStories_1.safetensor"):
      
      # Calculate loss
      loss = self.calculate_loss(x, lens)
      print(f'Loss: {loss}')
      
      # Sum up losses
      total_loss += loss.sum().item()
      total_tokens += loss.numel()
    
    return {
      'average_loss': total_loss / total_tokens,
      'perplexity': torch.exp(torch.tensor(total_loss / total_tokens)).item(),
      'total_tokens': total_tokens
    }
  
  def evaluate_directory(self, tokenized_dir_path: str) -> dict:
    """Evaluate cross entropy on all tokenized files in a directory."""
    all_results = []
    
    # Process each file in directory
    for filename in tqdm(os.listdir(tokenized_dir_path)):
      if filename.endswith('.json'): # Assuming tokenized files are saved as JSON
        file_path = os.path.join(tokenized_dir_path, filename)
        results = self.evaluate_file(file_path)
        results['filename'] = filename
        all_results.append(results)
    
    # Calculate aggregate metrics
    total_loss_sum = sum(r['average_loss'] * r['total_tokens'] for r in all_results)
    total_tokens = sum(r['total_tokens'] for r in all_results)
    
    return {
      'per_file_results': all_results,
      'aggregate_metrics': {
        'average_loss': total_loss_sum / total_tokens,
        'perplexity': torch.exp(torch.tensor(total_loss_sum / total_tokens)).item(),
        'total_tokens': total_tokens
      }
    }
  
if __name__ == '__main__':
  l = Llama()
  l.evaluate_file()
