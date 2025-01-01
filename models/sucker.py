from torch import nn
import torch.nn.functional as F

from transformers.models.seamless_m4t.modeling_seamless_m4t import shift_tokens_right

# This is parser for char based tokens
# Character are random embedding without any knowldege
# This will get knowldege out of stream of charcters.
# Like kitten and cat are different in character space but similar in meaning space.
# This is job of this Module.
# We will not run embedding of character, they will be basis vectors.
# We will have a decoder which given embedding can decode characters, from last to first, whereas encode encodes from first to last.
# So though we convert things to meaning space, but still we can decode them back to character space.
# For initial boost we will use already learnt embedding of models for getting training CNN.
# We will have each model linear decoder, with a very slow rate of decoder learner (so most of knowldege stays in model, instead of decoder layer)
# Open questions:
# Loss: It is assumed that log likelihood works better in n class classification, but even when those class have fixed (and orthogonal) embeddings?
# Knowledge less decoder? How to create them, maybe by only linear transforms randomly initialize with very slow learning rate.
# State space models or Mamba? Can get us of curse depth of CNN. But have opiniated decay and also input cotribution is independent of hidden state.
class Convulation(nn.Module):
  def __init__(self, dim: int):
    super().__init__()

  def forward(self, tokens):
    # tokens: batch, lenght, dimension
    shifted_tokens = F.pad(tokens[:, :-1], (0, 0, 1, 0)) # TODO: check this, this should shift memories to one right in lenght dim


class CNN(nn.Module):
  def __init__(self, dim: int):
    super().__init__()
    self.dim = dim
    
    # Directional encoder (left-to-right)
    self.encoder = nn.ModuleList([
      # Each layer can only see previous tokens
      nn.Conv1d(
        in_channels=embedding_dim if i == 0 else num_filters,
        out_channels=num_filters,
        kernel_size=3,
        padding=2, # Padding of 2 for causal convolution
        padding_mode='zeros'
      ) for i in range(3)
    ])
    self.encoder_relu = nn.ReLU()
    self.encoder_proj = nn.Linear(num_filters, dim)



