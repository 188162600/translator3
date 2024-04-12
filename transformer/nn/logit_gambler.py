import torch
import torch.nn as nn
import math

class LogitGambler(nn.Module):
    def __init__(self,  output_shape, temperature=1.0):
        super(LogitGambler, self).__init__()
      
        self.output_shape = output_shape
        self.temperature = temperature

        # Validate shapes
       
        if not isinstance(output_shape, (tuple, list)) or len(output_shape) == 0:
            raise ValueError("Output shape must be a non-empty tuple or list.")
       

        # Initialize learnable logits with the specified input shape
        self.logits = nn.Parameter(torch.zeros(*output_shape), requires_grad=True)
        torch.nn.init.uniform_(self.logits, a=math.log(0.01), b=math.log(1.0))  # Uniform initialization in log space

    def forward(self):
        # Check if the index tensor shape matches the input shape
        # assert isinstance(index,(tuple,list))
        # if len(index) != len(self.input_shape):
            # raise ValueError("Index tensor shape must match input shape.")
        # print(index.shape)
        # index_expand=index.view(self.index_expand_shape)
        # Select logits corresponding to the indices in the batch using advanced indexing
        selected_logits =self.logits

        # Adding Gumbel noise for the Gumbel-Softmax sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(selected_logits)))
        gumbel_logits = selected_logits + gumbel_noise

        # Applying the softmax at the specified temperature
        y_soft = torch.softmax(gumbel_logits / self.temperature, dim=-1)

        # Reshaping to the desired output shape, ensuring the total number of elements is compatible
        y_reshaped = y_soft.view(*self.output_shape)
        # print(y_soft)
        return y_reshaped
    def extra_repr(self) -> str:
        return f", output_shape={self.output_shape}, temperature={self.temperature}"