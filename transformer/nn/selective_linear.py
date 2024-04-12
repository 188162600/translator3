
from typing import Mapping
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import Module
from torch.nn import init
import torch.nn.functional as F
import math
import torch
from torch import Tensor
from torch.nn import Module, Parameter, init
import math
from typing import Any, Tuple


class SelectiveLinear(Module):
    def __init__(self, total_options:int,num_options: int, in_features: int, out_features: int, bias: bool = True, batch_index: int = 0, device=None, dtype=None):
        super(SelectiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_options = num_options
        self.total_options=total_options
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.is_non_selective = (self.num_options==1 or self.num_options is None) and ( self.total_options==1 or self.total_options is None)
        if self.is_non_selective:
            self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty((total_options, out_features, in_features), **factory_kwargs))
        if bias:
            if self.is_non_selective:
                self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.bias = Parameter(torch.empty(total_options, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.batch_index = batch_index
        
    def get_batched_weights_biases(self, selection_logits, temperature=1.0):
        assert selection_logits.dim() == 2
        
        # Calculate selection probabilities
        selection_probs = F.softmax(selection_logits / temperature, dim=-1)
        
        # Compute weighted sum of weights based on selection probabilities
        weighted_weights =self.activation( torch.einsum('nij,bn->bij', self.weight, selection_probs))
        
        if self.bias is not None:
            # Compute weighted sum of biases based on selection probabilities
            weighted_biases =self.activation( torch.einsum('ni,bn->bi', self.bias, selection_probs))
            return weighted_weights, weighted_biases
        return weighted_weights, None

        
        
        
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def forward_unbatched_logits(self, x, selection_logits):
        # print("selection_logits",selection_logits.shape,"x shape",x.shape)
        weighted_weight = torch.einsum('nij,n->ij', self.weight, selection_logits)
        # final_output = torch.einsum('ij,bnj->bni', weighted_weight, x)
        if self.bias is not None:
            weighted_biases = torch.einsum('ni,n->i', self.bias, selection_logits)
        else:
            weighted_biases = None
        return torch.nn.functional.linear(x, weighted_weight, weighted_biases)
            # final_output += weighted_biases.unsqueeze(1).expand(-1, x.size(1), -1)
        # return final_output
        
    
    def forward(self, x, selection_probs):
        # print("selection_logits",selection_logits.shape,"x shape",x.shape)
        # print(self.num_options,self.total_options,self.is_non_selective)
        if self.is_non_selective:
            # print("non selective")
            return torch.nn.functional.linear(x, self.weight, self.bias)
        if selection_probs.dim()==1:
            return self.forward_unbatched_logits(x, selection_probs)
        assert selection_probs.dim() == 2
        if self.batch_index != 0:
            x = x.transpose(0, self.batch_index)
        
        transformed = torch.einsum('nij,baj->bani', self.weights, x)
        # transformed=self.activation(transformed)
        # # transformed=self.activation(transformed)

        # # Compute the weighted sum of biases using the selection probabilities
        # weighted_biases = torch.einsum('ni,bn->bi', self.bias, selection_probs)
        # weighted_biases = self.activation(weighted_biases)
        # # weighted_biases = self.activation(weighted_biases)
        # # Sum the outputs using the selection probabilities to get the final output
        print("transformed",transformed.shape,"selection_probs",selection_probs.shape,"x shape",x.shape,"weight",self.weight.shape,"grad",torch.is_grad_enabled())
        final_output = torch.einsum('bani,bn->bai', transformed, selection_probs)
        # weighted_weight = torch.einsum('nij,bn->bij', self.weight, selection_probs)
       
       
        # final_output = torch.einsum('bij,bnj->bni', weighted_weight, x)
        
        
        
        if self.bias is not None:
            # Compute the weighted sum of biases using the selection probabilities
            weighted_biases = torch.einsum('ni,bn->bi', self.bias, selection_probs)
            # weighted_biases=self.activation(weighted_biases)
            # Add the weighted biases to the final output
            final_output += weighted_biases.unsqueeze(1).expand(-1, x.size(1), -1)
     
        if self.batch_index != 0:
            final_output = final_output.transpose(0, self.batch_index)
        
        return final_output
  

    def extra_repr(self) -> str:
        return f'total_options={self.total_options}, num_options={self.num_options}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        weight_key = prefix + 'weights'
        bias_key = prefix + 'bias'
        if weight_key in state_dict:
            weight = state_dict[weight_key]
            if weight.dim() == 2:
                weight = weight.unsqueeze(0).expand(self.num_options, -1, -1)
                state_dict[weight_key] = weight
                
        if bias_key in state_dict:
            bias = state_dict[bias_key]
            if bias.dim() == 1:
                bias = bias.unsqueeze(0).expand(self.num_options, -1)
                state_dict[bias_key] = bias
                
