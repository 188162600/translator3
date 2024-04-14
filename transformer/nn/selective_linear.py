
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
    def __init__(self, total_options:int,num_options: int, in_features: int, out_features: int, bias: bool = True, batch_index: int = 0,default_index=None ,device=None, dtype=None):
        super(SelectiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_options = num_options
        self.total_options=total_options
        assert isinstance(default_index,(int,type(None)))
        self.default_index=default_index
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
        # print("selection_logits",selection_logits.shape,"x shape",x.shape,"weight",self.weight.shape,"grad",torch.is_grad_enabled())
        weighted_weight = torch.einsum('nij,n->ij', self.weight, selection_logits)
        # final_output = torch.einsum('ij,bnj->bni', weighted_weight, x)
        if self.bias is not None:
            weighted_biases = torch.einsum('ni,n->i', self.bias, selection_logits)
        else:
            weighted_biases = None
        return torch.nn.functional.linear(x, weighted_weight, weighted_biases)
            # final_output += weighted_biases.unsqueeze(1).expand(-1, x.size(1), -1)
        # return final_output
    def fill_with_default_index(self,noise_scale=0.000):
        if self.default_index is None:
            return
        if self.is_non_selective:
            return
        for i in range(self.total_options):
            if i!=self.default_index:
                self.weight.data[i]=self.weight.data[self.default_index]+torch.randn_like(self.weight.data[self.default_index])*noise_scale
                if self.bias is not None:
                    self.bias.data[i]=self.bias.data[self.default_index]+torch.randn_like(self.bias.data[self.default_index])*noise_scale
        
    
    def forward(self, x, selection_probs):
        # print("selection_logits",selection_logits.shape,"x shape",x.shape)
        # print(self.num_options,self.total_options,self.is_non_selective)
        
        if self.is_non_selective:
            # print("non selective")
            return torch.nn.functional.linear(x, self.weight, self.bias)
        if selection_probs is None:
            assert self.default_index is not None
            # print("selection_probs is None")
            # print("default_index",self.default_index,"weight",self.weight.shape,"bias",self.bias.shape,"x",x.shape) 
            # print("default_index",self.default_index,"weight",self.weight.shape,"bias",self.bias.shape,"x",x.shape)
            return torch.nn.functional.linear(x, self.weight[self.default_index], self.bias[self.default_index])
        if selection_probs.dim()==1:
            return self.forward_unbatched_logits(x, selection_probs)
        assert selection_probs.dim() == 2
        if self.batch_index != 0:
            x = x.transpose(0, self.batch_index)
        
        transformed = torch.einsum('nij,baj->bani', self.weight, x)
        # transformed=self.activation(transformed)
        # # transformed=self.activation(transformed)

        # # Compute the weighted sum of biases using the selection probabilities
        # weighted_biases = torch.einsum('ni,bn->bi', self.bias, selection_probs)
        # weighted_biases = self.activation(weighted_biases)
        # # weighted_biases = self.activation(weighted_biases)
        # # Sum the outputs using the selection probabilities to get the final output
        # print("transformed",transformed.shape,"selection_probs",selection_probs.shape,"x shape",x.shape,"weight",self.weight.shape,"grad",torch.is_grad_enabled())
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
    def get_non_selective_params_data(self):
        if self.is_non_selective:
            return [self.weight.data,self.bias.data if self.bias is not None else None]
        assert self.default_index is not None
        return [self.weight.data[self.default_index],self.bias.data[self.default_index] if self.bias is not None else None]
    # def _save_to_state_dict(self, destination, prefix, keep_vars):
    #     weight_key = prefix + 'weight'
    #     weights_key = prefix + 'weights'
    #     bias_key = prefix + 'bias'
    #     biases_key = prefix + 'biases'
    #     # print("keep_vars",keep_vars)
    #     if self.is_non_selective:
    #         destination[weight_key] = self.weight if keep_vars else self.weight.data
    #         destination[bias_key] = self.bias if keep_vars else self.bias.data
    #     else:
    #         destination[weights_key] = self.weight if keep_vars else self.weight.data
    #         destination[biases_key] = self.bias if keep_vars else self.bias.data
    #         if self.default_index is not None:
    #             destination[weight_key]=self.weight[self.default_index] if keep_vars else self.weight.data[self.default_index]
    #             destination[bias_key]=self.bias[self.default_index] if keep_vars else self.bias.data[self.default_index]
           
    #     # return super()._save_to_state_dict(destination, prefix, keep_vars)
        
    
    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        
    #     weight_key = prefix + 'weight'
    #     weights_key = prefix + 'weights'
    #     bias_key = prefix + 'bias'
    #     biases_key = prefix + 'biases'
    #     if weights_key in state_dict:
    #         weights = state_dict[weights_key]
    #         if weights.dim() == 3:
    #             state_dict[weight_key] = weights
    #     elif weight_key in state_dict:
    #         weight = state_dict[weight_key]
    #         if weight.dim() == 2:
    #             weight = weight.unsqueeze(0).expand(self.num_options, -1, -1)
    #             state_dict[weight_key] = weight
    #     if biases_key in state_dict:
    #         biases = state_dict[biases_key]
    #         if biases.dim() == 2:
    #             state_dict[bias_key] = biases
    #     elif bias_key in state_dict:
    #         bias = state_dict[bias_key]
    #         if bias.dim() == 1:
    #             bias = bias.unsqueeze(0).expand(self.num_options, -1)
    #             state_dict[bias_key] = bias
            
    #     return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        # if weights_key in state_dict:
        #     weights = state_dict[weights_key]
        #     if weights.dim() == 3:
        #         state_dict[weight_key] = weights
                
        # if weight_key in state_dict:
        #     weight = state_dict[weight_key]
        #     if weight.dim() == 2:
        #         weight = weight.unsqueeze(0).expand(self.num_options, -1, -1)
        #         state_dict[weight_key] = weight
        # elif weights_key in state_dict:
        #     weights = state_dict[weights_key]
        #     if weights.dim() == 3:
        #         state_dict[weight_key] = weights
        # if bias_key in state_dict:
        #     bias = state_dict[bias_key]
        #     if bias.dim() == 1:
        #         bias = bias.unsqueeze(0).expand(self.num_options, -1)
        #         state_dict[bias_key] = bias
            
        # elif biases_key in state_dict:
        #     biases = state_dict[biases_key]
        #     if biases.dim() == 2:
        #         state_dict[bias_key] = biases
            
        

                
