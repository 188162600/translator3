
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
    def __init__(self, num_options: int, in_features: int, out_features: int, bias: bool = True, batch_index: int = 0, device=None, dtype=None):
        super(SelectiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_options = num_options
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty((num_options, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(num_options, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.batch_index = batch_index
        self.activation = torch.nn.Tanh()
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

    def forward(self, x, selection_logits, temperature=1.0):
        print("selection_logits",selection_logits.shape,"x shape",x.shape)
        if self.batch_index != 0:
            x = x.transpose(0, self.batch_index)
        
        selection_probs = F.softmax(selection_logits / temperature, dim=-1)
        weighted_weights = torch.einsum('nij,baj->bani', self.weight, x)
        weighted_weights = self.activation(weighted_weights)
        
        final_output = torch.einsum('bani,bn->bai', weighted_weights, selection_probs)

        if self.bias is not None:
            weighted_biases = torch.einsum('ni,bn->bi', self.bias, selection_probs)
            weighted_biases=self.activation(weighted_biases)
            final_output += weighted_biases.unsqueeze(1).expand(-1, final_output.size(1), -1)

        if self.batch_index != 0:
            final_output = final_output.transpose(0, self.batch_index)
        
        return final_output

    def extra_repr(self) -> str:
        return f'num_options={self.num_options}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

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
                
                
# class SelectiveLinear(Module):
#     def __init__(self, num_options: int, in_features: int, out_features: int, bias: bool = True,batch_index:int=0, device=None,
#                  dtype=None) -> None:
#         super(SelectiveLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_options = num_options
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         self.weight = Parameter(torch.empty((num_options, out_features, in_features), **factory_kwargs))
#         if bias:
#             self.bias = Parameter(torch.empty(num_options, out_features, **factory_kwargs))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#         self.batch_index=batch_index
#         #print("init weight.shape",self.weight.shape)

#     def reset_parameters(self) -> None:
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)

#     def forward(self, input: Tensor, index: Tensor) -> Tensor:
#         # Ensure the last dimension matches in_features
#         #print("weight.shape",self.weight.shape)
#         assert input.size(
#             -1) == self.in_features, f"Expected the last dimension of input to be {self.in_features}, got {input.size(-1)}"

#         batch_size=input.size(self.batch_index)
#        # print("batch_size",input.shape, batch_size, index.shape)
#         #print(index.shape)
#         assert batch_size == index.size(0), "Input batch size and index size should match"

#         # Select weights and biases for each item in the batch based on the index
#         selected_weight = self.weight[index]  # Shape: [batch_size, out_features, in_features]
#         selected_bias = self.bias[index] if self.bias is not None else None  # Shape: [batch_size, out_features]

#         # Reshape input to [batch_size, other_dims, in_features] for batch matrix multiplication
#         input_reshaped = input.reshape(-1, self.in_features)  # Flatten input while keeping the last dimension
#         #print(input_reshaped.view(batch_size, -1, self.in_features).shape, selected_weight.transpose(1, 2).shape)
#         output = torch.bmm(input_reshaped.view(batch_size, -1, self.in_features), selected_weight.transpose(1, 2))

#         if selected_bias is not None:
#             output += selected_bias.unsqueeze(1)

#         # Reshape output back to original input shape with out_features as the last dimension
#         output_dims = input.shape[:-1] + (self.out_features,)
#         return output.view(*output_dims)

#     def extra_repr(self) -> str:
#         return f'num_options={self.num_options}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
#         weight_key = prefix + 'weight'
#         bias_key = prefix + 'bias'
#         if weight_key  in state_dict:
            
#             weight=state_dict[weight_key] 
#             #print("weight.shape",weight.shape)
#             if weight.dim()==2:
#                 weight=weight.unsqueeze(0).expand(self.num_options,-1,-1)
               
#                 state_dict[weight_key]=weight
                
#         if bias_key  in state_dict:
#             bias=state_dict[bias_key]
#             #print("bias.shape",bias.shape)
#             if bias.dim()==1:
#                 bias=bias.unsqueeze(0).expand(self.num_options,-1)
                
#                 state_dict[bias_key]=bias
#         return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
#         super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
         
    # def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
       
    #     if "weight" in state_dict:
    #         weight=state_dict["weight"]
    #         if weight.dim()==2:
    #             weight=weight.unsqueeze(0).expand(self.num_options,-1,-1)
    #             state_dict["weight"]=weight
    #     if "bias" in state_dict:
    #         bias=state_dict["bias"]
    #         if bias.dim()==1:
    #             bias=bias.unsqueeze(0).expand(self.num_options,-1)
    #             state_dict["bias"]=bias
            
    #     return super().load_state_dict(state_dict, strict)

# class SelectiveLinear2d(Module):
#     def __init__(self, num_options: int, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
#         super(SelectiveLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_options = num_options
#         # Using factory_kwargs for device and dtype compatibility
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         self.weight = Parameter(torch.empty((num_options, out_features, in_features), **factory_kwargs))
#         if bias:
#             self.bias = Parameter(torch.empty(num_options, out_features, **factory_kwargs))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             init.uniform_(self.bias, -bound, bound)

#     def forward(self, input: Tensor, index: Tensor) -> Tensor:
#         assert input.size(0) == index.size(0), "Input batch size and index size should match"
#         batch_size = input.size(0)
#         # Selecting weights and biases based on index for each batch item
#         selected_weight = self.weight[index].view(batch_size, self.out_features, self.in_features)
#         selected_bias = self.bias[index].view(batch_size, self.out_features) if self.bias is not None else None
#         # Using batched matrix multiplication and addition for efficiency
#         output = torch.bmm(input.view(batch_size, 1, self.in_features), selected_weight.transpose(1, 2))
#         output = output.squeeze(1)  # Removing the middle dimension
#         if selected_bias is not None:
#             output += selected_bias
#         return output

#     def extra_repr(self) -> str:
#         return f'num_options={self.num_options}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

# class SelectiveLinear(Module):
#     def __init__(self, num_options: int, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
#         super(SelectiveLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_options = num_options
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         self.weight = Parameter(torch.empty((num_options, out_features, in_features), **factory_kwargs))
#         if bias:
#             self.bias = Parameter(torch.empty(num_options, out_features, **factory_kwargs))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)

#     def forward(self, input: Tensor, index: Tensor) -> Tensor:
#         # Ensure the last dimension matches in_features
#         assert input.size(-1) == self.in_features, f"Expected the last dimension of input to be {self.in_features}, got {input.size(-1)}"

#         batch_size, *other_dims, _ = input.shape
#         print("batch_size",batch_size,index.shape)
#         # assert batch_size == index.size(0), "Input batch size and index size should match"

#         # Select weights and biases for each item in the batch based on the index
#         selected_weight = self.weight[index]  # Shape: [batch_size, out_features, in_features]
#         selected_bias = self.bias[index] if self.bias is not None else None  # Shape: [batch_size, out_features]

#         # Reshape input to [batch_size, other_dims, in_features] for batch matrix multiplication
#         input_reshaped = input.view(-1, self.in_features)  # Flatten input while keeping the last dimension
#         output = torch.bmm(input_reshaped.view(batch_size, -1, self.in_features), selected_weight.transpose(1, 2))

#         if selected_bias is not None:
#             output += selected_bias.unsqueeze(1)

#         # Reshape output back to original input shape with out_features as the last dimension
#         output_dims = input.shape[:-1] + (self.out_features,)
#         return output.view(*output_dims)

#     def extra_repr(self) -> str:
#         return f'num_options={self.num_options}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
#     def copy_from(self,other):
    
#         assert self.in_features==other.in_features
#         assert self.out_features==other.out_features
#         if isinstance(other,torch.nn.Linear):
#             self.weight.data=other.weight.data.unsqueeze(0).expand(self.num_options,-1,-1)
#             if self.bias is not None:
#                 self.bias.data=other.bias.data.unsqueeze(0).expand(self.num_options,-1)
#         elif isinstance(other,SelectiveLinear):
#             self.weight.data=other.weight.data
#             if self.bias is not None:
#                 self.bias.data=other.bias.data
#         raise NotImplementedError("Copy from not implemented for ",type(other))
            
# linear= SelectiveLinear(3,4,5)
# input=torch.randn(2,10,4)

# linear2= torch.nn.Linear(4,5)
# print(linear2(input).shape)

# print(linear(input,torch.tensor([0,1])).shape)
