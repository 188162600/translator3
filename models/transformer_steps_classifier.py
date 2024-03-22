# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerEncoderBase":
        return "TransformerEncoder"
    else:
        return module_name
# class RestoredSteps:
#     def __init__(self,num_steps,num_options,num_samples,num_old,num_new,num_fresh,old_new_fresh_distribution=None) -> None:
#         self.softmax=torch.zeros(num_old+num_new+num_fresh,num_options,num_steps)
#         self.indices=torch.zeros(num_old+num_new+num_fresh,num_steps,dtype=torch.long)
#         self.losses=torch.zeros(num_old+num_new+num_fresh,num_samples)
#         self.occurrences=torch.zeros(num_old+num_new+num_fresh,dtype=torch.long)
#         self.old_new_fresh_distribution=old_new_fresh_distribution
        
        
#         self.current_sample=torch.zeros(num_old+num_new+num_fresh,dtype=torch.long)
#         self.num_samples=num_samples
#         self.tracking_index=num_new+num_old
        
         
#         torch.fill_(self.losses,math.inf)
       
#         self.num_fresh=num_fresh
#         self.num_old=num_old
#         self.num_new=num_new
#     def to(self,*args,**kwargs):
#         self.softmax=self.softmax.to(*args,**kwargs)
#         self.indices=self.indices.to(*args,**kwargs)
#         self.losses=self.losses.to(*args,**kwargs)
#         self.occurrences=self.occurrences.to(*args,**kwargs)
#         self.current_sample=self.current_sample.to(*args,**kwargs)
#         return self
#     @torch.no_grad()
#     def aggregate_losses(self):
#         # Unique occurrences
#         unique_occurrences, indices = torch.unique(self.occurrences, return_inverse=True)
#         # Initialize tensor for aggregated losses
#         aggregated_losses = torch.zeros_like(unique_occurrences, dtype=torch.float)

#         for i, occ in enumerate(unique_occurrences):
#             # Indices of all tracking events with the current occurrence
#             idx = (indices == i)
#             # Select the corresponding losses and sample counts
#             occ_losses = self.losses[idx]
#             occ_samples = self.current_sample[idx]

#             # Aggregate losses by calculating the mean for each occurrence
#             total_loss = 0
#             total_samples = 0
#             for j, samples in enumerate(occ_samples):
#                 if samples > 0:  # Ensure division is meaningful
#                     total_loss += occ_losses[j, :samples].sum()
#                     total_samples += samples
#             if total_samples > 0:  # Avoid division by zero
#                 aggregated_losses[i] = total_loss / total_samples
#             # else:
#             #     aggregated_losses[i]=math.inf

#         return unique_occurrences.float(), aggregated_losses
#     @torch.no_grad()
#     def linear_interp_loss(self, occurrence):
#         # Assuming torch_linear_interp is defined as before
#         occurrences, aggregated_losses = self.aggregate_losses()

#         # Make sure occurrences are sorted
#         sorted_indices = torch.argsort(occurrences)
#         sorted_occurrences = occurrences[sorted_indices]
#         sorted_losses = aggregated_losses[sorted_indices]

#         # Interpolate
#         interp_val = linear_interp(occurrence, sorted_occurrences, sorted_losses)
#         return interp_val
#     @torch.no_grad()
#     def get_losses(self):
#         sum_loss=self.losses.sum(dim=1)
#         return sum_loss/self.current_sample
#     @torch.no_grad()
#     def get_efficiency(self):
#         expected_loss=self.linear_interp_loss(self.occurrences)
#         diff= expected_loss-self.get_losses()
#         return diff
        
 
#     @torch.no_grad()
#     def track(self,loss,next_steps:NextSteps):
#         #print("loss",loss)
#         batch=next_steps.tensor.size(0)
#         if next_steps.restored_step_index is not None:
#             # index_start=next_steps.restored_step_index
#             # index_end=(next_steps.restored_step_index+batch)%(self.num_old+self.num_new)
#             index=next_steps.restored_step_index
            
#         else:
#             next_steps.indices
#             index_start=self.tracking_index
#             index_end=index_start+batch
#             #print(index_end)
#             if index_end>=self.num_old+self.num_new+self.num_fresh:
               
#                 index_start=self.num_old+self.num_new
#                 #print("tracking resetting to start")
#                 index_end=index_start+batch
#             self.tracking_index=index_end
#             index=torch.arange(index_start, index_end)
#         #print("index",index.shape)
#         n=index.size(0)
      
#         sample_index=self.current_sample[index]
#         # print(loss.shape,self.losses[index_start:index_end].shape)
#         # print("shape",self.losses[index_start:index_end].shape,self.losses[index_start:index_end][sample_index].shape,self.losses[index_start:index_end,sample_index].shape)
#         if loss.dim()==0:
            
#             self.losses[index[:,None] ,sample_index]=loss
#         else:
#             self.losses[index[:,None],sample_index]=loss[:n]
#         #print( self.softmax[index,sample_index].shape,next_steps.softmax[:n].shape)
#         self.softmax[index]=next_steps.softmax[:n]
#         self.indices[index]=next_steps.indices[:n]
#         # self.losses[index_start:index_end]=loss[:n]
#         self.occurrences[index]+=1
       
       
#         self.current_sample[index]=(self.current_sample[index]+1)%self.num_samples
#     @torch.no_grad()
#     def reset_fresh(self):
#         self.tracking_index=self.num_old+self.num_new
#         self.occurrences[self.num_new+self.num_old:]=0
#         self.current_sample[self.num_new+self.num_old:]=0
#         self.losses[self.num_new+self.num_old:]=math.inf
    
       
#     @torch.no_grad()
#     def update(self):
#         # indices=torch.argsort(self.get_efficiency(),descending=True)[self.num_old+self.num_new:]
#         # indices=torch.argsort(self.occurrences[indices],descending=True)
#         efficiency=self.get_efficiency()
#         efficiency[:self.num_old]=math.inf
#         sorted_indices_by_efficiency = torch.argsort(efficiency, descending=True)

#         # Select the subset of items, skipping the top self.num_old+self.num_new items.
#         subset_indices = sorted_indices_by_efficiency[self.num_old+self.num_new:]

#         # Now, get the occurrences of these selected items.
#         subset_occurrences = self.occurrences[subset_indices]

#         # Finally, sort these selected items by their occurrences in descending order.
#         # Note: We sort subset_occurrences, but we need to sort subset_indices based on these occurrences.
#         sorted_indices_by_occurrences = subset_indices[torch.argsort(subset_occurrences, descending=True)]

#         self.softmax[self.num_old+self.num_new:]=self.softmax[sorted_indices_by_occurrences]
#         self.indices[self.num_old+self.num_new]=self.indices[sorted_indices_by_occurrences]
#         self.losses[self.num_old+self.num_new]=self.losses[sorted_indices_by_occurrences]
#         self.occurrences[self.num_old+self.num_new]=self.occurrences[sorted_indices_by_occurrences]
#         self.current_sample[self.num_old+self.num_new]=self.current_sample[sorted_indices_by_occurrences]
#         self.reset_fresh()
        
        
        
#     def get_new(self,next_steps:NextSteps):
#        # print("self.softmax.shape,next_steps.softmax.shape",self.softmax.shape,next_steps.softmax.shape)
       
#         softmax=self.softmax[self.num_old:self.num_old+self.num_new].view(self.num_new,-1)
#         next_steps_softmax=next_steps.softmax.view(next_steps.softmax.size(0),-1)
#         with torch.no_grad():
#             similarity=cosine_similarity_2d(softmax,next_steps_softmax).detach()
        
#             index=torch.argmax(similarity,dim=0).detach()

       
#         confidence=torch.gather(similarity,0,index.unsqueeze(0)).squeeze(0) *next_steps.confidence
#         # print("softmax--",self.softmax[index].shape,next_steps.softmax.shape)
#         # print("indices--",self.indices[index].shape,next_steps.indices.shape)
#         # print("confidence",confidence.shape,next_steps.confidence.shape)
        
#         return NextSteps(self.softmax[index],self.indices[index],self.softmax[index],None,confidence=confidence,restored_step_index=index)
#     def get_old(self,next_steps:NextSteps):
#         #print("self.softmax.shape,next_steps.softmax.shape",self.softmax.shape,next_steps.softmax.shape)
#         softmax=self.softmax[:self.num_old].view(self.num_old,-1)
#         next_steps_softmax=next_steps.softmax.view(next_steps.softmax.size(0),-1)
#         with torch.no_grad():
#             similarity=cosine_similarity_2d(softmax,next_steps_softmax).detach()
        
#             index=torch.argmax(similarity,dim=0).detach()
        
        
#         confidence=torch.gather(similarity ,0,index.unsqueeze(0)).squeeze(0)*next_steps.confidence
#         # print("softmax--",self.softmax[index].shape,next_steps.softmax.shape)
#         # print("indices--",self.indices[index].shape,next_steps.indices.shape)
#         # print("confidence",confidence.shape,next_steps.confidence.shape)
#         return NextSteps(self.softmax[index],self.indices[index],self.softmax[index],None,confidence=confidence,restored_step_index=index)
#     def get_fresh(self,next_steps):
#         return next_steps
#     def get_random(self, next_steps):
#         weight=list(self.old_new_fresh_distribution)
#         #print(self.losses[0][0].item() ,self.losses[self.num_old][0].item() == math.inf)
#         if self.losses[0][0].item() == math.inf:
#             weight[0]=0
#         if self.losses[self.num_old][0].item() == math.inf:
#             weight[1]=0
#         which=random.choices((self.get_old,self.get_new,self.get_fresh),weights=weight,k=1)[0]
#         return which(next_steps)
        
    
class NextSteps:
    def __init__(self,tensor):
        self.tensor = tensor
       
        self.indices =None
     
      
        self.softmax =None
     
       
        
        self.probability =None
      
       
        self.confidence=torch.sum(self.probability,dim=1)
    def get_indices(self):
        if self.indices is None:
            self.indices = torch.argmax(self.tensor,dim=1)
        return self.indices
    def get_softmax(self):
        if self.softmax is None:
            self.softmax = torch.softmax(self.tensor, dim=1)
        return self.softmax
    def get_probability(self):
        if self.probability is None:
            expanded_indices = self.get_indices().unsqueeze(-1)
            self.probability = torch.gather(self.get_softmax(), 1,expanded_indices).squeeze(-1)
        return self.probability
    def get_confidence(self):
        if self.confidence is None:
            self.confidence=torch.sum(self.get_probability(),dim=1)
        return self.confidence
      
      
     
# class TransformerStepsClassifierBase(FairseqEncoder):
#     """
#     Transformer encoder consisting of *transformer_cfg.layers* layers. Each layer
#     is a :class:`TransformerEncoderLayer`.

#     Args:
#         args (argparse.Namespace): parsed command-line arguments
#         dictionary (~fairseq.data.Dictionary): encoding dictionary
#         embed_tokens (torch.nn.Embedding): input embedding
#     """

#     def __init__(self, transformer_transformer_cfg,transformer_cfg, dictionary, embed_tokens, return_fc=False,adaptive_softmax = None):
#         self.transformer_cfg = transformer_cfg
#         self.transformer_transformer_cfg=transformer_transformer_cfg
#         super().__init__(dictionary)
#         self.register_buffer("version", torch.Tensor([3]))

#         self.dropout_module = FairseqDropout(
#             transformer_cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
#         )
    
#         self.encoder_layerdrop = transformer_cfg.layerdrop
#         self.return_fc = return_fc
        

#         embed_dim = embed_tokens.embedding_dim
#         self.padding_idx = embed_tokens.padding_idx
#         self.max_source_positions = transformer_transformer_cfg.max_source_positions

#         self.embed_tokens = embed_tokens

#         self.embed_scale = 1.0 if transformer_cfg.no_scale_embedding else math.sqrt(embed_dim)

#         self.embed_positions = (
#             PositionalEmbedding(
#                 transformer_transformer_cfg.max_source_positions,
#                 embed_dim,
#                 self.padding_idx,
#                 learned=transformer_transformer_cfg.learned_pos,
#             )
#             if not transformer_transformer_cfg.no_token_positional_embeddings
#             else None
#         )
#         if transformer_cfg.layernorm_embedding:
#             self.layernorm_embedding = LayerNorm(embed_dim, export=transformer_cfg.export)
#         else:
#             self.layernorm_embedding = None

#         if not transformer_cfg.adaptive_input and transformer_cfg.quant_noise.pq > 0:
#             self.quant_noise = apply_quant_noise_(
#                 nn.Linear(embed_dim, embed_dim, bias=False),
#                 transformer_cfg.quant_noise.pq,
#                 transformer_cfg.quant_noise.pq_block_size,
#             )
#         else:
#             self.quant_noise = None

#         if self.encoder_layerdrop > 0.0:
#             self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
#         else:
#             self.layers = nn.ModuleList([])
#         self.layers.extend(
#             [self.build_encoder_layer(transformer_cfg) for i in range(transformer_cfg.layers)]
#         )
#         self.num_layers = len(self.layers)

#         if transformer_cfg.normalize_before:
#             self.layer_norm = LayerNorm(embed_dim, export=transformer_cfg.export)
#         else:
#             self.layer_norm = None
            
#         self.adaptive_softmax = None
#         self.build_output_projection(transformer_cfg,transformer_cfg, dictionary, embed_tokens)
#         #self.build_restored_steps(transformer_cfg,transformer_cfg,dictionary,embed_tokens)
#     #     self.bui
#     # def build_restored_steps(self,transformer_cfg,transformer_cfg,dictionary,embed_tokens):
#     #     return RestoredSteps(transformer_cfg.num_steps,transformer_cfg.num_options,transformer_cfg.num_samples,transformer_cfg.num_old_new_fresh,transformer_cfg.old_new_fresh_distribution)
            
#     def build_output_projection(self, transformer_cfg,transformer_cfg, dictionary, embed_tokens):
       
#             self.output_projection =torch.nn.Linear(
#                 self.output_embed_dim, len(dictionary), bias=False
#             )
#             nn.init.normal_(
#                 self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
#             )
      

#     def build_encoder_layer(self, transformer_cfg):
#         layer = transformer_layer.TransformerEncoderLayerBase(
#             transformer_cfg, return_fc=self.return_fc
#         )
#         checkpoint = transformer_cfg.checkpoint_activations
#         if checkpoint:
#             offload_to_cpu = transformer_cfg.offload_activations
#             layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
#         # if we are checkpointing, enforce that FSDP always wraps the
#         # checkpointed layer, regardless of layer size
#         min_params_to_wrap = transformer_cfg.min_params_to_wrap if not checkpoint else 0
#         layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
#         return layer

#     def forward_embedding(
#         self, src_tokens, token_embedding: Optional[torch.Tensor] = None
#     ):
#         # embed tokens and positions
#         if token_embedding is None:
#             token_embedding = self.embed_tokens(src_tokens)
#         x = embed = self.embed_scale * token_embedding
#         if self.embed_positions is not None:
#             x = embed + self.embed_positions(src_tokens)
#         if self.layernorm_embedding is not None:
#             x = self.layernorm_embedding(x)
#         x = self.dropout_module(x)
#         if self.quant_noise is not None:
#             x = self.quant_noise(x)
#         return x, embed
    
#     def output_layer(self, features):
#         """Project features to the vocabulary size."""
#         if self.adaptive_softmax is None:
#             # project back to size of vocabulary
#             output= self.output_projection(features[0])
#         else:
#             output= self.adaptive_softmax( features)
#         batch_size = features.size(0)
#         return output.view(batch_size,self.transformer_cfg.num_classes,self.transformer_cfg.num_steps )
        
#     def forward(
#         self,
#         src_tokens,
#         src_lengths: Optional[torch.Tensor] = None,
#         return_all_hiddens: bool = False,
#         token_embeddings: Optional[torch.Tensor] = None,
#     ):
#         """
#         Args:
#             src_tokens (LongTensor): tokens in the source language of shape
#                 `(batch, src_len)`
#             src_lengths (torch.LongTensor): lengths of each source sentence of
#                 shape `(batch)`
#             return_all_hiddens (bool, optional): also return all of the
#                 intermediate hidden states (default: False).
#             token_embeddings (torch.Tensor, optional): precomputed embeddings
#                 default `None` will recompute embeddings

#         Returns:
#             dict:
#                 - **encoder_out** (Tensor): the last encoder layer's output of
#                   shape `(src_len, batch, embed_dim)`
#                 - **encoder_padding_mask** (ByteTensor): the positions of
#                   padding elements of shape `(batch, src_len)`
#                 - **encoder_embedding** (Tensor): the (scaled) embedding lookup
#                   of shape `(batch, src_len, embed_dim)`
#                 - **encoder_states** (List[Tensor]): all intermediate
#                   hidden states of shape `(src_len, batch, embed_dim)`.
#                   Only populated if *return_all_hiddens* is True.
#         """
#         out= self.forward_scriptable(
#             src_tokens, src_lengths, return_all_hiddens, token_embeddings
#         )["encoder_out"]
#         return self.output_layer(out)
        

#     # TorchScript doesn't support super() method so that the scriptable Subclass
#     # can't access the base class model in Torchscript.
#     # Current workaround is to add a helper function with different name and
#     # call the helper function from scriptable Subclass.
#     def forward_scriptable(
#         self,
#         src_tokens,
#         src_lengths: Optional[torch.Tensor] = None,
#         return_all_hiddens: bool = False,
#         token_embeddings: Optional[torch.Tensor] = None,
#     ):
#         """
#         Args:
#             src_tokens (LongTensor): tokens in the source language of shape
#                 `(batch, src_len)`
#             src_lengths (torch.LongTensor): lengths of each source sentence of
#                 shape `(batch)`
#             return_all_hiddens (bool, optional): also return all of the
#                 intermediate hidden states (default: False).
#             token_embeddings (torch.Tensor, optional): precomputed embeddings
#                 default `None` will recompute embeddings

#         Returns:
#             dict:
#                 - **encoder_out** (Tensor): the last encoder layer's output of
#                   shape `(src_len, batch, embed_dim)`
#                 - **encoder_padding_mask** (ByteTensor): the positions of
#                   padding elements of shape `(batch, src_len)`
#                 - **encoder_embedding** (Tensor): the (scaled) embedding lookup
#                   of shape `(batch, src_len, embed_dim)`
#                 - **encoder_states** (List[Tensor]): all intermediate
#                   hidden states of shape `(src_len, batch, embed_dim)`.
#                   Only populated if *return_all_hiddens* is True.
#         """
#         # compute padding mask
#         encoder_padding_mask = src_tokens.eq(self.padding_idx)
#         has_pads = (
#             torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
#         )
#         # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
#         if torch.jit.is_scripting():
#             has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

#         x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

#         # account for padding while computing the representation
#         x = x * (
#             1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
#         )

#         # B x T x C -> T x B x C
#         x = x.transpose(0, 1)

#         encoder_states = []
#         fc_results = []

#         if return_all_hiddens:
#             encoder_states.append(x)

#         # encoder layers
#         for layer in self.layers:
#             lr = layer(
#                 x, encoder_padding_mask=encoder_padding_mask if has_pads else None
#             )

#             if isinstance(lr, tuple) and len(lr) == 2:
#                 x, fc_result = lr
#             else:
#                 x = lr
#                 fc_result = None

#             if return_all_hiddens and not torch.jit.is_scripting():
#                 assert encoder_states is not None
#                 encoder_states.append(x)
#                 fc_results.append(fc_result)

#         if self.layer_norm is not None:
#             x = self.layer_norm(x)

#         # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
#         # `forward` so we use a dictionary instead.
#         # TorchScript does not support mixed values so the values are all lists.
#         # The empty list is equivalent to None.
#         src_lengths = (
#             src_tokens.ne(self.padding_idx)
#             .sum(dim=1, dtype=torch.int32)
#             .reshape(-1, 1)
#             .contiguous()
#         )
#         return {
#             "encoder_out": [x],  # T x B x C
#             "encoder_padding_mask": [encoder_padding_mask],  # B x T
#             "encoder_embedding": [encoder_embedding],  # B x T x C
#             "encoder_states": encoder_states,  # List[T x B x C]
#             "fc_results": fc_results,  # List[T x B x C]
#             "src_tokens": [],
#             "src_lengths": [src_lengths],
#         }

#     @torch.jit.export
#     def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
#         """
#         Reorder encoder output according to *new_order*.

#         Args:
#             encoder_out: output from the ``forward()`` method
#             new_order (LongTensor): desired order

#         Returns:
#             *encoder_out* rearranged according to *new_order*
#         """
#         if len(encoder_out["encoder_out"]) == 0:
#             new_encoder_out = []
#         else:
#             new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
#         if len(encoder_out["encoder_padding_mask"]) == 0:
#             new_encoder_padding_mask = []
#         else:
#             new_encoder_padding_mask = [
#                 encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
#             ]
#         if len(encoder_out["encoder_embedding"]) == 0:
#             new_encoder_embedding = []
#         else:
#             new_encoder_embedding = [
#                 encoder_out["encoder_embedding"][0].index_select(0, new_order)
#             ]

#         if len(encoder_out["src_tokens"]) == 0:
#             src_tokens = []
#         else:
#             src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

#         if len(encoder_out["src_lengths"]) == 0:
#             src_lengths = []
#         else:
#             src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

#         encoder_states = encoder_out["encoder_states"]
#         if len(encoder_states) > 0:
#             for idx, state in enumerate(encoder_states):
#                 encoder_states[idx] = state.index_select(1, new_order)

#         return {
#             "encoder_out": new_encoder_out,  # T x B x C
#             "encoder_padding_mask": new_encoder_padding_mask,  # B x T
#             "encoder_embedding": new_encoder_embedding,  # B x T x C
#             "encoder_states": encoder_states,  # List[T x B x C]
#             "src_tokens": src_tokens,  # B x T
#             "src_lengths": src_lengths,  # B x 1
#         }

#     @torch.jit.export
#     def _reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
#         """Dummy re-order function for beamable enc-dec attention"""
#         return encoder_out

#     def max_positions(self):
#         """Maximum input length supported by the encoder."""
#         if self.embed_positions is None:
#             return self.max_source_positions
#         return min(self.max_source_positions, self.embed_positions.max_positions)

#     def upgrade_state_dict_named(self, state_dict, name):
#         """Upgrade a (possibly old) state dict for new versions of fairseq."""
#         for i in range(self.num_layers):
#             # update layer norms
#             self.layers[i].upgrade_state_dict_named(
#                 state_dict, "{}.layers.{}".format(name, i)
#             )

#         version_key = "{}.version".format(name)
#         if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
#             # earlier checkpoints did not normalize after the stack of layers
#             self.layer_norm = None
#             self.normalize = False
#             state_dict[version_key] = torch.Tensor([1])
#         return state_dict


# # class TransformerStepsClassifier(TransformerStepsClassifierBase):
# #     def __init__(self, args, dictionary, embed_tokens, return_fc=False):
# #         self.args = args
# #         super().__init__(
# #             TransformerConfig.from_namespace(args),
# #             dictionary,
# #             embed_tokens,
# #             return_fc=return_fc,
# #         )

# #     def build_encoder_layer(self, args):
# #         return super().build_encoder_layer(
# #             TransformerConfig.from_namespace(args),
# #         )


class TransformerStepsClassifierBase(FairseqEncoder):
    """
    Transformer encoder consisting of *classifier_cfg.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, transformer_cfg, classifier_cfg,dictionary, embed_tokens, return_fc=False):
        self.transformer_cfg = transformer_cfg
        self.classifier_cfg=classifier_cfg
        super().__init__(dictionary)
        
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            transformer_cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.encoder_layerdrop = classifier_cfg.layerdrop
        self.return_fc = return_fc

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
       

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if transformer_cfg.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                transformer_cfg.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=classifier_cfg.learned_pos,
            )
            if not transformer_cfg.no_token_positional_embeddings
            else None
        )
        if transformer_cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=transformer_cfg.export)
        else:
            self.layernorm_embedding = None

        if not transformer_cfg.adaptive_input and transformer_cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                transformer_cfg.quant_noise.pq,
                transformer_cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(transformer_cfg) for i in range(classifier_cfg.layers)]
        )
        self.num_layers = len(self.layers)

        if classifier_cfg.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=transformer_cfg.export)
        else:
            self.layer_norm = None
        self.build_output_projection(transformer_cfg, classifier_cfg, dictionary, embed_tokens)
        
    def build_output_projection(self,  transformer_cfg, classifier_cfg, dictionary, embed_tokens):
    
        self.output_projection =torch.nn.Linear(
            embed_tokens.embedding_dim,
            classifier_cfg.num_classes*classifier_cfg.num_steps , bias=False
        )
    def build_encoder_layer(self, transformer_cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(
            transformer_cfg, return_fc=self.return_fc
        )
        checkpoint = transformer_cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = transformer_cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = transformer_cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    def output_layer(self, features):
        
        output= self.output_projection(features[0])
        batch_size=output.size(0)
        return output.view(batch_size,self.classifier_cfg.num_classes,self.classifier_cfg.num_steps )
    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (
            torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        )
        # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        x = x * (
            1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
        )

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            lr = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        next_steps=self.output_layer(x)
        return {
            "next_steps":[next_steps],
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    # @torch.jit.export
    # def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
    #     """
    #     Reorder encoder output according to *new_order*.

    #     Args:
    #         encoder_out: output from the ``forward()`` method
    #         new_order (LongTensor): desired order

    #     Returns:
    #         *encoder_out* rearranged according to *new_order*
    #     """
    #     if len(encoder_out["encoder_out"]) == 0:
    #         new_encoder_out = []
    #     else:
    #         new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
    #     if len(encoder_out["encoder_padding_mask"]) == 0:
    #         new_encoder_padding_mask = []
    #     else:
    #         new_encoder_padding_mask = [
    #             encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
    #         ]
    #     if len(encoder_out["encoder_embedding"]) == 0:
    #         new_encoder_embedding = []
    #     else:
    #         new_encoder_embedding = [
    #             encoder_out["encoder_embedding"][0].index_select(0, new_order)
    #         ]

    #     if len(encoder_out["src_tokens"]) == 0:
    #         src_tokens = []
    #     else:
    #         src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

    #     if len(encoder_out["src_lengths"]) == 0:
    #         src_lengths = []
    #     else:
    #         src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

    #     encoder_states = encoder_out["encoder_states"]
    #     if len(encoder_states) > 0:
    #         for idx, state in enumerate(encoder_states):
    #             encoder_states[idx] = state.index_select(1, new_order)

    #     return {
    #         "encoder_out": new_encoder_out,  # T x B x C
    #         "encoder_padding_mask": new_encoder_padding_mask,  # B x T
    #         "encoder_embedding": new_encoder_embedding,  # B x T x C
    #         "encoder_states": encoder_states,  # List[T x B x C]
    #         "src_tokens": src_tokens,  # B x T
    #         "src_lengths": src_lengths,  # B x 1
    #     }

    # @torch.jit.export
    # def _reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
    #     """Dummy re-order function for beamable enc-dec attention"""
    #     return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict
class TransformerEncoderStepsClassifier(TransformerStepsClassifierBase):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        transformer_cfg= TransformerConfig.from_namespace(args)
        super().__init__(
           transformer_cfg,
           transformer_cfg.encoder_steps_classifier,
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )
    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args),
        )
class TransformerDecoderStepsClassifier(TransformerStepsClassifierBase):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        transformer_cfg= TransformerConfig.from_namespace(args)
        super().__init__(
           transformer_cfg,
           transformer_cfg.decoder_steps_classifier,
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )
    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args),
        )