# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor
from ..nn.transformer_layer import TransformerEncoderLayerBase

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
    #transformer_layer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerEncoderBase":
        return "TransformerEncoder"
    else:
        return module_name

class NextSteps:
    def __init__(self,tensor):
        self.tensor = tensor
       
        self._indices =None
     
      
        self._softmax =None
     
       
        
        self._probability =None
      
        self.mapped_indices=None
        self._confidence=None
    def get_indices(self):
        if self._indices is None:
            self._indices = torch.argmax(self.tensor,dim=1)
        return self._indices
    def get_mapped_indices(self):
        return self.mapped_indices
    def get_softmax(self):
        #print(torch.is_grad_enabled())
        if self._softmax is None:
            #print(self.tensor.grad_fn,"tensor grad_fn",torch.nn.functional.softmax(self.tensor, dim=1).grad_fn)
            softmax_result_immediate = torch.nn.functional.softmax(self.tensor, dim=1)
            #print(f"Immediate softmax grad_fn: {softmax_result_immediate.grad_fn}")
            self._softmax = torch.nn.functional.softmax(self.tensor, dim=1)
           # print(self._softmax.grad_fn,"softmax grad_fn2")
        return self._softmax
   
    def get_probability(self):
        if self._probability is None:
            expanded_indices = self.get_indices().unsqueeze(-1)
            self._probability = torch. gather(self.get_softmax(), 1,expanded_indices).squeeze(-1)
            # print(self._probability.grad_fn,"grad_fn")
            # print(self.tensor.grad_fn,"grad_fn2")
            # print(self._softmax.grad_fn,"grad_fn3")
        return self._probability
   
    def get_confidence(self):
        if self._confidence is None:
            self._confidence=torch.sum(self.get_probability(),dim=1)
        #print(self._confidence.grad_fn,"conf")
        return self._confidence
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
        self.build_index_mapping(transformer_cfg, classifier_cfg, dictionary, embed_tokens)
    def build_index_mapping(self,transformer_cfg, classifier_cfg, dictionary, embed_tokens):
        
        
        total_new=0
       
        self.index_mapping=torch.nn.Parameter( torch.zeros(classifier_cfg.steps_classifier_classes,classifier_cfg.num_steps,dtype=torch.long),requires_grad=False)
        #self.register_buffer("index_mapping", self.index_mapping)
        def build_index(index,num_new,num_shared=0,index_shared=None):
            nonlocal total_new
            if index_shared is not None:
                assert 0<=index_shared<index<classifier_cfg.num_steps
            else:
                assert 0<=index<classifier_cfg.num_steps
            #assert num_new+num_shared==classifier_cfg.steps_classifier_classes
            
                
            self. index_mapping[0:num_new,index]= torch.arange(total_new,total_new+num_new)
            if num_shared>0:
                self.index_mapping[num_new:num_new+num_shared,index]=self.index_mapping[0:num_shared,index_shared]
            #self.index_mapping[num_new:num_new+num_shared,index]=self.index_mapping[0:num_shared,index_shared]
            total_new+=num_new
        def build_random_index(index,num_random,starts_at):
            self.index_mapping[starts_at:starts_at+num_random,index]=torch.randperm(classifier_cfg.steps_classifier_classes)[:num_random]    
        
           
        if classifier_cfg.sharing_method=="none":
            self.index_mapping=None
        if classifier_cfg.sharing_method=="cycle_rev":
            assert classifier_cfg.num_steps%2==0
            for i in range( classifier_cfg.num_steps//2):
                build_index(i,num_new=classifier_cfg.steps_classifier_classes)
            new=classifier_cfg.steps_classifier_classes-classifier_cfg.steps_classifier_shared_classes
            for i in range(classifier_cfg.num_steps//2,classifier_cfg.num_steps):
                build_index(i,num_new=new,num_shared=classifier_cfg.steps_classifier_shared_classes,index_shared=classifier_cfg.num_steps-i-1)
        if classifier_cfg.sharing_method=="random":
            new=classifier_cfg.steps_classifier_classes-classifier_cfg.steps_classifier_shared_classes
            for i in range(classifier_cfg.num_steps):
                build_index(i,num_new=new)
            for i in range(classifier_cfg.num_steps):
                build_random_index(i,  classifier_cfg.steps_classifier_shared_classes,new)
        print(classifier_cfg.total_options,total_new)
        assert classifier_cfg.total_options==total_new
            
               
          
            
        
            
    def build_output_projection(self,  transformer_cfg, classifier_cfg, dictionary, embed_tokens):
    
        self.output_projection =torch.nn.Linear(
            embed_tokens.embedding_dim,
            classifier_cfg.steps_classifier_classes*classifier_cfg.num_steps , bias=False
        )
    def build_encoder_layer(self, transformer_cfg):
        layer = TransformerEncoderLayerBase(
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
        #print("features",features.shape)
        
        output= self.output_projection(features[0])
        batch_size=output.size(0)
        
        return output.view(batch_size,self.classifier_cfg.steps_classifier_classes,self.classifier_cfg.num_steps )
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
        #src_tokens=src_tokens.permute(1,0,2)
        # if token_embeddings is not None:
        #     token_embeddings=token_embeddings.permute(1,0,2)
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
        next_steps=NextSteps(next_steps)
        #print("next_steps",next_steps.get_indices().shape,self.index_mapping.shape)
        if self.index_mapping is not None:
            next_steps.mapped_indices=torch.gather(self.index_mapping,0,next_steps.get_indices())
        else:
            next_steps.mapped_indices=next_steps.get_indices()
            #next_steps._indices= next_steps.get_indices()[self.index_mapping]
        
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
    
    # def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
    #     return super().load_state_dict(state_dict, strict)
class TransformerEncoderStepsClassifier(TransformerStepsClassifierBase):
    def __init__(self, transformer_cfg, dictionary, embed_tokens, return_fc=False):
      
        #print(transformer_cfg.encoder_steps_classifier is None)
        super().__init__(
           transformer_cfg,
           transformer_cfg.encoder,
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )

class TransformerDecoderStepsClassifier(TransformerStepsClassifierBase):
    def __init__(self, transformer_cfg, dictionary, embed_tokens, return_fc=False):
       
        super().__init__(
           transformer_cfg,
           transformer_cfg.decoder,
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )
        
    # def forward_scriptable(
    #     self,
    #     src_tokens=None,
    #     src_lengths: Optional[torch.Tensor] = None,
    #     return_all_hiddens: bool = False,
    #     token_embeddings: Optional[torch.Tensor] = None,
    # ):
    #     """
    #     Args:
    #         src_tokens (LongTensor): tokens in the source language of shape
    #             `(batch, src_len)`
    #         src_lengths (torch.LongTensor): lengths of each source sentence of
    #             shape `(batch)`
    #         return_all_hiddens (bool, optional): also return all of the
    #             intermediate hidden states (default: False).
    #         token_embeddings (torch.Tensor, optional): precomputed embeddings
    #             default `None` will recompute embeddings

    #     Returns:
    #         dict:
    #             - **encoder_out** (Tensor): the last encoder layer's output of
    #               shape `(src_len, batch, embed_dim)`
    #             - **encoder_padding_mask** (ByteTensor): the positions of
    #               padding elements of shape `(batch, src_len)`
    #             - **encoder_embedding** (Tensor): the (scaled) embedding lookup
    #               of shape `(batch, src_len, embed_dim)`
    #             - **encoder_states** (List[Tensor]): all intermediate
    #               hidden states of shape `(src_len, batch, embed_dim)`.
    #               Only populated if *return_all_hiddens* is True.
    #     """
    #     # compute padding mask
        
      

    #     # encoder_padding_mask = src_tokens.eq(self.padding_idx)
    #     # has_pads = (
    #     #     torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
    #     # )
    #     # # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
    #     # if torch.jit.is_scripting():
    #     #     has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

    #     # x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

    #     # # account for padding while computing the representation
    #     # x = x * (
    #     #     1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
    #     # )

    #     # # B x T x C -> T x B x C
    #     # x = x.transpose(0, 1)
    #     x=token_embeddings
    #     print(x.shape)

    #     encoder_states = []
    #     fc_results = []

    #     if return_all_hiddens:
    #         encoder_states.append(x)
    #     #encoder_padding_mask=torch.zeros(x.size(0),x.size(1)).to(x.device)
    #     # encoder layers
    #     for layer in self.layers:
    #         lr = layer(
    #             #x, encoder_padding_mask=encoder_padding_mask if has_pads else None
    #             x, encoder_padding_mask=None
    #         )

    #         if isinstance(lr, tuple) and len(lr) == 2:
    #             x, fc_result = lr
    #         else:
    #             x = lr
    #             fc_result = None

    #         if return_all_hiddens and not torch.jit.is_scripting():
    #             assert encoder_states is not None
    #             encoder_states.append(x)
    #             fc_results.append(fc_result)

    #     if self.layer_norm is not None:
    #         x = self.layer_norm(x)

    #     # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
    #     # `forward` so we use a dictionary instead.
    #     # TorchScript does not support mixed values so the values are all lists.
    #     # The empty list is equivalent to None.
    #     # src_lengths = (
    #     #     src_tokens.ne(self.padding_idx)
    #     #     .sum(dim=1, dtype=torch.int32)
    #     #     .reshape(-1, 1)
    #     #     .contiguous()
    #     # )
    #     next_steps=self.output_layer(x)
    #     return {
    #         "next_steps":[next_steps],
    #         "encoder_out": [x],  # T x B x C
    #         # "encoder_padding_mask": [encoder_padding_mask],  # B x T
    #         # "encoder_embedding": [encoder_embedding],  # B x T x C
    #         "encoder_states": encoder_states,  # List[T x B x C]
    #         "fc_results": fc_results,  # List[T x B x C]
    #         "src_tokens": [],
    #         #"src_lengths": [src_lengths],
    #     }