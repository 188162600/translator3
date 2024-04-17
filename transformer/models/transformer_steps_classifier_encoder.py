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
    transformer_layer
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from ..nn.zero_lowest_k import zero_lowest_k
# from ..nn import transformer_layer
# import logging
# logger=logging.getLogger(__name__)

# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerEncoderBase":
        return "TransformerEncoder"
    else:
        return module_name


class TransformerStepsClassifierEncoderBase(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, classifier_cfg,dictionary=None, embed_tokens=None, return_fc=False):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.embed_tokens = embed_tokens
        self.encoder_layerdrop = cfg.encoder.layerdrop
        self.return_fc = return_fc
        if embed_tokens is not None:
           
           
            # if embed_dim is not None:
            embed_dim = embed_tokens.embedding_dim
            self.padding_idx = embed_tokens.padding_idx
            self.max_source_positions = cfg.max_source_positions

           

            self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

            self.embed_positions = (
                PositionalEmbedding(
                    cfg.max_source_positions,
                    embed_dim,
                    self.padding_idx,
                    learned=cfg.encoder.learned_pos,
                )
                if not cfg.no_token_positional_embeddings
                else None
            )
            if cfg.layernorm_embedding:
                self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
            else:
                self.layernorm_embedding = None

            if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
                self.quant_noise = apply_quant_noise_(
                    nn.Linear(embed_dim, embed_dim, bias=False),
                    cfg.quant_noise.pq,
                    cfg.quant_noise.pq_block_size,
                )
            else:
                self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(classifier_cfg.classifier_encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(
            cfg, return_fc=self.return_fc
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(
        self, src_tokens
    ):
        # embed tokens and positions
        # if token_embedding is None:
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
        src_tokens: Optional[torch.Tensor] = None,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        previous_encode: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
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
            src_tokens, src_lengths, return_all_hiddens, previous_encode, padding_mask
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens: Optional[torch.Tensor] = None,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        previous_encode: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
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
        assert (src_tokens is not None and self.embed_tokens is not None) or (previous_encode is not None and padding_mask is not None)
        # compute padding mask
        
        if padding_mask is not None:
            encoder_padding_mask=padding_mask
            has_pads=encoder_padding_mask.any()
            if torch.jit.is_scripting():
                has_pads = torch.tensor(1) if has_pads else torch.tensor(0)
        elif src_tokens is not None:
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
            has_pads = (
            torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
            )
            # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
            if torch.jit.is_scripting():
                has_pads = torch.tensor(1) if has_pads else torch.tensor(0)
        
       
        encoder_embedding=None
        if previous_encode is None:
            x, encoder_embedding = self.forward_embedding(src_tokens)

            # account for padding while computing the representation
        
            x = x * (
                1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
            )
            # print("x shape",x.shape,src_tokens.shape if src_tokens is not None else None,token_embeddings.shape if token_embeddings is not None else None)
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

         
        else:
            x=previous_encode

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
        # src_lengths = (
        #     src_tokens.ne(self.padding_idx)
        #     .sum(dim=1, dtype=torch.int32)
        #     .reshape(-1, 1)
        #     .contiguous()
        # )
        # x.register_hook(lambda grad: print("classifier encoder grad",grad.sum()))
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            # "src_lengths": [src_lengths],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    @torch.jit.export
    def _reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """Dummy re-order function for beamable enc-dec attention"""
        return encoder_out

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
    
class NextStepsForEncoderAttn:
    def __init__(self,instance) -> None:
        self.instance=instance
    def get_for_q_proj(self):
        return self.instance.get_for_encoder_attn_q_proj()
    def get_for_k_proj(self):
        return self.instance.get_for_encoder_attn_k_proj()
    def get_for_v_proj(self):
        return self.instance.get_for_encoder_attn_v_proj()
    def get_for_out_proj(self):
        return self.instance.get_for_encoder_attn_out_proj()
class NextSteps:
    def __init__(self,logits,cfg,encoder_decoder_cfg=None,index=0) -> None:
        # print("next steps",logits.shape if logits is not None else None)
        self.logits=logits
        self.index=index
        self.cfg=cfg
        self.encoder_decoder_cfg=encoder_decoder_cfg
  
    def get_for_encoder_attn(self):
        return NextStepsForEncoderAttn(self)
    # def get_layers(self):
    #     return self.encoder_decoder_cfg.transformer_layers
    def get_for_layer(self,index):
        return NextSteps(self.logits,self.cfg,self.encoder_decoder_cfg,index+self.index,)
    def get_for_encoder(self):
        return NextSteps(self.logits,self.cfg,self.cfg.encoder,self.index)
    def get_for_decoder(self):
        return NextSteps(self.logits,self.cfg,self.cfg.decoder,self.index)
    def get_for_fc1(self):
        if self.logits is None:
            return None
        return self.logits[:,self.index,self.encoder_decoder_cfg.fc1_selection_index]
    def get_for_fc2(self):
        if self.logits is None:
            return None
        return self.logits[:,self.index,self.encoder_decoder_cfg.fc2_selection_index]
    def get_for_q_proj(self):
        if self.logits is None:
            return None
        # print("self.logits.next",self.logits.shape)
        return self.logits[:,self.index,self.encoder_decoder_cfg.self_attn_q_proj_selection_index]
    def get_for_k_proj(self):
        if self.logits is None:
            return None
        return self.logits[:,self.index,self.encoder_decoder_cfg.self_attn_k_proj_selection_index]
    def get_for_v_proj(self):
        if self.logits is None:
            return None
        return self.logits[:,self.index,self.encoder_decoder_cfg.self_attn_v_proj_selection_index]
    def get_for_out_proj(self):
        if self.logits is None:
            return None
        return self.logits[:,self.index,self.encoder_decoder_cfg.self_attn_out_proj_selection_index]
    def get_for_encoder_attn_q_proj(self):
        if self.logits is None:
            return None
        return self.logits[:,self.index,self.encoder_decoder_cfg.encoder_attn_q_proj_selection_index]
    def get_for_encoder_attn_k_proj(self):
        if self.logits is None:
            return None
        return self.logits[:,self.index,self.encoder_decoder_cfg.encoder_attn_k_proj_selection_index]
    def get_for_encoder_attn_v_proj(self):
        if self.logits is None:
            return None
        return self.logits[:,self.index,self.encoder_decoder_cfg.encoder_attn_v_proj_selection_index]
    def get_for_encoder_attn_out_proj(self):
        if self.logits is None:
            return None
        return self.logits[:,self.index,self.encoder_decoder_cfg.encoder_attn_out_proj_selection_index]
        

class TransformerStepsClassifier(torch.nn.Module):
    def __init__(self, cfg, classifier_cfg,dictionary, embed_tokens, return_fc=False):
        super().__init__()
        self.encoder = TransformerStepsClassifierEncoderBase(
            cfg, classifier_cfg,dictionary, embed_tokens, return_fc=return_fc
        )
        self.cfg = cfg
        self.encoder_decoder_layers=cfg.encoder.layers+cfg.decoder.layers
        self.selective_layers=classifier_cfg.selective_layers
        self. total_options= classifier_cfg. total_options
        self.classifier_layer = nn.Linear(cfg.encoder.embed_dim,self.encoder_decoder_layers*self.total_options*self.selective_layers)
        self.classifier_cfg = classifier_cfg
        self.enable=classifier_cfg.enable_classifier
        # print("classifier enable",self.enable)
        
    def output_layer(self,features:Tensor):
        # print("features",features.shape)
        features=features.mean(dim=1)
        logits=self.classifier_layer(features)
        logits=logits.view(-1,self.encoder_decoder_layers,self.selective_layers,self.total_options)
        # logits.register_hook(lambda grad: print("classifier grad",grad.sum()))
        logits=zero_lowest_k(logits,self.classifier_cfg.total_options-self.classifier_cfg.options_each_layer ,dim=-1)
        # logits=torch.zeros_like(logits)
        # logits[:,0]=1
        # logits[:,1:]=0
        # logits=torch.ones_like(logits)
        # logits=logits.softmax(dim=-1)
        # logits=logits.softmax(dim=-1)
        # print("logits",logits)
        # epsilon = 1e-5
        # logits=logits/(logits. sum(dim=-1,keepdim=True)+epsilon)
    
        # logits.register_hook(lambda grad: print("classifier grad2",grad.sum()))
        # print("logits",logits)
        return NextSteps(logits,self.cfg)
    # def set_epoch(self,epoch):
    #     self.enable= epoch>=self.classifier_cfg.classifier_enable_epoch
    #     print("classifier enable",self.enable)
        
    def forward(self,src_tokens:Optional[Tensor]=None,src_lengths:Optional[torch.Tensor]=None,previous_encode:Optional[Dict]=None):
        # print("enable" ,self.enable)
        # batch=src_tokens.shape[0] if src_tokens is not None else previous_encode["encoder_out"][0].shape[1]
        # result= torch.zeros(batch,self.encoder_decoder_layers,self.selective_layers,self.total_options).to(src_tokens.device if src_tokens is not None else previous_encode["encoder_out"][0].device)
        # result=torch.softmax(result,dim=-1)
        # return NextSteps(result,self.cfg)
        if not self.enable:
            return NextSteps(None,self.cfg)
        prev_encoder_out=previous_encode["encoder_out"][0] if previous_encode is not None else None
        encoder_mask=previous_encode["encoder_padding_mask"][0] if previous_encode is not None else None
        out=self.encoder(src_tokens,src_lengths,previous_encode=prev_encoder_out,padding_mask=encoder_mask)
        
        out=out["encoder_out"][0]
        # out.register_hook(lambda grad: print("classifier out",grad.sum()))
        out=out.transpose(0,1)
        return self.output_layer(out)
        