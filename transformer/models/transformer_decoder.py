# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqIncrementalDecoder

from ..nn.selective_linear import SelectiveLinear

from ..nn.selective_transformer_layer import SelectiveTransformerDecoderLayerBase
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase

from ..nn.confidence_loss import confidence_loss

from ..models.transformer_config import TransformerConfig

from ..models.transformer_steps_classifier_encoder import NextSteps,TransformerStepsClassifier
# from fairseq.models.transformer import TransformerConfig
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    #transformer_layer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from ..nn.quant_noise import quant_noise as apply_quant_noise_


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerDecoderBase":
        return "TransformerDecoder"
    else:
        return module_name


class TransformerDecoderBase(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        cfg (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """
    # def set_last_loss(self, loss):
    #     self.next_steps_classifier.set_last_loss(loss)
    # def set_classifier_requires_grad(self,requires_grad):
    #     self.next_steps_classifier_requires_grad=requires_grad
    # def set_epoch(self, epoch):
    #     if self.cfg.decoder.classifier_learn_epoch>=epoch:
    #         self.set_requires_grad(True)
    #     if hasattr(super(),"set_epoch"):
    #         super().set_epoch(epoch)
    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder.layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder.embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                 Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
       
        self.layers.extend(
            [
                self.build_selective_decoder_layer(cfg, no_encoder_attn,index=index)
                # shared_layer
                for index in range(cfg.decoder.selective_layers)
            ]
        )
        # self.layers.extend(
        #     [
        #         self.build_non_selective_decoder_layer(cfg, no_encoder_attn)
        #         for _ in range(cfg.decoder.non_selective_layers)
        #     ]
        # )
        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            torch.nn. Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)
        # self.set_classifier_requires_grad(True)
        self.build_sharing(cfg.decoder.sharing_method)
        # if self.cfg.encoder.enable_classifier:
        #     self.drop_default_index()
    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
           
            self.output_projection = Linear(
                #self.cfg.decoder.num_options,
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = SelectiveLinear(
                self.cfg.decoder.num_options,
                 self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                batch_index=0,
                bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
            )
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )
    # def set_epoch(self, epoch):
    #     if self.cfg.encoder.classifier_learn_epoch>=epoch:
    #         print("decoder drop default index")
    #         self.drop_default_index()
    #     if hasattr(super(),"set_epoch"):
    #         super().set_epoch(epoch)
    def drop_default_index(self):
        if self.cfg.decoder.sharing_method=="all":
            return
        elif self.cfg.decoder.sharing_method=="none":
            for i in range(0,len(self.layers)):
                if self.cfg.decoder.fc1_selection_index is not None:
                    self.layers[i].fc1.fill_with_default_index()
                    self.layers[i].fc1.default_index=None
                if self.cfg.decoder.fc2_selection_index is not None:
                    self.layers[i].fc2.fill_with_default_index()
                    self.layers[i].fc2.default_index=None
                if self.cfg.decoder.self_attn_k_proj_selection_index is not None:
                    self.layers[i].self_attn.k_proj.fill_with_default_index()
                    self.layers[i].self_attn.k_proj.default_index=None
                if self.cfg.decoder.self_attn_v_proj_selection_index is not None:
                    self.layers[i].self_attn.v_proj.fill_with_default_index()
                    self.layers[i].self_attn.v_proj.default_index=None
                if self.cfg.decoder.self_attn_q_proj_selection_index is not None:
                    self.layers[i].self_attn.q_proj.fill_with_default_index()
                    self.layers[i].self_attn.q_proj.default_index=None
                if self.cfg.decoder.self_attn_out_proj_selection_index is not None:
                    self.layers[i].self_attn.out_proj.fill_with_default_index()
                    self.layers[i].self_attn.out_proj.default_index=None
                if self.layers[i].encoder_attn is not None:
                    if self.cfg.decoder.encoder_attn_k_proj_selection_index is not None:
                        self.layers[i].encoder_attn.k_proj.fill_with_default_index()
                        self.layers[i].encoder_attn.k_proj.default_index=None
                    if self.cfg.decoder.encoder_attn_v_proj_selection_index is not None:
                        self.layers[i].encoder_attn.v_proj.fill_with_default_index()
                        self.layers[i].encoder_attn.v_proj.default_index=None
                    if self.cfg.decoder.encoder_attn_q_proj_selection_index is not None:
                        self.layers[i].encoder_attn.q_proj.fill_with_default_index()
                        self.layers[i].encoder_attn.q_proj.default_index=None
                    if self.cfg.decoder.encoder_attn_out_proj_selection_index is not None:
                        self.layers[i].encoder_attn.out_proj.fill_with_default_index()
                        self.layers[i].encoder_attn.out_proj.default_index=None
                
        else:
            raise NotImplementedError("sharing method not implemented")
        
        
            
    def build_sharing(self,method):
        if method=="none":
            for i in range(0,len(self.layers)):
                self.layers[i].fc1.default_index=0
                self.layers[i].fc2.default_index=0
                self.layers[i].self_attn.k_proj.default_index=0
                self.layers[i].self_attn.v_proj.default_index=0
                self.layers[i].self_attn.q_proj.default_index=0
                self.layers[i].self_attn.out_proj.default_index=0
                if self.layers[i].encoder_attn is not None:
                    self.layers[i].encoder_attn.k_proj.default_index=0
                    self.layers[i].encoder_attn.v_proj.default_index=0
                    self.layers[i].encoder_attn.q_proj.default_index=0
                    self.layers[i].encoder_attn.out_proj.default_index=0
                    
        elif method=="all":
            base_layer=self.layers[0]
            for i in range(1,len(self.layers)):
                if self.cfg.decoder.fc1_selection_index is not None:
                    self.layers[i].fc1=base_layer.fc1
                  
                if self.cfg.decoder.fc2_selection_index is not None:
                    self.layers[i].fc2=base_layer.fc2
                   
                if self.cfg.decoder.self_attn_k_proj_selection_index is not None:
                    self.layers[i].self_attn.k_proj=base_layer.self_attn.k_proj
                  
                if self.cfg.decoder.self_attn_v_proj_selection_index is not None:
                    self.layers[i].self_attn.v_proj=base_layer.self_attn.v_proj
                   
                if self.cfg.decoder.self_attn_q_proj_selection_index is not None:
                    self.layers[i].self_attn.q_proj=base_layer.self_attn.q_proj
                    
                if self.cfg.decoder.self_attn_out_proj_selection_index is not None:
                    self.layers[i].self_attn.out_proj=base_layer.self_attn.out_proj
                   
                if base_layer.encoder_attn is not None:
                    if self.cfg.decoder.encoder_attn_k_proj_selection_index is not None:
                        self.layers[i].encoder_attn.k_proj=base_layer.encoder_attn.k_proj
                     
                    if self.cfg.decoder.encoder_attn_v_proj_selection_index is not None:
                        self.layers[i].encoder_attn.v_proj=base_layer.encoder_attn.v_proj
                      
                    if self.cfg.decoder.encoder_attn_q_proj_selection_index is not None:
                        self.layers[i].encoder_attn.q_proj=base_layer.encoder_attn.q_proj
                       
                    if self.cfg.decoder.encoder_attn_out_proj_selection_index is not None:
                        self.layers[i].encoder_attn.out_proj=base_layer.encoder_attn.out_proj
                       
        else:
            
            raise NotImplementedError("sharing method not implemented")
    def build_selective_decoder_layer(self, cfg, no_encoder_attn=False,index=None):
        layer = SelectiveTransformerDecoderLayerBase(cfg, no_encoder_attn,index=index)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    def build_non_selective_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = TransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        *,
        next_steps
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        #print("transformer_decoder.py:forward",prev_output_tokens.shape,index.shape)
        
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            next_steps=next_steps
        )
        #print("forward index",index.shape)
        if not features_only:
            x = self.output_layer(x,next_steps)
        # if self.next_steps_classifier_requires_grad:
        #     @torch.enable_grad()
        #     def backward_steps_classifier(grad):
        #         print("grad",grad.detach().sum(),"confidence",next_steps.get_confidence().mean(),"decoder")
        #         confidence_loss(grad.detach().sum(),next_steps.get_confidence().mean()).backward()
        #         #((grad.detach().sum())*next_steps.get_confidence().mean()).backward()
        #     if torch.is_grad_enabled():
        #         x.register_hook(backward_steps_classifier)
        # x.register_hook(lambda grad:print("decoder",grad.sum()))
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        next_steps=...  
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            next_steps=next_steps
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    
        next_steps=...
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        assert next_steps is not ...
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            # with torch.set_grad_enabled(next_steps.requires_grad_for_layer(idx)):
                # if isinstance(layer,SelectiveTransformerDecoderLayerBase):
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                index=next_steps.get_for_layer(idx)
            )
            
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features,index):
       
        """Project features to the vocabulary size."""
       
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return self.output_projection(features)
         
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict
class TransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
        
    ):
        #self.args = args
        super().__init__(
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )
        self.next_steps_classifier=TransformerStepsClassifier(cfg,cfg.decoder,None,None)
    # @torch.jit.ignore
    def forward( self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False):
        next_steps=self.next_steps_classifier(previous_encode=encoder_out)
        out= super().forward(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            features_only=features_only,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            next_steps= next_steps.get_for_decoder()
        )
        # self.next_steps_classifier().previous_steps=None
        return out
    
    

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


# class TransformerDecoder(TransformerDecoderBase):
#     def __init__(
#         self,
#         cfg,
#         dictionary,
#         embed_tokens,
#         no_encoder_attn=False,
#         output_projection=None,
#     ):
#         #self.args = args
#         super().__init__(
#             cfg,
#             dictionary,
#             embed_tokens,
#             no_encoder_attn=no_encoder_attn,
#             output_projection=output_projection,
#         )
    
#         self.next_steps_classifier=torch.nn.Module()

#     def forward(
#         self,
#         prev_output_tokens,
#         encoder_out: Optional[Dict[str, List[Tensor]]] = None,
#         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
#         features_only: bool = False,
#         full_context_alignment: bool = False,
#         alignment_layer: Optional[int] = None,
#         alignment_heads: Optional[int] = None,
#         src_lengths: Optional[Any] = None,
#         return_all_hiddens: bool = False,
       
#     ):
#         # print("transformer_decoder.py:forward0",prev_output_tokens.shape)
#         # print(prev_output_tokens.shape,"prev_output_tokens decoder")
        
#         token_embeddings=encoder_out["encoder_out"][0]
#         token_embeddings=token_embeddings.transpose(0,1)
#         # print(token_embeddings.shape,"token_embeddings")
#         # print("decoder forward")
#         next_steps,_=self.next_steps_classifier(src_tokens=None,src_lengths=src_lengths,prev_output_tokens=prev_output_tokens,token_embeddings=token_embeddings)
#         # print(next_steps.shape,"decoder next steps")
#         #next_steps=NextSteps(next_steps)
#         # print(prev_output_tokens.shape)
#         # print("input",prev_output_tokens.shape)
#         # print("next_steps",next_steps.get_indices().shape)
#         return super().forward(
#             prev_output_tokens,
#             encoder_out=encoder_out,
#             incremental_state=incremental_state,
#             features_only=features_only,
#             full_context_alignment=full_context_alignment,
#             alignment_layer=alignment_layer,
#             alignment_heads=alignment_heads,
#             src_lengths=src_lengths,
#             return_all_hiddens=return_all_hiddens,
#             next_steps=next_steps
#         )
        
# class TransformerDecoderSection(nn.Module):
#     def __init__(self, args,
#         dictionary,
#         embed_tokens,
#         no_encoder_attn=False,
#         output_projection=None) -> None:
#         self.transformer_encoder=TransformerDecoder(args,dictionary,embed_tokens,no_encoder_attn,output_projection)
#         self.cfg=self.transformer_encoder.cfg
#         #self.forward_classifier_encoder=NextStepsClassifierEncoder(args,dictionary,embed_tokens,no_encoder_attn,output_projection)
      
     