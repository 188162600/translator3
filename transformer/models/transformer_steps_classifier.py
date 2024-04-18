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

from ..models.transformer_steps_classifier_decoder import TransformerStepsClassifierDecoderBase
from ..models.transformer_steps_classifier_encoder import TransformerStepsClassifierEncoderBase
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
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


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
        self.steps=classifier_cfg.layers
        self.selective_layers=classifier_cfg.selective_layers
        self. total_options= classifier_cfg. total_options
        self.classifier_layer = nn.Linear(cfg.encoder.embed_dim,self.total_options*self.selective_layers)
        self.decoder=TransformerStepsClassifierDecoderBase(
            cfg,classifier_cfg,dictionary,embed_tokens
        )
        self.cfg = cfg
       
        
        self.classifier_cfg = classifier_cfg
        self.enable=classifier_cfg.enable_classifier
        # print("classifier enable",self.enable)
        
    def output_layer(self,features:Tensor):
        # print("features",features.shape)
        # features=features.mean(dim=1)
        logits=self.classifier_layer(features)
        # print("logits",logits.shape)
        logits=logits.view(-1,self.steps,self.selective_layers,self.total_options)
        # print("logits",logits.shape)
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
    def decode_steps(self, encode_out):
        # if self.decoder is None:
        #     return encode_out["encoder_out"][0].
        incremental_state = {}
       
        batch = encode_out["encoder_out"][0].shape[1]
        # Initialize `previous_encode` with zeros, of appropriate dimension and device
        previous_tokens = torch.zeros(batch, 1, self.cfg.decoder.embed_dim, device=encode_out["encoder_out"][0].device)
    
        for i in range(self.steps+1):
            # Decode one step at a time, updating `incremental_state`
            logits, _ = self.decoder(
                previous_tokens, 
                encode_out,
                incremental_state=incremental_state,
            )
            next_token = logits[:, -1, :]
            previous_tokens = torch.cat([previous_tokens, next_token], dim=1)
            
            # print(previous_encode[0,:,0])
            
          
        
        return previous_tokens

    
    def adjust_encoder_out_seq_length(self,out):
        pass
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
        encode_out=self.encoder(src_tokens,src_lengths,previous_encode=prev_encoder_out,padding_mask=encoder_mask)
       
        decode_out=self.decode_steps(encode_out)
        return self.output_layer(decode_out)
        