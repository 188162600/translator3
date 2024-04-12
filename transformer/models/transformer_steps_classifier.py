from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import weakref
import logging

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from ..models.transformer_config import TransformerConfig
from ..models.transformer_decoder import TransformerDecoder
from ..models.transformer_encoder import TransformerEncoder
from ..nn.logit_gambler import LogitGambler
from ..models.next_steps import NextStepsForEncoderAttn,NextSteps
from ..models.next_steps import GamblerNextSteps
# from ..models.transformer_steps_classifier import TransformerStepsClassifier,NextSteps

 
class GamblerNextStepsClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # print("__ini")
        self.cfg = cfg
        self.logit_gambler=LogitGambler((cfg.encoder.classifier_layers+cfg.encoder.classifier_layers,cfg.selective_layers ,cfg.options_each_layer,))
       
        # print("GamblerNextStepsClassifier2",self.logit_gambler())
        
    def forward(self,src_tokens,
        src_lengths,
        
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,):
        # print(torch.is_grad_enabled(),"torch.is_grad_enabled()")
        # self.logit_gambler.train()
        # print("vvv",self.logit_gambler())
        return self.logit_gambler()
        # return GamblerNextSteps(self.logit_gambler(),self.cfg)
    
    
class TransformerStepsClassifier(FairseqEncoderDecoderModel):
    def __init__(self, cfg, encoder,decoder):
        # print("TransformerStepsClassifier init")
      
        super().__init__( encoder, decoder)
        self.next_steps_classifier=GamblerNextStepsClassifier(cfg)
        self.cfg=cfg
        # print("TransformerStepsClassifier done",)
        # self.next_steps_classifier=
        self.selective_layers=cfg.selective_layers
        self.total_layers=cfg.encoder.layers+cfg.decoder.layers
        self.num_options=cfg.options_each_layer
        self.output_projection=nn.Linear(cfg.decoder.embed_dim, self.total_layers*self.selective_layers*self.num_options,bias=False)
    def output_layer(self,features):
        features=features[:,0,:]
        # print("features",features.shape)
        out=self.output_projection(features)
        
        
        batch_size=out.size(0)
        out=out.view(batch_size, self.total_layers,self.selective_layers,self.num_options)
        out=torch.softmax(out,dim=-1)
        return out
        # return NextSteps( out,self.cfg)
    def set_last_loss(self, loss):
        #loss=loss.detach()
        #  def set_last_loss(self, loss):
        
        
        if torch.is_grad_enabled():
            (torch.sum(self.previous_steps)*loss.detach() ).backward()
            self.previous_steps=None
       
    def forward(
        self,
        src_tokens,
        src_lengths,
       
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        
        next_steps=self.next_steps_classifier(src_tokens, src_lengths)
        self.previous_steps=next_steps
        
        # print("ff logits",next_steps.logit)
        # print(next_steps.logit,"next_steps.logit.shape")
        # print(next_steps.get_for_encoder().encoder_decoder_cfg is None,"next_steps.get_for_encoder().encoder_decoder_cfg")
        # with torch.no_grad():
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,next_steps=GamblerNextSteps( next_steps,self.cfg)
        )
        # print("encoder_out",encoder_out["encoder_out"][0].shape)
        # decoder_out,_ = self.decoder(
        #     prev_output_tokens,
        #     encoder_out=encoder_out,
        #     features_only=True,
        #     alignment_layer=alignment_layer,
        #     alignment_heads=alignment_heads,
        #     src_lengths=src_lengths,
        #     return_all_hiddens=return_all_hiddens,
          
        # )
        # print("decoder_out",decoder_out.shape)
        out=encoder_out["encoder_out"][0]
        out=out.transpose(0,1)
        if not features_only:
            decoder_out=self.output_layer(out)
        return decoder_out