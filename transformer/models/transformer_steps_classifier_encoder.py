from ..models.transformer_encoder import TransformerEncoderBase
from ..models.next_steps import GamblerNextSteps
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

class TransformerStepsClassifierEncoder(TransformerEncoderBase):
    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
    #self.args = args
        super().__init__(
            cfg,
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )
        self.next_steps_classifier =None
    # @torch.jit.ignore
    def forward(
        self,
        # prev_output_tokens,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        next_steps=None
    ):
        # print("encoder forward")
        # print(self.next_steps_classifier.__class__,"next steps classifier")
        if next_steps is None:
            next_steps=self.next_steps_classifier()(src_tokens,src_lengths)
            next_steps=GamblerNextSteps(next_steps,self.cfg).get_for_encoder()
        # next_steps=self.next_steps_classifier()(src_tokens,src_lengths,prev_output_tokens)
        # print("encoder next steps",next_steps.shape)
        #next_steps=NextSteps(next_steps)
        # print("en input",src_tokens.shape)
        # print("en next_steps",next_steps.get_indices().shape)
        # print((super().forward(
        #     src_tokens=src_tokens,
        #     src_lengths=src_lengths,
        #     return_all_hiddens=return_all_hiddens,
        #     token_embeddings=token_embeddings,
        #     next_steps=next_steps.get_for_encoder())|{"next_steps":next_steps}).keys(),"keys")
        return super().forward(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=token_embeddings,
            next_steps=next_steps.get_for_encoder())