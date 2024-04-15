# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import logging

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from ..models.transformer_config import TransformerConfig
from ..models.transformer_decoder import TransformerDecoder
from ..models.transformer_encoder import TransformerEncoder
from ..nn.logit_gambler import LogitGambler
# from ..models.transformer_steps_classifier import TransformerStepsClassifier,NextSteps

logger = logging.getLogger(__name__)
import weakref
from fairseq.models import BaseFairseqModel
# print("is instance",issubclass(BaseFairseqModel, FairseqEncoderDecoderModel))
class TransformerModelBase(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True
        # self. try_drop_default_index()
    def try_drop_default_index(self):
        if self.cfg.decoder.sharing_method=="all":
            return
        elif self.cfg.decoder.sharing_method=="none":
            if self.cfg.encoder.enable_classifier:
                
                for i in range(0,self.cfg.encoder.classifier_encoder_layers):
                    self.encoder.next_steps_classifier.encoder.layers[i].load_state_dict(self.encoder.layers[i].state_dict())
                   
                    
            if self.cfg.decoder.enable_classifier:
                start=max(0,self.cfg.encoder.classifier_encoder_layers-len(self.encoder.layers))
                for i in range(start,self.cfg.decoder.classifier_encoder_layers):
                    
                    self.encoder.next_steps_classifier.encoder.layers[i-start].load_state_dict(self.encoder.layers[i].state_dict())
                   
                # copy_from=self.cfg.encoder.classifier_encoder_layers[i]
                # copy_to=self.cfg.decoder.classifier_decoder_layers[i]
                # self.encoder.next_steps_classifier.layers[i]=self.encoder.layers[i]
                
                # if self.cfg.encoder.fc1_selection_index is not None:
                #     self.layers[i].fc1.fill_with_default_index()
                #     self.layers[i].default_index=None
                # if self.cfg.encoder.fc2_selection_index is not None:
                #     self.layers[i].fc2.fill_with_default_index()
                #     self.layers[i].default_index=None
                # if self.cfg.encoder.self_attn_k_proj_selection_index is not None:
                #     self.layers[i].self_attn.k_proj.fill_with_default_index()
                #     self.layers[i].default_index=None
                # if self.cfg.encoder.self_attn_v_proj_selection_index is not None:
                #     self.layers[i].self_attn.v_proj.fill_with_default_index()
                #     self. layers[i].default_index=None
                # if self.cfg.encoder.self_attn_q_proj_selection_index is not None:
                #     self.layers[i].self_attn.q_proj.fill_with_default_index()
                #     self.layers[i].default_index=None
                # if self.cfg.encoder.self_attn_out_proj_selection_index is not None:
                #     self.layers[i].self_attn.out_proj.fill_with_default_index()
                #     self. layers[i].default_index=None
        # self.set_epoch(1)
        # if next_steps_classifier is None:
            # next_steps_classifier=TransformerStepsClassifier(cfg,encoder,decoder)
        # self.next_steps_classifier=next_steps_classifier
        # encoder.next_steps_classifier=weakref.ref(next_steps_classifier)
        # decoder.next_steps_classifier=weakref.ref(next_steps_classifier)
        # self.next_steps_classifier=GamblerNextStepsClassifier(cfg)
        # encoder.next_steps_classifier=TransformerEncoderStepsClassifier(cfg,encoder.dictionary,encoder.embed_tokens,decoder.dictionary,decoder.embed_tokens)
        # decoder.next_steps_classifier=TransformerDecoderStepsClassifier(cfg,encoder.dictionary,encoder.embed_tokens,decoder.dictionary,decoder.embed_tokens)
    # def set_last_loss(self, loss):
    #     #loss=loss.detach()
    #     #  def set_last_loss(self, loss):
        
        
      
    #     if hasattr(self.next_steps_classifier,"set_last_loss"):
    #         self.next_steps_classifier.set_last_loss(loss)
    # def set_epoch(self, epoch):
    #     for m in self.modules():
    #         if hasattr(m, "set_epoch") and m != self:
    #             m.set_epoch(epoch)
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoder(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        
        # next_steps=self.next_steps_classifier(src_tokens, src_lengths, prev_output_tokens)
      
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        # print("encoder_out",encoder_out["encoder_out"][0].shape)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

   