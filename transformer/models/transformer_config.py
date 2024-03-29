# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re
from dataclasses import dataclass, field, fields
from typing import List, Optional

from omegaconf import II

from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.utils import safe_getattr, safe_hasattr
#from fairseq.models.transformer.transformer_config import DecoderConfig
DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

_NAME_PARSER = r"(decoder|encoder|quant_noise)_(.*)"
def _get_total_options(sharing_method:str,num_total,num_shared,num_non_shared,num_steps)->int:
    print("sharing method",sharing_method)
    if sharing_method=="none":
        return int(num_steps*num_total)
    elif sharing_method=="random":
        return int(num_non_shared*num_steps)
    elif sharing_method=="cycle_rev":
        return int(num_non_shared*num_steps/2+num_total*num_steps/2)
    
from fairseq.dataclass import FairseqDataclass
@dataclass
class EncDecBaseConfig(FairseqDataclass):
    
    embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained embedding"}
    )
    embed_dim: Optional[int] = field(
        default=512, metadata={"help": "embedding dimension"}
    )
    ffn_embed_dim: int = field(
        default=2048, metadata={"help": "embedding dimension for FFN"}
    )
    layers: int = field(default=6, metadata={"help": "number of layers"})
    attention_heads: int = field(
        default=8, metadata={"help": "number of attention heads"}
    )
    normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each block"}
    )
    learned_pos: bool = field(
        default=False, metadata={"help": "use learned positional embeddings"}
    )
    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    layerdrop: float = field(default=0, metadata={"help": "LayerDrop probability"})
    layers_to_keep: Optional[List[int]] = field(
        default=None, metadata={"help": "which layers to *keep* when pruning"}
    )

    xformers_att_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "config for xFormers attention, defined in xformers.components.attention.AttentionConfig"
        },
    )
    # def __setattr__(self, __name: str, __value: re.Any) -> None:
    #     if isinstance(__value,)
    #     return super().__setattr__(__name, __value)
@dataclass
class SelectiveEncDecBaseConfig(EncDecBaseConfig):
    sharing_method: str = field(
        default="cycle_rev", metadata={"help": "sharing method"}
    )
    classifier_learn_epoch:int = field(
        default=5,metadata={"help":"number of epochs to train the classifier"}
    )
    options_each_layer:int = field(
        default=4,metadata={"help":"number of options"}
    )
    shared_options_each_layer:int = field(
        default=2,metadata={"help":"number of shared options in each layer"}
    )
    
    
    
    total_options:int=None
    steps_classifier_shared_classes:int = field(
        default=II("model.encoder.shared_options_each_layer"),
    )
    steps_classifier_non_shared_classes:int = field(
        default=II("model.encoder.options_each_layer-model.encoder.shared_options_each_layer"),
    )
    
    steps_classifier_classes:int = field(
        default=II("model.encoder.options_each_layer"),
        metadata={"help":"number of classes in the classifier"}
    )
    num_steps:int = field(
        default=II("model.encoder.layers"),
        metadata={"help":"number of steps"}
    )
    def __post_init__(self):
        #  II doesn't work if we are just creating the object outside of hydra so fix that
        #super().__post_init__()
        if self.num_steps == II("model.encoder.layers"):
            self.num_steps = self.layers
        if self.steps_classifier_classes == II("model.encoder.options_each_layer"):
            self.steps_classifier_classes = self.options_each_layer
        if self.steps_classifier_shared_classes == II("model.encoder.shared_options_each_layer"):
            self.steps_classifier_shared_classes = self.shared_options_each_layer
        if self.steps_classifier_non_shared_classes == II("model.encoder.options_each_layer-model.encoder.shared_options_each_layer"):
            self.steps_classifier_non_shared_classes = self.options_each_layer-self.shared_options_each_layer
        #print(self.sharing_method,self.steps_classifier_classes,self.shared_options_each_layer,self.steps_classifier_non_shared_classes,self.num_steps)
        #if self.total_options is None:
        self.total_options=_get_total_options(self.sharing_method,self.steps_classifier_classes,self.shared_options_each_layer,self.steps_classifier_non_shared_classes,self.num_steps)
        print(self.total_options,"total options")
    
    # num_encoder_layers:int = field(
    #     default=1,metadata={"help":"number of encoder layers"}
    # )
    

# @dataclass
# class EncoderStepsClassifierConfig(EncDecBaseConfig):
#     layers:int = field(
#         default=2,metadata={"help":"number of decoder layers"}
#     )
#     steps_classifier_classes:int=II("model.encoder.num_options")
#     num_steps:int=II("model.encoder.layers")

# @dataclass
# class DecoderStepsClassifierConfig(EncDecBaseConfig):
#     layers:int = field(
#         default=4,metadata={"help":"number of decoder layers"}
#     )
#     steps_classifier_classes:int=II("model.decoder.num_options")
#     num_steps:int=field(
#         default=II("model.decoder.layers+2"),
#         metadata={"help":"number of decoder layers+embedding and output layers"}
#     )
#     def __post_init__(self):
#         #  II doesn't work if we are just creating the object outside of hydra so fix that
#         if self.num_steps == II("model.decoder.layers"):
#             self.num_steps = 
#     #=II("model.decoder.layers")+2 #input and output embeddings
    
@dataclass
class SelectiveDecoderConfig(SelectiveEncDecBaseConfig):
    num_decoder_layers:int = field(
        default=4,metadata={"help":"number of decoder layers"}
    )
    input_dim: int = II("model.decoder.embed_dim")
    output_dim: int = field(
        default=II("model.decoder.embed_dim"),
        metadata={
            "help": "decoder output dimension (extra linear layer if different from decoder embed dim)"
        },
    )
   
    num_steps:int = field(
        default=II("model.decoder.layers+2"),
        metadata={"help":"layers+embedding and output layers"}
    )
    total_options:int=None
    steps_classifier_shared_classes:int = field(
        default=II("model.decoder.shared_options_each_layer"),
    )
    steps_classifier_non_shared_classes:int = field(
        default=II("model.decoder.options_each_layer-model.decoder.shared_options_each_layer"),
    )
    
    steps_classifier_classes:int = field(
        default=II("model.decoder.options_each_layer"),
        metadata={"help":"number of classes in the classifier"}
    )
   
    
    def __post_init__(self):
        
        #  II doesn't work if we are just creating the object outside of hydra so fix that
        if self.input_dim == II("model.decoder.embed_dim"):
            self.input_dim = self.embed_dim
        if self.output_dim == II("model.decoder.embed_dim"):
            self.output_dim = self.embed_dim
        if self.num_steps == II("model.decoder.layers+2"):
            self.num_steps = self.layers+2
            

        if self.steps_classifier_classes == II("model.decoder.options_each_layer"):
            self.steps_classifier_classes = self.options_each_layer
        if self.steps_classifier_shared_classes == II("model.decoder.shared_options_each_layer"):
            self.steps_classifier_shared_classes = self.shared_options_each_layer
        if self.steps_classifier_non_shared_classes == II("model.decoder.options_each_layer-model.decoder.shared_options_each_layer"):
            self.steps_classifier_non_shared_classes = self.options_each_layer-self.shared_options_each_layer
        super().__post_init__()
        # if self.total_options is None:
        #     self.total_options=_get_total_options(self.sharing_method,self.steps_classifier_classes,self.shared_options_each_layer,self.steps_classifier_non_shared_classes,self.num_steps)
        
        


@dataclass
class QuantNoiseConfig(FairseqDataclass):
    pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )


@dataclass
class TransformerConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu",
        metadata={"help": "activation function to use"},
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN.",
            "alias": "--relu-dropout",
        },
    )
    adaptive_input: bool = False
    #encoder_steps_classifier:EncoderStepsClassifierConfig=EncoderStepsClassifierConfig()
    encoder: SelectiveEncDecBaseConfig =field(default_factory=SelectiveEncDecBaseConfig)# None#SelectiveEncDecBaseConfig()
    # TODO should really be in the encoder config
    max_source_positions: int = field(
        default=DEFAULT_MAX_SOURCE_POSITIONS,
        metadata={"help": "Maximum input length supported by the encoder"},
    )
   # decoder_steps_classifier:DecoderStepsClassifierConfig=DecoderStepsClassifierConfig()
    decoder: SelectiveDecoderConfig = field(default_factory=SelectiveDecoderConfig) #None#SelectiveDecoderConfig()
    # TODO should really be in the decoder config
    max_target_positions: int = field(
        default=DEFAULT_MAX_TARGET_POSITIONS,
        metadata={"help": "Maximum output length supported by the decoder"},
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    share_all_embeddings: bool = field(
        default=False,
        metadata={
            "help": "share encoder, decoder and output embeddings (requires shared dictionary and embed dim)"
        },
    )
    merge_src_tgt_embed: bool = field(
        default=False,
        metadata={
            "help": "if true then the source and target embedding table is "
            "merged into one table. This is going to make the model smaller but "
            "it might hurt performance."
        },
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if True, disables positional embeddings (outside self attention)"
        },
    )
    adaptive_softmax_cutoff: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "list of adaptive softmax cutoff points. Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0.0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, which saves GPU memory usage at the cost of some additional compute"
        },
    )
    offload_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations."
        },
    )
    # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
    no_cross_attention: bool = field(
        default=False, metadata={"help": "do not perform cross-attention"}
    )
    cross_self_attention: bool = field(
        default=False, metadata={"help": "perform cross+self-attention"}
    )
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise: QuantNoiseConfig =field(default_factory=QuantNoiseConfig) #field(default=QuantNoiseConfig())
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )
    # DEPRECATED field, but some old checkpoints might have it
    char_inputs: bool = field(
        default=False, metadata={"help": "if set, model takes character ids as input"}
    )
    relu_dropout: float = 0.0
    # config for "BASE Layers: Simplifying Training of Large, Sparse Models"
    base_layers: Optional[int] = field(
        default=0, metadata={"help": "number of BASE layers in total"}
    )
    base_sublayers: Optional[int] = field(
        default=1, metadata={"help": "number of sublayers in each BASE layer"}
    )
    base_shuffle: Optional[int] = field(
        default=1,
        metadata={"help": "shuffle tokens between workers before computing assignment"},
    )

    export: bool = field(
        default=False,
        metadata={"help": "make the layernorm exportable with torchscript."},
    )

    # copied from transformer_lm but expected in transformer_decoder:
    no_decoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"},
    )

    # We need to make this hierarchical dataclass like the flat namespace
    # __getattr__ and __setattr__ here allow backward compatibility
    # for subclasses of Transformer(Legacy) that depend on read/write on
    # the flat namespace.

    def __getattr__(self, name):
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            return safe_getattr(sub, match[2])
        raise AttributeError(f"invalid argument {name}.")

    def __setattr__(self, name, value):
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            setattr(sub, match[2], value)
        else:
            super().__setattr__(name, value)

    @staticmethod
    def _copy_keys(args, cls, prefix, seen):
        """
        copy the prefixed keys (decoder_embed_dim) to the DC fields: decoder.embed_dim
        """
        cfg = cls()
        for fld in fields(cls):
            # for all the fields in the DC, find the fields (e.g. embed_dim)
            # in the namespace with the prefix (e.g. decoder)
            # and set it on the dc.
            args_key = f"{prefix}_{fld.name}"
            if safe_hasattr(args, args_key):
                seen.add(args_key)
                setattr(cfg, fld.name, safe_getattr(args, args_key))
            if safe_hasattr(args, fld.name):
                seen.add(fld.name)
                setattr(cfg, fld.name, safe_getattr(args, fld.name))
        return cfg

    @classmethod
    def from_namespace(cls, args):
        
        if args is None:
            return None
        #print("args",args)
        if not isinstance(args, cls):
            seen = set()
            config = cls()
            # currently, we can go generically from DC fields to args hierarchically
            # but we can't easily deconstruct a flat namespace to a hierarchical
            # DC. Mostly because we could have a sub-dc called `decoder-foo` that should not
            # go to the sub struct called `decoder`. There are ways to go around this, but let's keep it simple
            # for now.
            for fld in fields(cls):
                # concretelly, the transformer_config know what sub-dc it has, so we go through all the dc fields
                # and if it's one that has a sub-dc, we build that sub-dc with `copy_keys()`
                if fld.name == "decoder":
                    if safe_hasattr(args, "decoder"):
                        #  in some cases, the args we receive is already structured (as DictConfigs), so let's just build the correct DC
                        seen.add("decoder")
                        config.decoder = SelectiveDecoderConfig(**args.decoder)
                    else:
                        config.decoder = cls._copy_keys(
                            args, SelectiveDecoderConfig, "decoder", seen
                        )
                elif fld.name == "encoder":
                    # same but for encoder
                    if safe_hasattr(args, "encoder"):
                        seen.add("encoder")
                        config.encoder = SelectiveEncDecBaseConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(
                            args, SelectiveEncDecBaseConfig, "encoder", seen
                        )
                # elif fld.name=="encoder_steps_classifier":
                #     if safe_hasattr(args, "encoder_steps_classifier"):
                #         seen.add("encoder_steps_classifier")
                #         config.encoder_steps_classifier = EncoderStepsClassifierConfig(**args.encoder_steps_classifier)
                #     else:
                #         config.encoder_steps_classifier = cls._copy_keys(
                #             args, EncoderStepsClassifierConfig, "encoder_steps_classifier", seen
                #         )
                # elif fld.name == "decoder_steps_classifier":
                #     if safe_hasattr(args, "decoder_steps_classifier"):
                #         seen.add("decoder_steps_classifier")
                #         config.decoder_steps_classifier = DecoderStepsClassifierConfig(**args.decoder_steps_classifier)
                #         #config.decoder_steps_classifier.num_steps=config.decoder_steps_classifier.layers+2
                #     else:
                #         config.decoder_steps_classifier = cls._copy_keys(
                #             args, DecoderStepsClassifierConfig, "decoder_steps_classifier", seen
                #         )
                        
                elif fld.name == "quant_noise":
                    # same but for quant_noise
                    if safe_hasattr(args, "quant_noise"):
                        seen.add("quant_noise")
                        config.quant_noise = QuantNoiseConfig(**args.quant_noise)
                    else:
                        config.quant_noise = cls._copy_keys(
                            args, QuantNoiseConfig, "quant_noise", seen
                        )
                
                elif safe_hasattr(args, fld.name):
                    # if it's not a structure field, it's just a normal field, copy it over
                    seen.add(fld.name)
                    setattr(config, fld.name, safe_getattr(args, fld.name))
            # we got all the fields defined in the dataclass, but
            # the argparse namespace might have extra args for two reasons:
            #   - we are in a legacy class so all the args are not declared in the dataclass. Ideally once everyone has defined a dataclass for their model, we won't need this
            #   - some places expect args to be there but never define them
            args_dict = (
                args._asdict()
                if safe_hasattr(args, "_asdict")
                else vars(args)
                if safe_hasattr(args, "__dict__")
                else {}
            )  # namedtupled doesn't have __dict__ :-/
            for key, value in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)
            return config
        else:
            return args
