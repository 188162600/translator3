import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from dataclasses import dataclass, field
from omegaconf import II
from fairseq import metrics, utils
import math
@dataclass
class CrossEntropyWithDecayCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    decay_rate: float =field(default=0.9, metadata={"help": "Decay rate for the loss weights"})
    min_loss: float = field(default=0.3, metadata={"help": "Minimum loss value for the weights"})
    report_original_loss: bool = field(default=True, metadata={"help": "Report the original loss"})
    
@register_criterion("cross_entropy_with_decay", dataclass=CrossEntropyWithDecayCriterionConfig)
class CrossEntropyCriterionWithDecay(FairseqCriterion):
    def __init__(self, task, sentence_avg, decay_rate, min_loss,report_original_loss):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.decay_rate = decay_rate
        self.min_loss = min_loss
        self.weights=self.define_loss_weights(128)
        self.report_original_loss=report_original_loss
    def define_loss_weights(self, seq_len):
        weights= torch.tensor([max(self.decay_rate ** i, self.min_loss) for i in range(seq_len)])
       
        return weights
    def get_loss_weights(self, seq_len, device):
        
        if self.weights.size(0) < seq_len:
            self.weights = self.define_loss_weights(seq_len)
        if self.weights.device != device:
            # if self.weights.device.type=='cpu':
            self.weights = self.weights.to(device)
            
        weight= self.weights[:seq_len].to(device)
        # print("weight",weight)
        weight=weight/weight.mean(dim=0)
        # print("weight",weight)
        return weight
    def compute_loss(self, model, net_output, sample, reduce=True):
        """
        Compute the weighted loss with a greater penalty for early mistakes and enforce a minimum loss threshold.
        """
        lprobs = model.get_normalized_probs(net_output, log_probs=True)

        
        weight=self.get_loss_weights(lprobs.size(1), lprobs.device)
        weighted_lprobs=torch.einsum('ijk,j->ijk',lprobs,weight)
        weighted_lprobs = weighted_lprobs.view(-1, weighted_lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

    
        loss = F.nll_loss(
            weighted_lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        orig_loss=0
        if self.report_original_loss:
            lprobs = lprobs.view(-1, lprobs.size(-1))
            orig_loss = F.nll_loss(
                lprobs.detach(),
                target,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )
            
            
 
        return loss, orig_loss
   
    

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, original_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "orig_loss":original_loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            
        }
        return loss, sample_size, logging_output
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        orig_loss_sum = sum(log.get("orig_loss", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "orig_loss", orig_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True