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
class PositionalMaxCrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    size:int=II("dataset.max_tokens")
    report_original_loss: bool = field(default=True, metadata={"help": "Report the original loss"})
    init_decay_rate: float = field(default=0.9, metadata={"help": "Initial decay rate for the loss weights"})
    init_min_loss: float = field(default=0.5, metadata={"help": "Initial minimum loss value for the weights"})
    min_loss:float=field(default=0.1, metadata={"help": "Minimum loss value for the weights"})
    max_loss:float=field(default=10, metadata={"help": "Maximum loss value for the weights"})
    first_token_max_loss:float=field(default=0.3, metadata={"help": "The model always overfits the first token as the previous token is fixed, so we can set a lower max loss for it"})
@register_criterion("positional_max_cross_entropy", dataclass=PositionalMaxCrossEntropyCriterionConfig)
class PositionalMaxCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, size,report_original_loss,init_decay_rate,init_min_loss,min_loss,max_loss,first_token_max_loss):
        super().__init__(task)
        self.sentence_avg = sentence_avg
     
        self.weights=torch.nn.Parameter( self.build_weights(size,init_decay_rate,init_min_loss,first_token_max_loss),requires_grad=True)
        # self._init_weights(size,init_decay_rate,init_min_loss)
        self.report_original_loss=report_original_loss
        self.min_loss=min_loss
        self.max_loss=max_loss
        for param in self.parameters():
            param.register_hook(self.maximize_loss)
        # self.register_backward_hook(self.maximize_loss)
        # self.to(task.device)
  
    def build_weights(self, size, decay_rate, min_loss,first_token_max_loss):
        result= torch.tensor([1. for i in range(size)])
        
        # result[0]=min(first_token_max_loss,result[0])
        return result
    @staticmethod
    def maximize_loss( grad):
        return -grad
    def get_weights(self,length,device):
       
        self.weights.data=torch.clamp(self.weights.data,min=self.min_loss,max=self.max_loss)
        #print(self.weights[0:10])
        weight=self.weights[:length]
        return (weight.mean(dim=0)/weight)
    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        # print(lprobs.shape)
        
        weight=self.get_weights(lprobs.size(1), lprobs.device)
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
            "weight":self.weights[0:20]
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
            metrics.log_scalar("weight",logging_outputs[0]["weight"])
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True