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
class PositionalCrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    size:int=II("dataset.max_tokens")
    report_original_loss: bool = field(default=True, metadata={"help": "Report the original loss"})
    init_decay_rate: float = field(default=0.9, metadata={"help": "Initial decay rate for the loss weights"})
    init_min_loss: float = field(default=0.6, metadata={"help": "Initial minimum loss value for the weights"})
    min_loss:float=field(default=0.5, metadata={"help": "Minimum loss value for the weights"})
 
    max_loss:float=field(default=1.25, metadata={"help": "Maximum loss value for the weights"})
    
@register_criterion("positional_cross_entropy", dataclass=PositionalCrossEntropyCriterionConfig)
class PositionalCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, size,report_original_loss,init_decay_rate,init_min_loss,min_loss,max_loss):
        super().__init__(task)
        self.sentence_avg = sentence_avg
     
        self.weights=torch.nn.Parameter( torch.ones(size),requires_grad=True)
        self.sign=torch.nn.Parameter(-torch.ones(self.weights.size(0),dtype=torch.int8),requires_grad=False)
        # self._init_weights(size,init_decay_rate,init_min_loss)
        self.report_original_loss=report_original_loss
        self.min_loss=min_loss
        self.max_loss=max_loss
        
        def grad_hook(grad):
            # print(grad)
            return grad*self.sign
        self.weights.register_hook(grad_hook)

        # self.register_backward_hook(self.maximize_loss)
        # self.to(task.device)
  
    def build_weights(self, size, decay_rate, min_loss):
        result= torch.tensor([max(decay_rate**i,min_loss) for i in range(size)])
        # result[0]=0.3
        # result[0]=min(first_token_max_loss,result[0])
        return result
    @staticmethod
    def maximize_loss( grad):
        return -grad
    def get_weights(self, length, device):
        # Define conditions for each stage based on weights
        stage_max_loss = (self.weights < self.min_loss)
        stage_min_loss = (self.weights > self.max_loss)

        # # Detect changes needed for transition between stages
        # change_to_stage_min = (self.sign[stage_min_loss] != 1)
        # change_to_stage_max = (self.sign[stage_max_loss] != -1)
    
        # Update signs according to stage changes
        if stage_max_loss.any():
            self.weights.data[stage_max_loss]=1.
            self.sign[stage_max_loss] = 1
        if stage_min_loss.any():
            self.weights.data[stage_min_loss]=1.
            self.sign[stage_min_loss] = 1

        # Select weights according to the specified length and ensure they are on the correct device
        weight = self.weights[:length].to(device)

        # Calculate the mean of weights and normalize
        mean_weight = weight.mean()
        normalized_weights =weight/ mean_weight 
        return normalized_weights

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