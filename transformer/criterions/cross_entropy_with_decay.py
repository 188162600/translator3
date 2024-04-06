import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from dataclasses import dataclass, field
from omegaconf import II

@dataclass
class CrossEntropyWithDecayCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    decay_rate: float =field(default=0.9, metadata={"help": "Decay rate for the loss weights"})
    min_loss: float = field(default=0.3, metadata={"help": "Minimum loss value for the weights"})
    report_accuracy: bool = field(default=False, metadata={"help": "report accuracy metric"})
    
@register_criterion("cross_entropy_with_decay", dataclass=CrossEntropyWithDecayCriterionConfig)
class CrossEntropyCriterionWithDecay(FairseqCriterion):
    def __init__(self, task, sentence_avg, decay_rate, min_loss):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.decay_rate = decay_rate
        self.min_loss = min_loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        """
        Compute the weighted loss with a greater penalty for early mistakes and enforce a minimum loss threshold.
        """
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        batch_size, seq_len = sample['target'].shape

        # Create weights that decay over the sequence length, with a minimum value applied
        weights = torch.tensor([max(self.decay_rate ** i, self.min_loss) for i in range(seq_len)]).repeat(batch_size, 1).view(-1)
        
        weights = weights.to(lprobs.device)

        # Compute weighted loss
        loss = F.nll_loss(
            lprobs,
            target,
            weight=weights,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

        # Normalize the loss to keep the expected loss comparable
        normalization_factor = weights.sum()
        adjusted_loss = loss / normalization_factor if reduce else loss

        return adjusted_loss, loss
    

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output
