from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion,LabelSmoothedCrossEntropyCriterionConfig
from fairseq.criterions import register_criterion
from fairseq.criterions import FairseqCriterion, register_criterion

@register_criterion(
    "last_loss_label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)


class LabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def forward(self, model, sample, reduce=True):
        loss, sample_size, logging_output= super().forward(model, sample, reduce)
        # print(loss/sample_size)
        model.set_last_loss(loss.detach()/sample_size)
        return loss, sample_size, logging_output
        
        
        