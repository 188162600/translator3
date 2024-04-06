# import math
# from dataclasses import dataclass, field

# import torch
# from fairseq import metrics, utils
# from fairseq.criterions import FairseqCriterion, register_criterion
# from fairseq.dataclass import FairseqDataclass
# from omegaconf import II
# from torch import nn

# class KLDivLossConfig(FairseqDataclass):
#     sentence_avg: bool = field(default=True, metadata={"help": "if true, average over sentence rather than batch"})
#     report_accuracy: bool = field(default=True, metadata={"help": "report accuracy metric"})
    
# @register_criterion('kldiv_loss', dataclass=KLDivLossConfig)
# class KLDivLoss(FairseqCriterion):
#     def __init__(self,  task,sentence_avg=True,report_accuracy=True):
#         super().__init__(task)
#         self.kldiv_loss = nn.KLDivLoss(reduction='batchmean' if sentence_avg else 'sum')
#         self.sentence_avg = sentence_avg
#         self.report_accuracy = report_accuracy

#     def forward(self, model, sample, reduce=True):
#         """Compute the loss for the given sample."""
        
#         sample_size = (
#             sample["target"].size(0) if self.sentence_avg else sample["target"][0].size(0)*len(sample["target"])
#         )
#         net_output = model(**sample["net_input"])
#         print(net_output[0].shape,sample["target"].shape)
#         logits = net_output[0]
#         target_probs = self.task.target_distribution(sample)  # Ensure your task provides this method
#         log_probs = torch.log_softmax(logits, dim=-1)
#         loss = self.kldiv_loss(log_probs, target_probs)
   
#         logging_output = {
#             "loss": loss.data,
#             "ntokens": sample["ntokens"],
#             "nsentences": sample["target"].size(0),
#             "sample_size": sample_size,
#         }
#         if self.report_accuracy:
#             n_correct, total = self.compute_accuracy(model, net_output, sample)
#             logging_output["n_correct"] = utils.item(n_correct.data)
#             logging_output["total"] = utils.item(total.data)
#         return loss, sample_size, logging_output


#         # net_output = model(**sample['net_input'])
#         # logits = net_output[0]
#         # target_probs = self.task.target_distribution(sample)  # Ensure your task provides this method
#         # log_probs = torch.log_softmax(logits, dim=-1)
#         # loss = self.kldiv_loss(log_probs, target_probs)

#         # sample_size = sample['target'].size(0) if self.cfg.sentence_avg else sample['ntokens']
#         # return loss, sample_size