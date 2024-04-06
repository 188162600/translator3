import sys
import os


from transformer.models.transformer_legacy import *
from transformer.criterions.label_smoothed_cross_entropy import *
from transformer.criterions.kldiv_loss import *
from transformer.criterions.cross_entropy_with_decay import *
#from transformer.criterions.label_smoothed_cross_entropy2 import LabelSmoothedCrossEntropyCriterion as LabelSmoothedCrossEntropyCriterion2