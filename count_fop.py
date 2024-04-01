import torch
from fvcore.nn import FlopCountAnalysis
from transformer.models.transformer_legacy import *

model_name="transformer_wmt_en_de"
model=transformer_wmt_en_de({})
input=torch.randn(1, 1, 512)
flops = FlopCountAnalysis(model, input)

print(f"Total FLOPs: {flops.total()}")
