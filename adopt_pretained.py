import torch
from fairseq.models.transformer import TransformerModel
from transformer.models.transformer_legacy import TransformerModel as TransformerModel2
# Load the pre-trained model
path= 'C:/Users/18816/.cache/torch/pytorch_fairseq/81a0be5cbbf1c106320ef94681844d4594031c94c16b0475be11faa5a5120c48.63b093d59e7e0814ff799bb965ed4cbde30200b8c93a44bf8c1e5e98f5c54db3'
pretained_model = TransformerModel.from_pretrained(path)
torch.save(pretained_model,"C:/Users/18816/.cache/torch/pytorch_fairseq/1")

