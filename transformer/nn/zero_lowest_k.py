import torch

def zero_lowest_k(tensor, k, dim):
    if k == 0:
        return tensor
    # Compute the indices of the values to zero out using `argsort`
    indices = torch.argsort(tensor, dim=dim, descending=False)
    
    # Use `scatter_` to create a mask that will be False (zero out) for the lowest k values
    mask = torch.ones_like(tensor, dtype=torch.bool)
    mask.scatter_(dim=dim, index=indices.narrow(dim, 0, k), value=False)
    
    # Use the `where` function to selectively zero out the lowest k values
    # This applies the condition across the tensor to zero values where mask is False
    result = torch.where(mask, tensor, torch.zeros_like(tensor))

    return result
