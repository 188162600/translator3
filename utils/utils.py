import torch
def linear_interp(x:torch.Tensor, xp:torch.Tensor, fp:torch.Tensor):
    """
    Linear interpolation on PyTorch tensors.
    :param x: The x-coordinates at which to evaluate the interpolated values.
    :param xp: The x-coordinates of the data points, must be increasing.
    :param fp: The y-coordinates of the data points, same length as xp.
    :return: Interpolated values same shape as x.
    """
    # Find indices of the nearest x points, such that xp[ind-1] < x <= xp[ind]
    ind = torch.searchsorted(xp, x, right=True)
    ind = ind.clamp(min=1, max=len(xp)-1)
    xp_left = xp[ind - 1]
    xp_right = xp[ind]
    fp_left = fp[ind - 1]
    fp_right = fp[ind]

    # Linear interpolation formula
    interp_val = fp_left + (x - xp_left) * (fp_right - fp_left) / (xp_right - xp_left)
    return interp_val
def cosine_similarity_2d(x:torch.Tensor, y:torch.Tensor):
    """
    Compute the cosine similarity between two 2D tensors.
    :param a: A tensor of shape (N, M).
    :param b: A tensor of shape (N, M).
    :return: A tensor of shape (N,) containing the cosine similarity between each row of a and b.
    """

    x_norm = x / x.norm(dim=1, keepdim=True)
    y_norm = y / y.norm(dim=1, keepdim=True)

    # Compute the cosine similarity
    # Resulting shape is [16, 2]
    return  torch.mm(x_norm, y_norm.transpose(0, 1))
def confidence_loss(probability,loss):
    return (probability/1.7+0.7)*loss