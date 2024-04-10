def confidence_loss(probability,loss):
    assert probability.dim()<=1 and loss.dim()<=1
    probability=probability.mean()
    probability=probability/probability
    return abs(probability*loss)