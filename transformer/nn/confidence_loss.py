def confidence_loss(probability,loss):
    assert probability.dim()<=1 and loss.dim()<=1
    return (probability/1.7+0.7)*abs(loss)