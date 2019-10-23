from sklearn.metrics import log_loss

# ### logloss
def LogLoss_(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    loss = log_loss(y_true, y_pred)
    return loss


