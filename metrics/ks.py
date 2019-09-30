# ### KS
def KsScore(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    fpr,tpr,thresholds= roc_curve(y_true, y_pred)
    ks_score = max(tpr-fpr)
    return ks_score

# feval=evalerror
# ks指标不稳定,一般用auc来停