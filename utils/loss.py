import numpy as np
import utils.regularization as reg
import utils.utils as util


def regularization(model, reg_type='l2', lam=1e-3):
    reg_types = dict(
        l1=reg.l1_reg,
        l2=reg.l2_reg
    )

    if reg_type not in reg_types.keys():
        raise Exception('Regularization type must be either "l1" or "l2"!')

    reg_loss = np.sum([
        reg_types[reg_type](model[k], lam)
        for k in model.keys()
        if k.startswith('W')
    ])

    return reg_loss


def cross_entropy(model, y_pred, y_train, lam=1e-3):
    m = y_pred.shape[0]

    prob = util.softmax(y_pred)
    log_like = -np.log(prob[range(m), y_train])

    data_loss = np.sum(log_like) / m
    reg_loss = regularization(model, reg_type='l2', lam=lam)

    return data_loss + reg_loss


def dcross_entropy(y_pred, y_train):
    m = y_pred.shape[0]

    grad_y = util.softmax(y_pred)
    grad_y[range(m), y_train] -= 1.
    grad_y /= m

    return grad_y