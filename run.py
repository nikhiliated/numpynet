import numpy as np
import utils.neuralnet as nn
from utils.solver import *
import sys


n_iter = 1000
alpha = 1e-3
mb_size = 64
reg = 1e-5
print_after = 100
p_dropout = 0.8
loss = 'cross_ent'
nonlin = 'relu'
solver = 'adam'


def prepro(X_train, X_val, X_test):
    mean = np.mean(X_train)
    return X_train - mean, X_val - mean, X_test - mean


if __name__ == '__main__':


    X_train, y_train = np.random.randn(55000, 1024), np.random.randint(9, size=(55000,))
    X_val, y_val = np.random.randn(5000, 1024), np.random.randint(9,size=(5000,))
    X_test, y_test = np.random.randn(10000, 1024), np.random.randint(9, size=(10000,))
    #print(y_test)

    #print('heyyyyyyyyy')
    #print(X_train.shape)
    #print(y_train.shape)
    #print(X_val.shape)
    #print(X_test.shape)
    #print('byeeeeeeee')

    M, D, C = X_train.shape[0], X_train.shape[1], y_train.max() + 1

    X_train, X_val, X_test = prepro(X_train, X_val, X_test)

    
    img_shape = (1, 32, 32)
    X_train = X_train.reshape(-1, *img_shape)
    X_val = X_val.reshape(-1, *img_shape)
    X_test = X_test.reshape(-1, *img_shape)

    solvers = dict(
        adam=adam
    )

    solver_fun = solvers[solver]
    accs = np.zeros(1)

    print()
    print('Itteration:')
    net = nn.ConvNet(10, C, H=128)

    net = solver_fun(
        net, X_train, y_train, val_set=(X_val, y_val), mb_size=mb_size, alpha=alpha,
        n_iter=n_iter, print_after=print_after
    )

    y_pred = net.predict(X_test)
    accs = np.mean(y_pred == y_test)

    print()
    print('Mean accuracy: {:.4f}, std: {:.4f}'.format(accs.mean(), accs.std()))
