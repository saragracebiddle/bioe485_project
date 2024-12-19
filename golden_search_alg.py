import numpy as np
import math
from compute_V import *
from compute_tau import *
from laguerre_basis import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def do_predictions(k, alpha, L, dt, iirf, datasets):
    V = compute_V(k, alpha, L, iirf)
    B = basis_array(k, alpha, L)
    regr = linear_model.LinearRegression()
    pred_taus = []
    for d in datasets:
        regr.fit(V,d)
        taus = compute_tau_avg(regr.coef_, dt, B, k) * 1e9
        pred_taus.append(taus)

    return pred_taus



def golden_search(L, dt, iirf, datasets, tau, k, n_iter = 1):
    lower_limit = 0
    upper_limit = 1
    alpha = upper_limit + lower_limit /2

    invphi = (math.sqrt(5) - 1) / 2  # 1 / phi

    train_data = datasets[:640]
    train_taus = tau[:640]
    val_data = datasets[640:]
    val_taus = tau[640:]

    loss = []
    val_loss = []
    for n in range(n_iter):
        alpha = (upper_limit + lower_limit) /2
        # record mse with test data
        tau_alpha = do_predictions(k=k, alpha=alpha, L=L, dt=dt, iirf=iirf, datasets=train_data)
        loss.append(mean_squared_error(train_taus, tau_alpha))
        # record loss with validation data
        val_tau_alpha = do_predictions(k=k, alpha=alpha, L=L, dt=dt, iirf=iirf, datasets=val_data)
        val_loss.append(mean_squared_error(val_taus, val_tau_alpha))

        # next iteration
        c = upper_limit - (upper_limit-lower_limit) * invphi

        d = lower_limit + (upper_limit-lower_limit) * invphi
        

        tau_c = do_predictions(k=k, alpha =c, L=L, dt=dt, iirf=iirf, datasets=train_data)
        tau_d = do_predictions(k=k, alpha = d, L=L, dt=dt, iirf=iirf, datasets=train_data)

        # update parameters
        if mean_squared_error(train_taus, tau_c) < mean_squared_error(train_taus, tau_d):
            upper_limit = d
        else:
            lower_limit = c
        

    return loss, val_loss,  alpha