import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from scipy import stats, optimize
import numpy as np



def func_for_nlinfit(x, bayta1, bayta2, bayta3, bayta4, bayta5):  
    logisticPart = 0.5 - 1.0 / (1 + np.exp(bayta2 * (x - bayta3)))
    yhat = bayta1 * logisticPart + bayta4 * x + bayta5
    return yhat


def f_err(p, y, x):
    return (y - func_for_nlinfit(x, *p)) * (y - func_for_nlinfit(x, *p))


def nlinfit(mos, predict_mos):
    beta1 = 10
    beta2 = 0
    beta3 = np.mean(predict_mos)
    beta4 = 0.1
    beta5 = 0.1
    beta = np.array([beta1, beta2, beta3, beta4, beta5], dtype=float)
    c = optimize.leastsq(f_err, beta, args=(mos, predict_mos),
                         maxfev=1000000) 
    return c


def nlinfit2(mos, predict_mos):
    p_est, err_est = optimize.curve_fit(func_for_nlinfit, predict_mos, mos)
    return p_est


def corr_value(mos, predict_mos, fit_flag=True):
    if fit_flag:
        c = nlinfit(mos, predict_mos)
        yhat = func_for_nlinfit(predict_mos, *c[0])
        PLCC, l1 = stats.pearsonr(mos, yhat)
        SROCC, l1 = stats.spearmanr(mos, predict_mos)
        KROCC, l2 = stats.kendalltau(mos, predict_mos)
        RMSE = np.sqrt(np.sum((yhat - predict_mos) ** 2) / len(mos))
    else:
        PLCC, l1 = stats.pearsonr(mos, predict_mos)
        SROCC, l1 = stats.spearmanr(mos, predict_mos)
        KROCC, l2 = stats.kendalltau(mos, predict_mos)
        RMSE = np.sqrt(np.sum((mos - predict_mos) ** 2) / len(mos))
    return PLCC, SROCC, KROCC, RMSE
