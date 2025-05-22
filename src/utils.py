import numpy as np

def pehe(cate_pred, cate_true):
    pehe = np.mean((cate_true - cate_pred) ** 2)
    sqrt_pehe = np.sqrt(pehe)
    return pehe, sqrt_pehe

def cate_evaluations(cate_pred_train, cate_pred_test, mu0_train, mu1_train, mu0_test, mu1_test):
    cate_train = mu1_train - mu0_train
    pehe_train, sqrt_pehe_train = pehe(cate_pred_train, cate_train)
    cate_test = mu1_test - mu0_test
    pehe_test, sqrt_pehe_test = pehe(cate_pred_test, cate_test)
    return pehe_train, sqrt_pehe_train, pehe_test, sqrt_pehe_test

