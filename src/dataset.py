import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
pd.set_option('display.max_columns', None)

def load_ihdp_1000_data(index=0):
    data_train = np.load('IHDP/ihdp_b/ihdp_npci_1-1000.train.npz')
    data_test = np.load('IHDP/ihdp_b/ihdp_npci_1-1000.test.npz')

    X_train = data_train['x'][:,:,index].astype(np.float32)
    t_train = data_train['t'][:, index].astype(np.float32)
    y_train = data_train['yf'][:, index].astype(np.float32)
    mu0_train = data_train['mu0'][:, index].astype(np.float32)
    mu1_train = data_train['mu1'][:, index].astype(np.float32)
    
    X_test = data_test['x'][:,:,index].astype(np.float32)
    mu0_test = data_test['mu0'][:, index].astype(np.float32)
    mu1_test = data_test['mu1'][:, index].astype(np.float32)

    return X_train, t_train, y_train, mu0_train, mu1_train, X_test, mu0_test, mu1_test


def load_acic_data(dgp_folder=1, file_index=0):
    data = pd.read_csv('acic_2016/data_cf_all/x.csv')
    categorical_cols = ['x_2', 'x_21', 'x_24']
    X = pd.get_dummies(data, columns=categorical_cols, drop_first=True).astype(np.float32).values
    directory = f'acic_2016/data_cf_all/{dgp_folder}'
    all_files = os.listdir(directory)
    file = pd.read_csv(f'{directory}/{all_files[file_index]}').values.astype(np.float32)

    mu0 = file[:, 3]
    mu1 = file[:, 4]
    lower_mu0, upper_mu0 = np.percentile(mu0, [1, 99])
    lower_mu1, upper_mu1 = np.percentile(mu1, [1, 99])
    mask = (mu0 >= lower_mu0) & (mu0 <= upper_mu0) & (mu1 >= lower_mu1) & (mu1 <= upper_mu1)
    X = X[mask]
    file = file[mask]

    X_train, X_test, file_train, file_test = train_test_split(X, file, test_size=0.1, random_state=0)
    t_train = file_train[:,0]
    y_train = t_train * file_train[:,2] + (1 - t_train) * file_train[:,1]
    mu0_train = file_train[:,3]
    mu1_train = file_train[:,4]

    mu0_test = file_test[:,3]
    mu1_test = file_test[:,4]

    return X_train, t_train, y_train, mu0_train, mu1_train, X_test, mu0_test, mu1_test