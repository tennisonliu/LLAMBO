import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity

def calculate_mahalanobis_dist(hp_constraints, candidate, observed):
    observed = observed.copy()
    candidate = candidate.copy()
    if type(candidate) == list:
        candidate = pd.DataFrame(candidate).to_numpy()
        
    if type(observed) == pd.DataFrame:
        assert observed.columns.tolist() == list(hp_constraints.keys())
        observed = observed.to_numpy()

    # min-max scale observed and candidate to prevent certain hyperparameter dominating the distance
    for i, (hyperparam, value) in enumerate(hp_constraints.items()):
        observed[:, i] = (observed[:, i] - value[2][0]) / (value[2][1] - value[2][0])
        candidate[:, i] = (candidate[:, i] - value[2][0]) / (value[2][1] - value[2][0])

    # calculate malahanobis distance
    observed_mean = observed.mean(axis=0)
    cov = np.cov(observed, rowvar=False)
    if np.linalg.matrix_rank(cov) < cov.shape[0]:
        # use pseudo inverse
        inv_cov = np.linalg.pinv(cov)
    else:
        # inv_cov = np.linalg.inv(cov+1e-12*np.eye(cov.shape[0]))
        inv_cov = np.linalg.inv(cov)
    dist = []
    for i in range(candidate.shape[0]):
        diff = candidate[i, :] - observed_mean
        dist.append(np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T)))
    
    return np.mean(dist)


def calculate_gen_var(hp_constraints, candidate, observed):
    candidate = candidate.copy()
    observed = observed.copy()
    if type(candidate) == list:
        candidate = pd.DataFrame(candidate).to_numpy()

    if type(observed) == pd.DataFrame:
        assert observed.columns.tolist() == list(hp_constraints.keys())
        observed = observed.to_numpy()

    mean = observed.mean(axis=0)
    std = observed.std(axis=0)
    candidate = (candidate - mean) / std

    # remove duplicate row
    candidate = np.unique(candidate, axis=0)
    
    # calculate covariance matrix
    cov = np.cov(candidate, rowvar=False)

    # get determinant
    return np.linalg.det(cov + 1e-12*np.eye(cov.shape[0]))


def calculate_loglikelihood(hp_constraints, candidate, observed):
    observed = observed.copy()
    candidate = candidate.copy()
    if type(candidate) == list:
        candidate = pd.DataFrame(candidate).to_numpy()
        
    if type(observed) == pd.DataFrame:
        assert observed.columns.tolist() == list(hp_constraints.keys())
        observed = observed.to_numpy()

    # standardize against global mean and std of observed to ensure comparability
    mean = observed.mean(axis=0)
    std = observed.std(axis=0)
    candidate = (candidate - mean) / std
    # standardize observed to ensure comparability
    observed = (observed - mean) / std

    # fit KDE on observed
    kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(observed)

    # get log prob of candidate
    return kde.score_samples(candidate).mean()
    