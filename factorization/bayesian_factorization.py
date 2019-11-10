from typing import Dict, List, Tuple

import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import wishart


def factorize(
    row_col_map: Dict[int, Tuple[np.ndarray, np.ndarray]],
    col_row_map: Dict[int, Tuple[np.ndarray, np.ndarray]],
    U: np.ndarray,
    V: np.ndarray,
    beta: float,
    alpha: float,
    n_iter: int = 100,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Args:
        row_col_map: row: (columns' indices, values)
        col_row_map: column: (rows' indices, values)
        U: initial point for user matrix
        V: inital point for movie matrix
        beta: scaling parameter in Normal-Wishart distribution
        alpha: standard deviation in N(r_ij | u_i^{T} * w_j, alpha ** 2)
        n_iter: number of steps in Gibbs sampling

    Returns:
        two lists with posterior samples of
        users and movies matrices, i.e. U and V.
    """
    assert U.shape[1] == V.shape[1]

    N = max(row_col_map.keys()) + 1
    M = max(col_row_map.keys()) + 1

    assert U.shape[0] == N
    assert V.shape[0] == M

    D = U.shape[1]

    # some hyperparameters
    W0 = np.eye(D)
    df = D

    U_list = []
    V_list = []

    for _ in range(n_iter):
        # sample precision_u
        mu = U.mean(axis=0)
        coef = beta * N / (beta + N)
        W0_ = W0 + np.dot((U - mu).T, U - mu) + coef * np.dot(mu.reshape((-1, 1)), mu.reshape((1, -1)))
        df_ = df + N
        precision_u = wishart.rvs(df_, np.linalg.inv(W0_))
        mu_u = multivariate_normal(N * mu / (beta + N), cov=np.linalg.inv((beta + N) * precision_u))
        mu_u = mu_u.reshape((-1, 1))

        # sample precision_v
        mu = V.mean(axis=0)
        coef = beta * M / (beta + M)
        W0_ = W0 + np.dot((V - mu).T, V - mu) + coef * np.dot(mu.reshape((-1, 1)), mu.reshape((1, -1)))
        df_ = df + M
        precision_v = wishart.rvs(df_, np.linalg.inv(W0_))
        mu_v = multivariate_normal(M * mu / (beta + M), cov=np.linalg.inv((beta + M) * precision_v))
        mu_v = mu_v.reshape((-1, 1))

        # sample vectors from U
        for i in range(N):
            indices, ratings = row_col_map[i]
            precision_ = precision_u + np.dot(V[indices].T, V[indices]) / alpha

            mu_ = (V[indices].T * ratings).sum(axis=1, keepdims=True) / alpha
            mu_ = mu_ + np.dot(precision_u, mu_u)
            mu_ = np.dot(np.linalg.inv(precision_), mu_).flatten()

            U[i] = multivariate_normal(mu_, cov=np.linalg.inv(precision_))

        # sample vectors from V
        for j in range(M):
            indices, ratings = col_row_map[j]
            precision_ = precision_v + np.dot(U[indices].T, U[indices]) / alpha

            mu_ = (U[indices].T * ratings).sum(axis=1, keepdims=True) / alpha
            mu_ = mu_ + np.dot(precision_v, mu_v)
            mu_ = np.dot(np.linalg.inv(precision_), mu_).flatten()

            V[j] = multivariate_normal(mu_, cov=np.linalg.inv(precision_))

        U_list.append(U.copy())
        V_list.append(V.copy())
    return U_list, V_list
