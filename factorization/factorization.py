from typing import Dict, Tuple 

import numpy as np 


def factorize(
	row_col_map: Dict[int, Tuple[np.ndarray, np.ndarray]],
	col_row_map: Dict[int, Tuple[np.ndarray, np.ndarray]],
	D: int,
	n_iter: int=100, 
	reg: float=0.1, 
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Args:
		row_col_map: row: (columns' indices, values)
		col_row_map: column: (rows' indices, values)
		D: column dimension of resulting matrices
		n_iter: number of scans of all mappings
	
	Returns: 
		(U, W, B, C, MU) of shape 
		(N, D), (M, D), (N,), (M,) (,)
		such that
		np.dot(U[i], W[j]) + B[i] + C[j] + MU approximates R[i,j]
	"""
	N = max(row_col_map.keys()) + 1
	M = max(col_row_map.keys()) + 1

	U = np.random.normal(size=(N, D))
	W = np.random.normal(size=(M, D))
	B = np.random.normal(size=N)
	C = np.random.normal(size=M)
	MU = 0
	i = 0
	for _, values in row_col_map.values():
		MU += np.sum(values)
		i += len(values)

	MU = MU / i 

	for _ in range(n_iter):

		for row_id, (indices, values) in row_col_map.items():
			# w1^T*w1 + w2^T*w2 + ...
			n_cols = len(values)
			A = np.dot(W[indices].T, W[indices]) + reg * np.eye(D)
			b = np.sum(W[indices].T
			 * (values -  B[np.array([row_id] * n_cols)] - C[indices] - MU), axis=1)

			U[row_id] = np.linalg.solve(A, b)
			B[row_id] = (values - np.dot(W[indices], U[row_id].T) - C[indices] - MU).sum() / (n_cols + reg)

		for col_id, (indices, values) in col_row_map.items():
			# u1^T*u1 + u2^T*u2 + ...
			n_rows = len(values)
			A = np.dot(U[indices].T, U[indices]) + reg * np.eye(D)
			b = np.sum(U[indices].T 
				* (values - B[indices] - C[np.array([col_id] * n_rows)] - MU), axis=1)

			W[col_id] = np.linalg.solve(A, b)
			C[col_id] = (values - np.dot(U[indices], W[col_id].T) - B[indices] - MU).sum() / (n_rows + reg)

	return U, W, B, C, MU

"""
def predict(row_id, col_id):
		return np.dot(U[user_id], W[movie_id]) + B[user_id] + C[movie_id] + MU

	return predict
"""
