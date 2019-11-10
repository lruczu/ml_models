from typing import Dict, Tuple 

import numpy as np 


def factorize(
	row_col_map: Dict[int, Tuple[np.ndarray, np.ndarray]],
	col_row_map: Dict[int, Tuple[np.ndarray, np.ndarray]],
	D: int,
	n_iter: int=100, 
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Args:
		row_col_map: row: (columns' indices, values)
		col_row_map: column: (rows' indices, values)
		D: column dimension of resulting matrices
		n_iter: number of scans of all mappings
	
	Returns: 
		(U, W) of shape (N, D), (M, D) such that 
		np.dot(U, W.T) approximates R.
	"""
	N = max(row_col_map.keys()) + 1
	M = max(col_row_map.keys()) + 1

	U = np.random.normal(size=(N, D))
	W = np.random.normal(size=(M, D))

	for _ in range(n_iter):

		for row_id, (indices, values) in row_col_map.items():
			# w1^T*w1 + w2^T*w2 + ...
			A = np.dot(W[indices].T, W[indices])
			b = np.sum(W[indices].T * values, axis=1)

			U[row_id] = np.linalg.solve(A, b)

		for col_id, (indices, values) in col_row_map.items():
			# u1^T*u1 + u2^T*u2 + ...
			A = np.dot(U[indices].T, U[indices])
			b = np.sum(U[indices].T * values, axis=1)

			W[row_id] = np.linalg.solve(A, b)

	return U, W

"""
def predict(row_id, col_id):
	return np.dot(U[row_id], W[col_id])

return predict
"""
