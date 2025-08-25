import numpy as np

def dtw(series_1, series_2, norm_func = np.linalg.norm):
	"""
	Efficient DTW to help with segments_refine logic
	Credits: https://github.com/talcs/simpledtw
	"""
	matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
	matrix[0,:] = np.inf
	matrix[:,0] = np.inf
	matrix[0,0] = 0
	for i, vec1 in enumerate(series_1):
		for j, vec2 in enumerate(series_2):
			cost = norm_func(vec1 - vec2)
			matrix[i + 1, j + 1] = cost + min(matrix[i, j + 1], matrix[i + 1, j], matrix[i, j])
	matrix = matrix[1:,1:]
	i = matrix.shape[0] - 1
	j = matrix.shape[1] - 1
	matches = []
	mappings_series_1 = [list() for v in range(matrix.shape[0])]
	mappings_series_2 = [list() for v in range(matrix.shape[1])]
	while i > 0 or j > 0:
		matches.append((i, j))
		mappings_series_1[i].append(j)
		mappings_series_2[j].append(i)
		option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
		option_up = matrix[i - 1, j] if i > 0 else np.inf
		option_left = matrix[i, j - 1] if j > 0 else np.inf
		move = np.argmin([option_diag, option_up, option_left])
		if move == 0:
			i -= 1
			j -= 1
		elif move == 1:
			i -= 1
		else:
			j -= 1
	matches.append((0, 0))
	mappings_series_1[0].append(0)
	mappings_series_2[0].append(0)
	matches.reverse()
	for mp in mappings_series_1:
		mp.reverse()
	for mp in mappings_series_2:
		mp.reverse()
	
	return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix