import numpy as np

# Sample NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 7]])

# Count unique values
for i, j in np.unique(arr, return_counts=True):
    print(i, j)
