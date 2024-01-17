import numpy as np

def get_lower_triangle(matrix):
  return np.tril(matrix, -1)

def get_diagonal(matrix):
  return np.diag(np.diag(matrix))

def get_upper_triangle(matrix):
  return np.triu(matrix, 1)

def get_matrix_inverse(matrix):
  return np.linalg.inv(matrix)

def get_matrix_norm(matrix):
  return np.linalg.norm(matrix)

def get_b(matrix,n):
  return np.dot(matrix, np.ones(n))

def print_matrix(matrix):
  for row in matrix:
    print(row)

def check_divergence(matrix):
  return matrix.max() > 1e100