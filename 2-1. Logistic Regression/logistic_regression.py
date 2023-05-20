"""HW2: Logistic Regression."""

import numpy as np
import math


def hello():
    print('Hello from logistic_regression.py')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def naive_logistic_regression(X: np.ndarray, Y: np.ndarray, max_iters = 100) -> np.ndarray:
    """Computes the coefficients w from the datset (X, Y).

    This implementation uses a naive set of nested loops over the data.
    Specifically, we are required to use Newton's method (w = w - inv(H)*grad).

    Inputs:
      - X: Numpy array of shape (num_data, num_features+1).
           The first column of each row is always 1.
      - Y: Numpy array of shape (num_data) that has 0/1.
      - max_iters: Maximum number of iterations
    Returns:
      - w: Numpy array of shape (num_features+1) w[i] is the coefficient for the i-th
           column of X. The dimension should be matched with the second dimension of X.
    """
    N, d = X.shape
    w = np.zeros(d, dtype=X.dtype)

    for iter in range(max_iters):
        grad = np.zeros(d)
        H = np.zeros((d, d))
        for data_x, data_y in zip(X, Y):
            x = data_x.reshape(1,-1).T
            h = sigmoid(w.dot(x))
            H += x.dot(x.T)*(1-h)*h # should be -
            grad += (h - data_y)*data_x

        w =  w - np.matmul(np.linalg.inv(H), grad)
    return w


def vectorized_logistic_regression(X: np.ndarray, Y: np.ndarray, max_iters = 100) -> np.ndarray:
    """Computes the coefficients w from the dataset (X, Y).

    This implementation will vectorize the implementation in naive_logistic_regression,
    which implements Newton's method (w = w - inv(H)*grad).

    Inputs:
      - X: Numpy array of shape (num_data, num_features+1).
           The first column of each row is always 1.
      - Y: Numpy array of shape (num_data) that has 0/1.
      - max_iters: Maximum number of iterations
    Returns:
      - w: Numpy array of shape (num_features+1) w[i] is the coefficient for the i-th
           column of X. The dimension should be matched with the second dimension of X.
  """
    N, d = X.shape
    w = np.zeros(d, dtype=X.dtype)
    for iter in range(max_iters):
        # Implement this function using Newton's method.                                                           
        # * grad: gradient from all data samples.                             
        # * H: Hessian matrix from all data samples.                          
        # The shape of grad and H is the same as 'naive_logistic_regression'. 

        M = len(Y); x = np.dot(X,w); h = sigmoid(x)
        grad = 1 / M* X.T.dot(h-Y)  # grad.shape should be (d, ) at the end of this block
        diag = np.diag((h * (1 - h)).flatten())
        H = 1 / M * np.dot(X.T.dot(diag), X)  # H.shape should be (d, d) at the end of this block
        
        w =  w - np.matmul(np.linalg.inv(H), grad)
    return w


def compute_y_boundary(X_coord: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes the matched y coordinate value for the decision boundary from
    the x coordinate and coefficients w.

    Inputs:
      - X_coord: Numpy array of shape (d, ). List of x coordinate values.
      - w: Numpy array of shape (3, ) that stores the coefficients.

    Returns:
      - Y_coord: Numpy array of shape (d, ).
                 List of y coordinate values with respect to the coefficients w.
    """
    Y_coord = None
    # x_coord and coefficients w. + return/save y_coordindate into y_coord parameter
    # Assume that w[2] will not be zero.

    w0 = w[0]; w1 = w[1]; w2 = w[2]
    c = -w0/w2; m = -w1/w2
    Y_coord = m*X_coord + c

    return Y_coord
