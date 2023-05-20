"""
An implementation of SVMs using cvxopt.

"""
import warnings
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


def kernel_dot(X1, X2, kernel_params):
    """
    Returns the elementwise kernel vector between X1 and X2.
    I.e. kernel_dot(X1, X2)_i = k(X1_i, X2_i)
    Parameters
    ----------
    X1 : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    X2 : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -------
    np.ndarray (float64) of shape (n_samples,)
    """
    kp = kernel_params
    if kp['kernel'] == 'linear':
        return (X1 * X2).sum(1)
    elif kp['kernel'] == 'poly':
        return (kp['gamma'] * (X1 * X2).sum(1) + kp['coef0']) ** kp['degree']
    elif kp['kernel'] == 'rbf':
        return np.exp(-kp['gamma'] * ((X1 - X2)**2).sum(1))
    elif kp['kernel'] == 'sigmoid':
        return np.tanh(kp['gamma'] * (X1 * X2).sum(1) + kp['coef0'])
    else:
        raise ValueError(f"Unknown parameter: {kp['kernel']}")


def kernel_matrix(X1, X2, kernel_params):
    """
    Returns the pairwise kernel matrix between X1 and X2 (aka. gram matrix).
    I.e. kernel_dot(X1, X2)_{ij} = k(X1_i, X2_j)
    Parameters
    ----------
    X1 : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    X2 : np.ndarray (float64) of shape (m_samples, n_features)
        The input samples
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -------
    np.ndarray (float64) of shape (n_samples, m_samples)
    """
    kp = kernel_params
    if kp['kernel'] == 'linear':
        return X1 @ X2.T
    elif kp['kernel'] == 'poly':
        return (kp['gamma'] * X1 @ X2.T + kp['coef0']) ** kp['degree']
    elif kp['kernel'] == 'rbf':
        pw_norm = ((np.expand_dims(X1, 1) - np.expand_dims(X2, 0))**2).sum(2)
        return np.exp(-kp['gamma'] * pw_norm)
    elif kp['kernel'] == 'sigmoid':
        return np.tanh(kp['gamma'] * X1 @ X2.T + kp['coef0'])
    else:
        raise ValueError(f"Unknown parameter: {kp['kernel']}")


def get_qp_params(X, y, C, kernel_params):
    """
    Return the parameters to pass into cvxopt.solvers.qp for the SVM dual problem.
    Parameters
    ----------
    X : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    y : np.ndarray (int64) of shape (n_samples,)
        Target labels, with values either -1 or 1.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -------
    Arguments to be passed into cvxopt.solvers.qp
    P : ndarray of shape (n_samples, n_samples)
    q : ndarray of shape (n_samples,)
    G : ndarray of shape (n_1, n_samples)
    h : ndarray of shape (n_1,)
    A : ndarray of shape (n_2, n_samples)
    b : ndarray of shape (n_2,)
    """
    P, q, G, h, A, b = None, None, None, None, None, None
    N,_ = X.shape
    
    ker_mat = kernel_matrix(X,X, kernel_params)
    y = y.reshape(-1,1)
    
    P = np.dot(y,y.transpose())*ker_mat
    q = -np.ones((N))

    G = np.vstack((np.eye(N)*-1,np.eye(N)))
    h = np.hstack((np.zeros(N), np.ones(N)*C))
    
    A = y.reshape(1, -1)
    A = A.astype(G.dtype)
    b = np.zeros(1)

    # Check for shapes
    assert P.dtype == q.dtype == G.dtype == h.dtype \
            == A.dtype == b.dtype == np.float_, 'outputs must be numpy floats'
    assert len(P.shape) == len(G.shape) == len(A.shape) == 2, 'P, G, and A must be matrices'
    assert len(q.shape) == len(h.shape) == len(b.shape) == 1, 'q, h, and b must be vectors'
    assert P.shape[0] == P.shape[1] == X.shape[0], 'wrong shape for P'
    assert q.shape[0] == X.shape[0], 'wrong shape for q'
    assert G.shape == (h.shape[0], X.shape[0]), 'wrong shape for G or h'
    assert A.shape == (b.shape[0], X.shape[0]), 'wrong shape for A or b'
    return P, q, G, h, A, b


def fit_bias(X, y, alpha, kernel_params):
    """
    Return the calculated values the bias b using the given support vector values (alpha).
    Parameters
    X : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples used to find alpha
    y : np.ndarray (int64) of shape (n_samples,)
        Target labels, with values either -1 or 1, used to find alpha
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -----
    Scalar (float64)
    """
    # Note: due to cvxopt qp implementation and numerical floats,
    #   alpha[i] may not be a support vector even if alpha[i] > 0
    #   Instead, we use:
    is_support = alpha > 1e-4
    S = is_support.flatten()
    alpha_s = alpha[S] ; y_s = y[S] ; X_s = X[S]
    
    ker_mat = kernel_matrix(X_s, X_s, kernel_params)

    term1 = np.sum(y_s * alpha_s * ker_mat, axis=1)
    b = np.sum(y_s - term1)/X_s.shape[0]

    return b


def decision_function(X, X_train, y_train, b, alpha, kernel_params):
    """
    Return the calculated values for (w^T X + b) using the given support vector values (alpha).
    Parameters
    ----------
    X : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    X_train : np.ndarray (float64) of shape (n_train_samples, n_features)
        The input samples used to find alpha
    y_train : np.ndarray (int64) of shape (n_train_samples,)
        Target labels, with values either -1 or 1, used to find alpha
    b : scalar (float64)
        Bias value computed after training (along with alpha)
    alpha : np.ndarray (float64) of shape (n_train_samples,)
        Support vector values (solution of the cvxopt qp method)
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -------
    np.ndarray (float64) of shape (n_samples)
    """
    # Note: due to cvxopt qp implementation and numerical floats,
    #   alpha[i] may not be a support vector even if alpha[i] > 0
    #   Instead, we use:
    is_support = alpha > 1e-4
    h = np.zeros(X.shape[0])

    # Calculate h(x) = w^T X + b using the dual representation
    S = is_support.flatten()
    alpha_s = alpha[S] ; y_s = y_train[S] ; X_s = X_train[S]
    
    ker_mat = kernel_matrix(X,X_s, kernel_params)

    term1 = np.sum(y_s * alpha_s * ker_mat, axis=1)
    h = term1 + b

    assert h.shape == (X.shape[0],)
    return h


class CVXOPTSVC:
    """C-Support Vector Classification.

    Sklearn-style SVM implementation using the cvxopt solver.
    
    Parameters and interface are adapted from
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
        default='rbf'
        Specifies the kernel type to be used in the algorithm.
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.
    gamma : float, default=1.0
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        - must be non-negative.
    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    """

    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma=1.0,
        coef0=0.0
    ):
        self.C = C
        self.kernel_params = {
            'kernel': kernel,
            'degree': degree,
            'gamma': gamma,
            'coef0': coef0
        }

    @staticmethod
    def _H_linear(X, y):
        Xy = X * np.expand_dims(y, 1)
        return Xy @ Xy.T

    def fit(self, X, y):
        # Initialize and computing H. Note the 1. to force to float type
        y = y * 2 - 1  # transform to [-1, 1]
        self.X_train = X.copy()
        self.y_train = y.copy()

        # Convert into cvxopt format
        _P, _q, _G, _h, _A, _b = get_qp_params(X, y, self.C, self.kernel_params)
        P = cvxopt_matrix(_P)
        q = cvxopt_matrix(np.expand_dims(_q, 1))
        G = cvxopt_matrix(_G)
        h = cvxopt_matrix(_h)
        A = cvxopt_matrix(_A)
        b = cvxopt_matrix(_b)

        # Run solver
        cvxopt_solvers.options['show_progress'] = False
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(sol['x']).squeeze(1)

        self.support_ = np.where(self.alpha > 1e-4)
        self.b = fit_bias(X, y, self.alpha, self.kernel_params)

        return self

    def decision_function(self, X):
        return decision_function(X, self.X_train, self.y_train, self.b, self.alpha, self.kernel_params)

    def predict(self, X):
        h = decision_function(X, self.X_train, self.y_train, self.b, self.alpha, self.kernel_params)
        return (h >= 0).astype(np.int_)

    def score(self, X, y):
        return (self.predict(X) == y).mean()
