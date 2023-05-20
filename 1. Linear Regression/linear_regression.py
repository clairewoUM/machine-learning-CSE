"""Task 1: Linear Regression."""

from typing import Any, Dict, Tuple
#from sklearn.utils import shuffle
import numpy as np


def load_data():
    """Load the data required for Q2."""
    x_train = np.load('data/q2xTrain.npy')
    y_train = np.load('data/q2yTrain.npy')
    x_test = np.load('data/q2xTest.npy')
    y_test = np.load('data/q2yTest.npy')
    return x_train, y_train, x_test, y_test


def generate_polynomial_features(x: np.ndarray, M: int) -> np.ndarray:
    """Generate the polynomial features.

    Args:
        x: A numpy array with shape (N, ).
        M: the degree of the polynomial.
    Returns:
        phi: A feature vector represented by a numpy array with shape (N, M+1);
          each row being (x^{(i)})^j, for 0 <= j <= M.
    """
    N = len(x)
    phi = np.zeros((N, M + 1))
    for m in range(M + 1):
        phi[:, m] = np.power(x, m)
    return phi


def loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    r"""The least squares training objective for the linear regression.

    Args:
        X: the feature matrix, with shape (N, M+1).
        y: the target label for regression, with shape (N, ).
        w: the linear regression coefficient, with shape (M+1, ).
    Returns:
        The least square error term with respect to the coefficient weight w,
        E(\mathbf{w}).
    """
    y_pred = np.matmul(X, w)
    squared_error = (y - y_pred) ** 2

    return 0.5 * np.sum(squared_error)


def batch_gradient_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    eta: float = 0.01,
    max_epochs: int = 10000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Batch gradient descent for linear regression that fits the
    feature matrix `X_train` to target `y_train`.

    Args:
        X_train: the feature matrix, with shape (N, M+1).
        y_train: the target label for regression, with shape (N, ).
        eta: Learning rate.
        max_epochs: Maximum iterations (epochs) allowed.
    Returns: A tuple (w, info)
        w: The coefficient of linear regression found by GD. Shape (M+1, ).
        info: A dict that contains additional information (see the notebook).
    """
    converg = False
    iter = 0
    n = X_train.shape[0] # number of samples
    # Randomly initializing weights
    w = np.random.random(X_train.shape[1])
    loss_list = [] 

    while not converg:
        counter = 0
        iter = iter + 1
        # Gradient for each sample
        g = sum([np.dot((np.dot(w.T, X_train[i]) - y_train[i]), X_train[i]) for i in range(n)])
        
        # Update temp for w
        temp = w - eta*g
        # Update w from temp
        w = temp
        
        # Loss function
        loss_ = loss(X_train, y_train, w) #sum([(np.dot(w.T, X_train[i]) - y_train[i])**2 for i in range(n)])
        loss_list.append(loss_)

        if (iter >= max_epochs):
            converg = True
            break
            #print("Exceeds maximum iterations-")
        else:
            if (loss_ / n <= 0.2):
                counter = counter + 1
                if (counter == 100):
                    converg = True
                    #break

    info = {'train_losses': loss_list, 'number of iterations': iter}

    return w, info


def stochastic_gradient_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    eta=4e-2,
    max_epochs=10000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Stochastic gradient descent for linear regression that fits the
    feature matrix `X_train` to target `y_train`.

    Args:
        X_train: the feature matrix, with shape (N, M+1).
        y_train: the target label for regression, with shape (N, ).
        eta: Learning rate.
        max_epochs: Maximum iterations (epochs) allowed.
    Returns: A tuple (w, info)
        w: The coefficient of linear regression found by SGD. Shape (M+1, ).
        info: A dict that contains additional information (see the notebook).
    """
    np.random.seed(545)
    converg = False
    iter = 0
    n = X_train.shape[0] # number of samples
    loss_list = []
    # Randomly initializing weights
    w = np.random.random(X_train.shape[1])
    temp = np.random.random(X_train.shape[1])

    while not converg:
        counter = 0
        iter = iter + 1
        #X_train = shuffle(X_train)
        for i in range(n):
            
            # Gradient for each training sample
            g_pre = np.dot(X_train[i], w.T) - y_train[i]
            g = np.dot(g_pre, X_train[i])

            # Update temp for w
            temp = w - eta*g
            # Update w from temp
            w = temp 

            # Loss function
            loss_ = loss(X_train, y_train, w) #sum([(y_train[i] - np.dot(X_train[i], w.T))**2 for i in range(n)])
        loss_list.append(loss_)

        if (iter >= max_epochs):
            converg = True
            break
            #print("Exceeds maximum iterations-")
        else:
            if (loss_ /n <= 0.2):
                counter = counter+1
                if (counter == 100):
                    converg = True
                    #break
            

    info = {'train_losses': loss_list, 'number of iterations': iter}

    return w, info


def closed_form(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    lam: float = 0.0,
) -> np.ndarray:
    """Return the closed form solution of linear regression.

    Arguments:
        X_train: The X feature matrix, shape (N, M+1).
        y_train: The y vector, shape (N).
        M: The degree of the polynomial to generate features for.
        lam: The regularization coefficient lambda.

    Returns:
        The (optimal) coefficient w for the linear regression problem found,
        a numpy array of shape (M+1, ).
    """
    n = X_train.shape[1] # number of samples

    # Coefficients of optimal solution (Closed-form solution)
    #w = np.matmul(np.matmul(inv(np.matmul(X_train, X_train)), X_train), y_train)
    xTx = np.matmul(X_train.T, X_train) + (np.identity(n)*lam)
    w = np.matmul(np.matmul(np.linalg.inv(xTx), X_train.T), y_train)

    return w


def closed_form_locally_weighted(
    X_train: np.ndarray,
    y_train: np.ndarray,
    r_train: np.ndarray,
) -> np.ndarray:
    """Return the closed form solution of locally weighted linear regression.

    Arguments:
        x_train: The X feature matrix, shape (N, M+1).
        y_train: The y vector, shape (N, ).
        r_train: The local weights for data point. Shape (N, ).

    Returns:
        The (optimal) coefficient for the locally weighted linear regression
        problem found. A numpy array of shape (M+1, ).
    """
    xTr = np.matmul(X_train.T, np.diag(r_train))

    w = np.matmul(np.linalg.inv(np.matmul(xTr, X_train)), np.matmul(xTr, y_train))

    return w
