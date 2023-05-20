"""
Implements a two-layer Neural Network classifier in PyTorch.
WARNING: SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import numpy as np
import random
import statistics

def hello():
    print('Hello from two_layer_net.py!')

def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, d_in) and contains a minibatch of N
    examples, where each example x[i] has d_in element.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_in)
    - w: A numpy array of weights, of shape (d_in, d_out)
    - b: A numpy array of biases, of shape (d_out,)
    Returns a tuple of:
    - out: output, of shape (N, d_out)
    - cache: (x, w, b)
    """
    out = None
    num_inputs = x.shape[0]
    input_shape = x.shape[1:]
    output_dim = b.shape[0]
    out = x.reshape(num_inputs, np.prod(input_shape)).dot(w) + b

    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, d_out)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_in)
      - w: Weights, of shape (d_in, d_out)
      - b: Biases, of shape (d_out,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_in)
    - dw: Gradient with respect to w, of shape (d_in, d_out)
    - db: Gradient with respect to b, of shape (d_out,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    num_inputs = x.shape[0]
    input_shape = x.shape[1:]
    output_dim = b.shape[0]

    dx = dout.dot(w.T)

    # reshaping to flatten the RGB image from CIFAR-10 dataset
    dx = dx.reshape(x.shape)
    dw = x.reshape(num_inputs, np.prod(input_shape)).T.dot(dout)
    db = np.sum(dout, axis = 0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(x, 0)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = np.where(x>=0, 1, 0)

    # need to multiply by the incoming upstream derivative 
    dx = dout*dx

    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)

    loss = 0.0
    N = x.shape[0]
    dx = probs.copy()
    # print(y.shape, probs.shape, dx.shape, dx[1].shape, np.shape(dx[:,1])) # y: (50,) // probs: (50, 10) // dx: (50, 10) // dx[i]: (10, ) // dx[:,i]: (50, )
    
    # Loss function
    loss = np.mean(-log_probs[range(N), y])

    # Gradient dx (= dL/dh)
    dx[range(N), y] -= 1.0
    dx /= N

    """
    for i in range(N):
       dx[:,y[i]] -= 1
    print(loss, dx)    
    """
    return loss, dx


class TwoLayerNet:
    """
    A fully-connected neural network with softmax loss that uses a modular
    layer design.

    We assume an input dimension of D, a hidden dimension of H,
    and perform classification over C classes.
    The architecture should be fc - relu - fc - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100,
                 num_classes=10, weight_scale=1e-3):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.params = {}
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Inputs:
        - X: Array of input data of shape (N, d_in)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        N = X.shape[0]

        # first layer activation
        H_1, cache_H1 = fc_forward(X, self.params['W1'], self.params['b1'])
        A_1, cache_relu = relu_forward(H_1)

        # second layer activation
        scores, cache_scores = fc_forward(A_1, self.params['W2'], self.params['b2'])

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, grad_L_wrt_scores = softmax_loss(scores, y)

        ## loss += 0.5 * (np.sum(self.params['W1'] * self.params['W1']) + np.sum(self.params['W2'] * self.params['W2']))
        grad_L_wrt_A_2, grad_L_wrt_W2, grad_L_wrt_b2 = fc_backward(grad_L_wrt_scores, cache_scores)
        grad_L_wrt_H_1 = relu_backward(grad_L_wrt_A_2, cache_relu)
        grad_L_wrt_X, grad_L_wrt_W1, grad_L_wrt_b1 = fc_backward(grad_L_wrt_H_1, cache_H1)

        grads['W1'] = grad_L_wrt_W1 #+ self.params['W1']
        grads['b1'] = grad_L_wrt_b1
        grads['W2'] = grad_L_wrt_W2 #+ self.params['W2'] 
        grads['b2'] = grad_L_wrt_b2

        return loss, grads
