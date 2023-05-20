import numpy as np

from layers import relu_forward, fc_forward, fc_backward, relu_backward, softmax_loss
from cnn_layers import conv_forward, conv_backward, max_pool_forward, max_pool_backward


def hello():
    print("Hello from cnn.py!")


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters_1=6, num_filters_2=16, filter_size=5,
               hidden_dim=100, num_classes=10, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters_1: Number of filters to use in the first convolutional layer
        - num_filters_2: Number of filters to use in the second convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.dtype = dtype
        (self.C, self.H, self.W) = input_dim
        self.filter_size = filter_size
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # max pooling with pool height and width 2 with stride 2
        # For Linear layers, weights and biases should be initialized from a uniform distribution from -sqrt(k) to sqrt(k), where k = 1 / in_features
        # For Conv. layers, weights should be initialized from a uniform distribution from -sqrt(k) to sqrt(k), where k = 1 / (channels_in * kernel_size^2)                                    #
        # Same initialization as pytorch
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        pool_size = 2
        pool_stride = 2

        # Intermediate output dimension - Channels and P
        out_c1_dim = (num_filters_1, self.H - filter_size + 1, self.W - filter_size + 1)
        out_p1_dim = (num_filters_1, np.floor((out_c1_dim[1] - pool_size) / pool_stride + 1).astype(int), np.floor((out_c1_dim[2] - pool_size) / pool_stride + 1).astype(int))
        
        # Convolutional Layers
        self.params['W1'] = np.random.uniform(-1/np.sqrt(self.C * filter_size**2), 1/np.sqrt(self.C * filter_size**2), 
                                              (num_filters_1, self.C, filter_size, filter_size))
        
        self.params['W2'] = np.random.uniform(-1/np.sqrt(num_filters_1 * filter_size**2), 1/np.sqrt(num_filters_1 * filter_size**2), 
                                              (num_filters_2, num_filters_1, filter_size, filter_size))
        
        # Output dimensions
        out_c2_dim = (num_filters_2, out_p1_dim[1] - filter_size + 1, out_p1_dim[2] - filter_size + 1)
        out_p2_dim = (num_filters_2, np.floor((out_c2_dim[1] - pool_size) / pool_stride + 1).astype(int), np.floor((out_c2_dim[2] - pool_size) / pool_stride + 1).astype(int))
        
        # Linear Layers - hidden fully-connected layer
        self.params['W3'] = np.random.uniform(-1/np.sqrt(3*hidden_dim), 1/np.sqrt(3*hidden_dim), (np.prod(out_p2_dim), hidden_dim))
        self.params['b3'] = np.random.uniform(-1/np.sqrt(hidden_dim), 1/np.sqrt(hidden_dim), hidden_dim)
        # output affine layer
        self.params['W4'] = np.random.uniform(-1/np.sqrt(num_classes), 1/np.sqrt(num_classes), (hidden_dim, num_classes))
        self.params['b4'] = np.random.uniform(-1/np.sqrt(num_classes), 1/np.sqrt(num_classes), num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # forward pass for the three-layer convolutional net
        c1_out, c1_cache = conv_forward(X, W1)

        r1_out, r1_cache = relu_forward(c1_out)
        p1_out, p1_cache = max_pool_forward(r1_out, pool_param)

        c2_out, c2_cache = conv_forward(p1_out, W2)

        r2_out, r2_cache = relu_forward(c2_out)
        p2_out, p2_cache = max_pool_forward(r2_out, pool_param)
        
        fc3_out, fc3_cache = fc_forward(p2_out.reshape((p2_out.shape[0], W3.shape[0])), W3, b3)
        r3_out, r3_cache = relu_forward(fc3_out)
        scores, fc4_cache = fc_forward(r3_out, W4, b4)

        if y is None:
            return scores

        loss, grads = 0, {}
        
        # Backward pass for the three-layer convolutional net
        loss, soft_max_grads = softmax_loss(scores, y)
        fc4_grads, grads['W4'], grads['b4'] = fc_backward(soft_max_grads, fc4_cache)
        r3_grads = relu_backward(fc4_grads, r3_cache)
        fc3_grads, grads['W3'], grads['b3'] = fc_backward(r3_grads, fc3_cache)
        # print(f"fc3_grads.shape = {fc3_grads.shape}")
        
        p2_grads = max_pool_backward(fc3_grads.reshape(p2_out.shape), p2_cache)
        r2_grads = relu_backward(p2_grads, r2_cache)
        c2_grads, grads['W2'] = conv_backward(r2_grads, c2_cache)
        # print(f"c2_grads.shape = {c2_grads.shape}")
        
        p1_grads = max_pool_backward(c2_grads, p1_cache)
        r1_grads = relu_backward(p1_grads, r1_cache)
        _, grads['W1'] = conv_backward(r1_grads, c1_cache)

        return loss, grads
