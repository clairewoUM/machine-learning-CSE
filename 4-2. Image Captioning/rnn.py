import numpy as np
from layers import fc_backward, fc_forward
from rnn_layers import (rnn_backward, rnn_forward, rnn_step_forward,
                        temporal_fc_backward, temporal_fc_forward,
                        temporal_softmax_loss, word_embedding_backward,
                        word_embedding_forward)

def hello():
    print("Hello from rnn.py!")

class CaptioningRNN:
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.
    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.
    Note that we don't use any regularization for the CaptioningRNN.

    We use the following architecture:

                                          fc        fc
                                           ^         ^
    (pretrained image features) -> fc ->  rnn  ->   rnn  -> ...
                                           ^         ^
                                          embed     embed
                                           ^         ^
                                          (word1)   (word2)
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; for this question, we just use 'rnn'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        self.params['Wx'] = np.random.randn(wordvec_dim, hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN to compute
        loss and gradients on all parameters.
        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V
        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this for the temporal softmax loss
        # (we only want to backpropagate on non-null captions)
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        ############################################################################
        # Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following.                   #
        # Follow each step in this order:                                          #
        # (1) Use an fc transformation to compute the initial hidden state         #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T-1, W).       #
        # (3) Use a vanilla RNN to process the sequence of input word vectors      #
        #     of shape (T, N, W), and produce hidden state vectors for all         #
        #     timesteps, producing an array of shape (T-1, N, H).                  #
        # (4) Use a (temporal) fc transformation to compute scores over the        #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T-1, V).                                          #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        # where N is the batch size, T caption length, V is the vocab size,        #
        # and W is the word vector dimension.                                      #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        ############################################################################
        # Step 1 
        initial_hidden_state, cache_ini_hiddenstate = fc_forward(features,W_proj,b_proj)
        
        #Step 2 
        word_vectors,cache_wv = word_embedding_forward(captions_in,W_embed)
        # N,T,W = word_vectors.shape
        # word_vectors = word_vectors.reshape(T,N,W)
        word_vectors = word_vectors.transpose(1,0,2)
        
        #Step 3
        hidden_state_vectors,cache_hsv = rnn_forward(word_vectors,initial_hidden_state,Wx,Wh,b)
        # T,N,H = hidden_state_vectors.shape
        # hidden_state_vectors = hidden_state_vectors.reshape(N,T,H)
        hidden_state_vectors = hidden_state_vectors.transpose(1,0,2)
        
        #Step 4 
        scores,cache_scores = temporal_fc_forward(hidden_state_vectors,W_vocab,b_vocab)
        # print(scores.shape)
        
        #Step 5
        loss,dout = temporal_softmax_loss(scores,captions_out,mask)
        
        ## Back propagation
        d_temp_back,grads['W_vocab'],grads['b_vocab'] = temporal_fc_backward(dout,cache_scores)
        d_temp_back = d_temp_back.transpose(1,0,2)
        
        d_rnn_back,d_initial_hs,grads['Wx'],grads['Wh'],grads['b'] = rnn_backward(d_temp_back,cache_hsv)
        # T,N,D = d_rnn_back.shape
        # d_rnn_back = d_rnn_back.reshape(N,T,D)
        d_rnn_back = d_rnn_back.transpose(1,0,2)
        
        grads['W_embed'] = word_embedding_backward(d_rnn_back,cache_wv)
        
        _,grads['W_proj'],grads['b_proj'] = fc_backward(d_initial_hs,cache_ini_hiddenstate) 

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.
        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.
        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.
        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        ###########################################################################
        # Implement test-time sampling for the model                              #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned fc transformation to the next hidden state to     #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     to the appropriate slot in the captions variable                    #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You may not use the rnn_forward functions here; you'll            #
        # need to call rnn_step_forward in a loop.                                #
        ###########################################################################
        h0 = features.dot(W_proj) + b_proj
        
        # print("h0 shape is:",h0.shape)
        # print("h0 is:",h0)
        start = (self._start * np.ones(N)).astype(np.int32)
        
        # print("start shape is",start.shape)
        # print(self._start)
        
        x = W_embed[start, :]      
        # print("W_embed shape is:",W_embed.shape)
        # print("W_embed is:",W_embed)
        # print("x shape is",x.shape)
        # print(x)
        
        h = h0
        
        for i in range(max_length):
          h,_ = rnn_step_forward(x,h,Wx,Wh,b)
          scores = np.dot(h,W_vocab) + b_vocab
          max_score = np.argmax(scores,axis=1)
          x = W_embed[max_score,:]
          captions[:,i] = max_score

        return captions
