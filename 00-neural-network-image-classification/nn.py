from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import optimization

# 1. data preprocessing
def data_preprocess_from_keras_mnist(validation_number):
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    length = X_train.shape[0]

    # dataset visualization
    print("The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.")

    fig, axs = plt.subplots(ncols = 8, nrows = 6)#, layout = 'constrained')
    cnt = 0
    for row in range(6):
      for col in range(8):
          axs[row, col].imshow(X_train[cnt], cmap = 'gray')
          axs[row, col].axis('off')
          cnt += 1
    plt.show()

    X_validation = X_train[:validation_number].astype('float32').reshape((validation_number, -1)) # flatten images
    y_validation = y_train[:validation_number].astype('int32')

    X_train = X_train[validation_number:].astype('float32').reshape((length - validation_number, -1)) # flatten images
    y_train = y_train[validation_number:].astype('int32')

    X_test = X_test.astype('float32').reshape((X_test.shape[0], -1)) # flatten images
    y_test = y_test.astype('int32')

    print('<Training set array size>:')
    print('X_train =', X_train.shape)
    print('y_train =', y_train.shape, '\n')
    print('<Validation set array size>:')
    print('X_validation =', X_validation.shape)
    print('y_validation =', y_validation.shape, '\n')
    print('<Test set array size>:')
    print('X_test =', X_test.shape)
    print('t_test =', y_test.shape)

    data = {}
    data['X_train'] = X_train
    data['y_train'] = y_train
    data['X_validation'] = X_validation
    data['y_validation'] = y_validation
    data['X_test'] = X_test
    data['y_test'] = y_test 

    return data


# 2. model architecture and implementation
class FCNeuralNet:
    """
    The architecture of this fully-connected neural network is fc - relu - fc - softmax.
    """

    # 2-1 Parameter initialization
    def __init__(self, input_dim=28*28, hidden_dim=100,
                 num_classes=10, weight_scale=1e-3):
        """
        Initialize a new network.
        """
        self.params = {}
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        
        Returns:
        run a test-time forward pass of the model and return if y is None:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        o/w run a training-time forward and backward pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """

        # fc - relu - fc - softmax
        out1, cache1 = nn_forward(X, self.params["W1"], self.params["b1"])
        out2, cache2 = relu_forward(out1)
        scores, cache3 = nn_forward(out2, self.params["W2"], self.params["b2"])

        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dscores = softmax_loss(scores, y) 
        dx2, dW2, db2 = nn_backward(dscores, cache3)
        dx_relu = relu_backward(dx2, cache2)
        _, dW1, db1 = nn_backward(dx_relu, cache1)

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads
    

def nn_forward(x, w, b):
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

    out = np.matmul(x, w) + b
    cache = (x, w, b)
    return out, cache


def nn_backward(dout, cache):
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
    dx = np.matmul(dout, w.T)
    dw = np.matmul(x.T, dout)
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
    out = np.where(x > 0, x, 0)
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
    dx = np.where(x > 0, dout, 0)
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
    #dx = x * (probs - np.reshape(y, (y.shape[0], 1))) / N
    #y_diag = np.diag(y)
    #loss = np.sum(np.matmul(probs.T, y_diag)) / N
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx


#. Optimization
class Solver(object):
    """
    This Solver performs stochastic gradient descent using different
    update rules defined in optimization.py.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data with the following:
            'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
            'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
            'y_train': Array of shape (N_train,) giving labels for training images
            'y_val': Array of shape (N_val,) giving labels for validation images

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optimization.py.
            Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
            passed to the chosen update rule. Each update rule requires different
            hyperparameters (see optimization.py) but all update rules require a
            'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the learning
            rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient during
            training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every print_every
            iterations.
        - verbose: Boolean; if set to false then no output will be printed during
            training.
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_validation']
        self.y_val = data['y_validation']

        # Unpack keyword arguments
        update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 500)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 1)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optimization, update_rule):
            raise ValueError('Invalid update_rule "%s"' % update_rule)
        self.update_rule = getattr(optimization, update_rule)

        self._reset()


    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config


    def check_accuracy(self, X, y, num_samples=None):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_in)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
            on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using too
            much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
            classified by the model.
        """
        if self.verbose:
            print('check accuracy')
        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = int(N / self.batch_size)
        if N % self.batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            if self.verbose:
                print('batch index', i)
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        if self.verbose:
            print('check accuracy done')
        return acc


    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = int(self.num_epochs * iterations_per_epoch)

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                             t + 1, num_iterations, self.loss_history[-1]))

            # At the end of every epoch, increment the epoch counter and decay the
            # learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                print('(Epoch {} / {}) train acc: {:.2f}%    val_acc: {:.2f}%'.format(
                             self.epoch, self.num_epochs, train_acc * 100, val_acc * 100))

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params