import numpy as np
import cPickle
import gzip
import os
import sys
import time

import theano
import theano.tensor as T
import theano.printing
import theano.tensor.shared_randomstreams

from logistic_sgd import LogisticRegression
from load_data import load_umontreal_data, load_mnist


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None):

        self.input = input
        self.activation = activation

        if W is None:
            # Initialization from deeplearning.net tutorial.
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        self.params = [self.W, self.b]


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)

        self.output = _dropout_from_layer(rng, self.output, p=0.5)


class MLP(object):
    """A multilayer perceptron with all the trappings required to do dropout
    training.

    """
    def __init__(self,
            rng,
            input,
            layer_sizes):

        rectified_linear_activation = lambda x: T.maximum(0.0, x)

        # Set up all the hidden layers
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        next_layer_input = input
        # dropout the input with prob 0.2
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=0.2)
        for n_in, n_out in weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=next_dropout_layer_input,
                    activation=rectified_linear_activation,
                    n_in=n_in, n_out=n_out)
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the paramters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=rectified_linear_activation,
                    W=next_dropout_layer.W * 0.5,
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
        
        # Set up the output layer
        n_in, n_out = weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
                input=next_dropout_layer_input,
                n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        output_layer = LogisticRegression(
            input=next_layer_input,
            W=dropout_output_layer.W * 0.5,
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors

        # Grab all the parameters together.
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]


def test_mlp(learning_rate, n_epochs, batch_size, dropout, results_file_name,
        layer_sizes, dataset):
    """
    The dataset is the one from the mlp demo on deeplearning.net.  This training
    function is lifted from there almost exactly.

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    """
    datasets = load_mnist(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = np.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x,
           layer_sizes=layer_sizes)

    # Build the expresson for the cost function.
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)

    # Compile theano function for testing.
    #errors = classifier.dropout_errors(y) if dropout else classifier.errors(y)
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)

    # Compile theano function for validation.
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(validate_model, outfile="validate_file.png",
    #        var_with_name_simple=True)

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a dictionary
    updates = {}
    for param, gparam in zip(classifier.params, gparams):
        updates[param] = param - learning_rate * gparam

    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.
    output = dropout_cost if dropout else cost
    train_model = theano.function(inputs=[index], outputs=output,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(train_model, outfile="train_file.png",
    #        var_with_name_simple=True)

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    #patience = 10000  # look as this many examples regardless
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    results_file = open(results_file_name, 'wb')

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = epoch * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                total_validation_errors = np.sum(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.))

                results_file.write("{0}\n".format(total_validation_errors))
                results_file.flush()

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                    done_looping = True
                    break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    import sys

    learning_rate = 0.01
    n_epochs = 3000
    batch_size = 100
    layer_sizes = [ 28*28, 800, 800, 10 ]
    dataset = 'data/mnist_batches.npz'
    #dataset = 'data/mnist.pkl.gz'

    if len(sys.argv) < 2:
        print "Usage: {0} [dropout|backprop]".format(sys.argv[0])
        exit(1)

    elif sys.argv[1] == "dropout":
        dropout = True
        results_file_name = "results_dropout.txt"

    elif sys.argv[1] == "backprop":
        dropout = False
        results_file_name = "results_backprop.txt"

    else:
        print "I don't know how to '{0}'".format(sys.argv[1])
        exit(1)

    test_mlp(learning_rate=learning_rate,
             n_epochs=n_epochs,
             batch_size=batch_size,
             layer_sizes=layer_sizes,
             dropout=dropout,
             dataset=dataset,
             results_file_name=results_file_name)

