"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np
import math

from .linear_classifier import LinearClassifier


def sftmax (x):
    """
    input: x- vector like array , input to the mathematical softmax function
    output: output of softmax funtion, array shaped as input
    function is implemented with numerical stability in mind
    """

    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)
    

def cross_entropy_loss_naive(W, X, y, reg):

    return cross_entropy_loss_vectorized(W, X, y, reg)

def compute_reg_term(W,reg):
    """
    
    """
    frob_norm = np.linalg.norm(W)
    return reg/(2)*frob_norm*frob_norm


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength


    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    #compute
    for i, sample in enumerate(X):
        S = sftmax(sample@ W)
        assert (abs(np.sum(S)-1) <= 0.0001), "sum of sftmax output not 1! Sum is " + str(np.sum(S))
        assert (S[y[i]]<=1), "sftmax member bigger than 1!"
        #print (S)
        loss = loss - math.log(S[y[i]])
        S[y[i]]=S[y[i]]-1. #kroneckerdelta, see derivation https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        dW=dW+sample.reshape(-1,1)@ S.reshape(1,-1)
    
    #add frobenius norm regularization
    #print(compute_reg_term(W,reg))
    loss = loss + compute_reg_term(W,reg)
    #print(dW)
    dW = dW + (reg)*W # see matrix cookbook this is the derivative of frobenius norm
    #print(dW)
    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
