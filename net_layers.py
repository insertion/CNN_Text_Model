# -*- coding: utf-8 -*-
import numpy,theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet   import conv
# from theano.tensor.nnet   import conv2d  新版接口并不好用，这个比上面那个慢了一倍的时间
from theano.tensor.shared_randomstreams import RandomStreams

rng = numpy.random.RandomState(1234)

def ReLU(x):
    y = T.maximum(0.01,x)
    # T.max(x，axis=None),return maximum of x along axis
    # T.maximum(a,b) return max by element of a and b
    return y
def Jump(x):
    y = T.where(T.gt(x,0),1,0)
    # Comparisons has special method
    # T.lt(a,b) ==> a < b less than
    # T.gt(a,b) ==> a > b greater than
    # T.le(a,b) ==> less and equal
    # switch alis for where
    # T.where(condition,x,y)
    # ==> condition?x:y
    return y

class Top_Layer(object):
    """
    logistic regression layer
    """
    def __init__(self,input,n_in,n_out,W=None,b=None):
        self.input = input
        W_value = numpy.zeros((n_in, n_out),dtype = theano.config.floatX)
        b_value = numpy.zeros((n_out),dtype = theano.config.floatX)
        if W is None:
            self.W = theano.shared(value = W_value,borrow =True)
            self.b = theano.shared(value = b_value,borrow =True)
        else:
            self.W = W
            self.b = b
        self.output = T.nnet.softmax(T.dot(input,self.W)+self.b)
        self.params = [self.W,self.b]
        self.entropy = T.nnet.categorical_crossentropy(self.output,self.output).mean()

    def negative_log_likelihood(self,y):
        """
        Return the mean of the negative log-likelihood of the prediction
        """
        y_pred = T.cast(T.argmax(self.output,axis = 1),'int32')
        likelihood = T.log(self.output)[T.arange(y.shape[0]),y]
        return -T.mean(likelihood)
    def errors(self,y):
        """
        Return the error rate of prediction
        """
        y_pred = T.cast(T.argmax(self.output,axis = 1),'int32')
        # check if y has same dimension of y_pred
        if y.ndim != y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type))
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(y_pred, y))
        else:
            raise NotImplementedError()
        
class Hidden_Layer(object):

    """
    Typical hidden layer of a MLP
    """
    def __init__(self,input,n_in,n_out,W=None,b=None,activation=T.tanh):
        # The definition of asarray is:
        #     def asarray(a, dtype=None, order=None):
        #         return array(a, dtype, copy=False, order=order)

        # So it is like array, except it has fewer options, and copy = False. 
        # while array has copy = True by default.
        # I think the main difference is that array (by default) will make a copy of the object, 
        # while asarray will not unless necessary.
        
        # numpy.ndarray : which first params is shape to create a class instance 
        W_value = numpy.asarray(
            rng.uniform(
                low   = -numpy.sqrt(6.0 / (n_in + n_out)),
                high  =  numpy.sqrt(6.0 / (n_in + n_out)),
                size  = (n_in,n_out)
            ),
            dtype = theano.config.floatX
        )
        b_value = numpy.zeros((n_out),dtype = theano.config.floatX)
        self.input = input
        if W is None:
            self.W = theano.shared(value = W_value,borrow =True)
            self.b = theano.shared(value = b_value,borrow =True)
        else:
            self.W = W
            self.b = b
        self.output = activation(T.dot(input,self.W)+self.b)
        self.params = [self.W,self.b]

class Conv_Pool_layer(object):
    
    """
    the composition of convolution and max-pooling layer
    """
    def __init__(self,input,input_shape,filter_shape,pool_shape,W=None,b=None,activation = T.tanh):
    
        # input_shape:                        filter_shape:  
        #              batch size,                          filter number,
        #              input maps,                          input  maps,
        #              input height,                        filter height, 
        #              input width                          filter width
        assert input_shape[1] == filter_shape[1]
        
        n_in  = numpy.prod(filter_shape[1:])
        n_out = numpy.prod(filter_shape[2:]) * filter_shape[0] / numpy.prod(pool_shape)
   
        W_value = numpy.asarray(
            rng.uniform(
                low   = -numpy.sqrt(6.0 / (n_in + n_out)),
                high  =  numpy.sqrt(6.0 / (n_in + n_out)),
                size  =  filter_shape
            ),
            dtype = theano.config.floatX
            # shared vaiable need to define the dtype
        )
        # These replicated units share the same parameterization (weight vector and bias)
        # so b's shape equal to the number of filters
        b_value = numpy.zeros(filter_shape[0],dtype = theano.config.floatX)
        
        if W is None:
            # if Theano is using a GPU device, then the borrow flag has no effect
            self.W = theano.shared(value = W_value,borrow =True)
            self.b = theano.shared(value = b_value,borrow =True)
        else:
            self.W = W
            self.b = b 

        conv_out = conv.conv2d(
                            input        = input,
                            filters      = self.W,
                            image_shape  = input_shape,
                            filter_shape = filter_shape,
                            subsample    = (1,1)
                          )
        # input shape a tuple of constant int values
        # return : 4Dtensor 
        #                    batch  size, 
        #                    output channels, 
        #                    output rows, 
        #                    output columns             
        pool_out=pool.pool_2d(
                                input         = conv_out,
                                ds            = pool_shape,
                                ignore_border = True
                              )
        # Downscale the input by a specified factor
        self.input  = input
        self.output = activation(pool_out + self.b.dimshuffle('x',0,'x','x'))
        self.params = [self.W,self.b]

                              
                
                            