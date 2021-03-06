# -*- coding: utf-8 -*-
from net_layers import *
from utils import get_idx_from_sent,make_idx_data_cv
import cPickle,time,sys
from collections import  OrderedDict

theano_rng = RandomStreams(rng.randint(2 ** 30))
#  in Theano you first express everything symbolically and afterwards compile this expression to get functions
# theano_rng 和 rng 的区别是
# theano_rng 是编译进function的，每次训练都会运行，
# rng 只运行一次，不会被编译进function
def dropout(input,dropout_rate):
    corrupted_matrix = theano_rng.binomial(
                                size  = input.shape,
                                n     = 1,
                                p     = 1-dropout_rate,
                                dtype = theano.config.floatX
                           )
    return corrupted_matrix*input
def as_floatX(variable):
    if isinstance(variable, float):
        return numpy.cast[theano.config.floatX](variable)

    if isinstance(variable, numpy.ndarray):
        return numpy.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)    
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates         = OrderedDict({})
    exp_sqr_grads   = OrderedDict({})
    exp_sqr_ups     = OrderedDict({})
    gparams         = []
    for param in params:
        empty                = numpy.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value  =   as_floatX(empty),name   ="exp_grad_%s" % param.name)
        gp                   = T.grad(cost, param)
        exp_sqr_ups[param]   = theano.shared(value  =   as_floatX(empty),name   ="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg      = exp_sqr_grads[param]
        exp_su      = exp_sqr_ups[param]
        up_exp_sg   = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg]  = up_exp_sg
        step             =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su]  = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param    = param + step
        
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms       = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms   = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale           = desired_norms / (1e-7 + col_norms)
            updates[param]  = stepped_param * scale
        else:
            updates[param]  = stepped_param      
    return updates 
def build_model(layer0_input,input_h,input_w,batch_size,filter_hs=[3,4,5]):
    """
    construct the model 
    return params and top_layer
    """
    input_maps    = 1
    filter_maps   = int(sys.argv[1])
    top_n_in      = int(sys.argv[2])
    #filter_w      = input_w

    filter_shapes = []
    pool_sizes    = []
    layer1_n_in   = 0

    for filter_h in filter_hs:
        filter_w   = input_w #filter_h*3
        layer1_n_in += filter_maps*((input_w - filter_w)/filter_w + 1)
        #   (W−F+2P)/S+1
        shape = (
                    filter_maps,
                    input_maps,
                    filter_h,
                    filter_w 
                )
        filter_shapes.append(shape)
        pool_size = (
                    input_h - filter_h + 1,
                    1 # filter_w 
                    )
        pool_sizes.append(pool_size)

    conv_layers = []
    layer0_outputs = []

    for i in xrange(len(filter_hs)):
        conv_layer = Conv_Pool_layer(
                                        input        = layer0_input,
                                        input_shape  = (batch_size,input_maps,input_h,input_w),
                                        filter_shape = filter_shapes[i],
                                        pool_shape   = pool_sizes[i],
                                        activation   = ReLU
                                        )                  
        layer0_output = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer0_outputs.append(layer0_output)

    layer1_input = T.concatenate(layer0_outputs,1)
    #### train ###################################################################################
    Drop_layer1_input = dropout(layer1_input,0.5)                                               ##
    Drop_hidden_layer =  Hidden_Layer(                                                          ##
                                    input      = Drop_layer1_input,                             ##
                                    n_in       = layer1_n_in,                                   ##
                                    n_out      = top_n_in,                                            ##
                                    activation = ReLU                                           ##
                                )                                                               ##
                                                                                                ##
    Drop_top_layer   =  Top_Layer(                                                              ##
                                input = Drop_hidden_layer.output,                               ##
                                n_in  = top_n_in,                                                     ##
                                n_out = 2                                                       ##
                            )                                                                   ##
                                                                                                ##
    #### validation ##############################################################################
    #
    # reuse paramters in the dropout output.scale the weight matrix W with (1-p)
    hidden_layer        =  Hidden_Layer(
                                    input      = layer1_input,
                                    n_in       = layer1_n_in,
                                    n_out      = top_n_in,
                                    W          = Drop_hidden_layer.W * 0.5,
                                    b          = Drop_hidden_layer.b,
                                    activation = ReLU
                                )  
    top_layer   =  Top_Layer(
                                input = hidden_layer.output,
                                n_in  = top_n_in,
                                n_out = 2,
                                W     = Drop_top_layer.W * 0.5,
                                b     = Drop_top_layer.b
                            )  


    params =  Drop_top_layer.params + Drop_hidden_layer.params
    for conv_layer in conv_layers:
        params += conv_layer.params

    return Drop_top_layer,top_layer,params

def load_data(batch_size =50):
    print "loading data ..."
    sentences, vectors,rand_vectors, word_idx_map, _ = cPickle.load(open('dataset.pkl','rb'))
    # if len(sys.argv) <= 1:
    #     print "usage: please select the mode between '-rand' and '-word2vec' "
    #     mode = '-word2vec'
    # else:    
    #     mode= sys.argv[1]
    # if mode == '-rand':
    #     print "using the rand vectors"
    #     U  = rand_vectors 
    # else:
    #     print "using word2vec vectors"
    U  = vectors
    
    dataset,testset = make_idx_data_cv(sentences, word_idx_map, cv = 0)
    # dataset = []
    # for sentence in sentences:
    #     sent_Byindex = get_idx_from_sent(sentence["text"],word_idx_map)
    #     # get_idx_from_sent 还有三个参数，input_h input_w
    #     sent_Byindex.append(sentence['y'])
    #     dataset.append(sent_Byindex)
    
    # dataset = numpy.asarray(dataset,dtype ='int32')

    if dataset.shape[0] % batch_size > 0:
        add_data_num = batch_size  - dataset.shape[0] % batch_size
        extra_data   = rng.permutation(dataset)[:add_data_num]
        dataset      = numpy.append(dataset,extra_data,axis = 0) 
    
    n_batches       = dataset.shape[0] / batch_size
    n_train_batches = int(numpy.round(n_batches*0.9))
    n_valid_batches = n_batches - n_train_batches

    dataset         = rng.permutation(dataset)
    train_set       = dataset[:n_train_batches * batch_size,:]
    valid_set       = dataset[n_train_batches * batch_size:,:]
    
    input_h = train_set.shape[1] -1
    input_w = U.shape[1]

    train_x = T.cast(theano.shared(train_set[:,:-1],borrow = True),dtype="int32")
    train_y = T.cast(theano.shared(train_set[:,-1] ,borrow = True),dtype="int32")
    valid_x = T.cast(theano.shared(valid_set[:,:-1],borrow = True),dtype="int32")
    valid_y = T.cast(theano.shared(valid_set[:,-1] ,borrow = True),dtype="int32")

    return train_x,train_y,valid_x,valid_y,U,n_train_batches,n_valid_batches,input_h,input_w

def train(batch_size =50,learning_rate = 0.1,epochs = 100):
    
    (
        train_x,
        train_y,
        valid_x,
        valid_y,
        U,
        n_train_batches,
        n_valid_batches,
        input_h,
        input_w 
    ) = load_data(batch_size)
    
    index    = T.lscalar('index_real_name')
    x     = T.imatrix('data')
    y     = T.ivector('label')
    # x,y  type is int32
    Words = theano.shared(value = U, name = "Words")
    layer0_input = Words[x.flatten()].reshape((batch_size,1,input_h,input_w)) 
    # x.shape[0]              ==> batch_size
    # input_maps = 1,         ==> input_maps
    # x.shape[1]              ==> input_h 
    # words.shape[1]          ==> input_w
    classifier,validator,params = build_model(
                                        layer0_input = layer0_input,
                                        input_h      = input_h,
                                        input_w      = input_w,
                                        batch_size   = batch_size,
                                    )
    cost = classifier.negative_log_likelihood(y) 
    grads = T.grad(cost = cost,wrt = params )
    grad_updates = [ 
                        (param,param - learning_rate * grad) for param,grad in zip(params,grads) 
                   ]
    # grad_updates = sgd_updates_adadelta(cost = cost,params = params)
    train_model = theano.function(
        [index], 
        [cost,classifier.errors(y)],
        updates = grad_updates,
        givens={
                x: train_x[index * batch_size : (index + 1) * batch_size],
                y: train_y[index * batch_size : (index + 1) * batch_size]
                },
        allow_input_downcast = True,
        on_unused_input='warn'
        )

    val_model = theano.function(
        [index], 
        validator.errors(y),
        givens={
            x: valid_x[index * batch_size: (index + 1) * batch_size],
            y: valid_y[index * batch_size: (index + 1) * batch_size]
            },
        allow_input_downcast=True,
        on_unused_input='warn'
        )

    #############
    # Training #
    #############
    print '...training'
    epoch = 0
    while epoch < epochs :
        start_time = time.time()
        epoch = epoch + 1
        train_losses = []
        # random shuffle (洗牌) the sample 
        for minibatch_index in rng.permutation(range(n_train_batches)):
            cost_index,train_errors = train_model(index_real_name = minibatch_index) 
            train_losses.append(train_errors) 
        val_losses = [val_model(i) for i in xrange(n_valid_batches)]
        
        val_perf   = 1- numpy.mean(val_losses)
        train_perf = 1- numpy.mean(train_losses)   
        print 'epoch: %2i, training time: %.2f secs, val perf: %.2f %%, train perf: %.2f %%, cost: %.2f %%' % (epoch, time.time()-start_time, val_perf*100, train_perf*100, cost_index*100) 


if __name__ == "__main__":
    train()