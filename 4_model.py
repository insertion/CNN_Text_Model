# -*- coding: utf-8 -*-
from net_layers import *
from utils import get_idx_from_sent,make_idx_data_cv
import cPickle,time,sys
from collections import  OrderedDict
import theano.tensor.nlinalg
from theano.tensor.signal import pool
# manually set theano.config.device, theano.config.floatX
# heano.config.device = 'gpu'
# theano.config.floatX = 'float32'
theano_rng = RandomStreams(rng.randint(2 ** 30))
#  in Theano you first express everything symbolically and afterwards compile this expression to get functions
# theano_rng 和 rng 的区别是
# theano_rng 是编译进function的，每次训练都会运行，
# rng 只运行一次，不会被编译进function
def load_data(batch_size =50):
    print "loading data ..."
    sentences, vectors,rand_vectors, word_idx_map, _ = cPickle.load(open('dataset.pkl','rb'))
    if len(sys.argv) <= 1:
        print "usage: please select the mode between '-rand' and '-word2vec' "
        mode = '-word2vec'
    else:    
        mode= sys.argv[1]
    if mode == '-rand':
        print "using the rand vectors"
        U  = rand_vectors 
    else:
        print "using word2vec vectors"
        U  = vectors
    
    dataset,testset = make_idx_data_cv(sentences, word_idx_map, cv = 0)

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
    #print valid_y.type()
    return train_x,train_y,valid_x,valid_y,U,n_train_batches,n_valid_batches,input_h,input_w

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

def build_model(learning_rate,y,layer0_input,input_h,input_w,batch_size,filter_hs=[3,4,5]):
    """
    construct the model 
    return params and top_layer
    """
    layer0_input -= T.mean(layer0_input, axis = 0) # zero-center 可以减少模型抖动,81.5%
    input_maps      = 1
    filter_maps     = 100
    filter_w        = input_w

    filter_shapes   = []
    pool_sizes      = []

    for filter_h in filter_hs:
        shape = (
                    filter_maps,
                    input_maps,
                    filter_h,
                    filter_w
                )
        filter_shapes.append(shape)
        pool_size = (
                    input_h - filter_h + 1,
                    1  
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
                                        activation   = T.nnet.relu
                                        # 激活函数对模型的性能影响很大,relu比sigmod，softplus，tanh 好太多
                                        )                  
        layer0_output = conv_layer.output.flatten(2).dimshuffle(0,'x',1)
        # conlayer.output ==>(batch_size,maps,1,1) 
        # flatten   ==>(batch_size,maps)
        # dimshuffle ==>(batch_size,1,maps)
        conv_layers.append(conv_layer)
        layer0_outputs.append(layer0_output)

    # any number of ‘x’ characters in dimensions where this tensor should be broadcasted.  
    layer1_input = T.concatenate(layer0_outputs,1)
    # (batch_size,3,maps)
    layer1_input = pool.pool_2d(
                            input = layer1_input,
                            ds    = (3,1),
                            ignore_border = True
                         )
    # (batch_size,3/3,maps)
    layer1_input = layer1_input.flatten(2)
    # Max pooling will be done over the 2 last dimensions.
    # layer1_input = layer1_input.max(1)
    layer1_input -= T.mean(layer1_input, axis = 0)  # zero-center 可以减少模型抖动
    
    # var = T.var(layer1_input,axis = 0).mean() / layer1_input.shape[0]
    # cov = T.dot(T.transpose(layer1_input),layer1_input) / layer1_input.shape[0]
    # var = theano.tensor.nlinalg.trace(cov) 
    # cov = T.abs_(cov).sum()
    # U,S,V = theano.tensor.nlinalg.svd(cov)
    # layer1_input = T.dot(layer1_input,U)
    # 矩阵相乘，实际上就是空间映射
    # 这里每个mini-batch 的支持向量不同，PCA没有意义
    # 即不同数据的pca被映射到不同的维度空间，没有可比性
    # 对角线上分别是x和y的方差，非对角线上是协方差。
    # 协方差大于0表示x和y若有一个增，另一个也增；小于0表示一个增，一个减；
    # 协方差为0时，两者独立。
    # 协方差绝对值越大，两者对彼此的影响越大，反之越小
    # hidden_layer = Hidden_Layer(
    #                                 input = layer1_input,
    #                                 n_in  = 300,
    #                                 n_out = 7,
    #                                 activation = T.nnet.relu

    #                             )

    top_layer   =  Top_Layer(                                                              
                                input = layer1_input,                 
                                n_in  = 100,                                                   
                                n_out = 2                                                      
                            )  

    params =[]
    for conv_layer in conv_layers:
        params += conv_layer.params
    
    params =  top_layer.params + params


    
    # positive_num = T.sum(y)
    # positive = T.transpose(layer1_input) * y
    # positive = T.sum(T.transpose(positive),axis = 0) / positive_num

    # negative_num = T.sum(1-y)
    # negative = T.transpose(layer1_input) * (1-y)
    # negative = T.sum(T.transpose(negative),axis = 0) / negative_num

    # positive = T.nnet.softmax(positive)
    # negative = T.nnet.softmax(negative)

    # crossentropy0 = T.nnet.binary_crossentropy(positive, negative).sum()
    # crossentropy1 = T.nnet.binary_crossentropy(negative, positive).sum()

    # entropy0      = T.nnet.binary_crossentropy(positive,positive).sum()
    # entropy1      = T.nnet.binary_crossentropy(negative,negative).sum()

    # entropy       = (-entropy0 - entropy1) / layer1_input.shape[1]

    # 训练的过程是一个减小不确定度的过程，是一个熵减的过程，是一个逐步提取信息的过程，熵的大小表示未知的信息的度量
    entropy = T.nnet.categorical_crossentropy(T.nnet.softmax(layer1_input),T.nnet.softmax(layer1_input)).mean()
    c_cost = top_layer.negative_log_likelihood(y) #- top_layer.entropy #  + entropy * 0.1
    
    c_grads = T.grad(cost = c_cost,wrt = params)
    c_grad_updates = [ 
                        (param,param - learning_rate* grad) for param,grad in zip(params,c_grads) 
                   ]

    

    return c_grad_updates,top_layer,layer1_input

def train(batch_size =100,learning_rate = 0.1,epochs = 1000):
    
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

    c_grad_updates,classifier,hiddenLayer_outputs = build_model(
                                                learning_rate= learning_rate,
                                                y            = y,
                                                layer0_input = layer0_input,
                                                input_h      = input_h,
                                                input_w      = input_w,
                                                batch_size   = batch_size,
                                              )
    # hiddenLayer_output 100*300
    
    #############
    # fine_tune #
    #############
    #for hiddenLayer_output in hiddenLayer_outputs:
    
    # [1,300] row 和vector 有没有区别
    # vector: Return a Variable for a 1-dimensional ndarray
    # row   : Return a Variable for a 2-dimensional ndarray in which the number of rows is guaranteed to be 1.

    # cov = T.dot(T.transpose(hiddenLayer_outputs),hiddenLayer_outputs) /hiddenLayer_outputs.shape[0]
    # U,S,V = theano.tensor.nlinalg.svd(cov)
    # hiddenLayer_outputs = T.dot(hiddenLayer_outputs,U)
    
    # cov = T.dot(T.transpose(hiddenLayer_outputs),hiddenLayer_outputs) / hiddenLayer_outputs.shape[0]
    # cov = T.abs_(cov).mean()
    # 训练的过程是一个减小不确定度的过程，是一个熵减的过程
    hiddenLayer_outputs = T.nnet.softmax(hiddenLayer_outputs)
    entropy0 = T.nnet.categorical_crossentropy(hiddenLayer_outputs, hiddenLayer_outputs).mean()
    # positive_num = T.sum(y)
    # positive = T.transpose(hiddenLayer_outputs) * y
    # positive = T.sum(T.transpose(positive),axis = 0) / positive_num

    # negative_num = T.sum(1-y)
    # negative = T.transpose(hiddenLayer_outputs) * (1-y)
    # negative = T.sum(T.transpose(negative),axis = 0) / negative_num

    # bais     = T.abs_(positive - negative)
    # bais     = T.nnet.softmax(bais)
    # positive = T.nnet.softmax(positive)
    # negative = T.nnet.softmax(negative)

    # mycrossentry  = -T.sum(positive*T.log(negative) + (1-positive) * T.log(1-negative))
    # crossentropy0 = T.nnet.binary_crossentropy(positive, negative).sum()
    # crossentropy1 = T.nnet.binary_crossentropy(negative, positive).sum()

    # entropy0      = T.nnet.binary_crossentropy(positive,positive).sum()
    # entropy1      = T.nnet.binary_crossentropy(negative,negative).sum()
    # entropy       = T.nnet.binary_crossentropy(bais,bais).sum()
    # kl            = crossentropy0 + crossentropy1 - entropy0 - entropy1
    # 熵表示数据分布的不确定性，熵越小表示分布越不均匀，熵也可以表示编码长度，熵越小表示编码长度越小，所载的信息越少
    # 熵也可以描述各个神经元的差异性
    # 交叉熵表示，用一种分布来表示另一种分布所需要的编码长度的期望
    # 相对熵，真实编码长度 - 非真实编码长度

    # // 结果取整 ==> %
    # / 正常除法
    # 按理说这里应该不能broadcast的,必须具有相同的列数
    # Input dimension mis-match. (input[0].shape[1] = 300, input[1].shape[1] = 100)
    # 只有作为结果输出时才会报错，应该是再计算节点中时才会报错
    # 如何不在计算节点中出现，只会检查语法错误，不会报运行时错误
    
    fine_tune_model = theano.function(
                        [index], 
                        [classifier.negative_log_likelihood(y),classifier.errors(y),entropy0,classifier.entropy],
                        updates = c_grad_updates,
                        givens={
                                x: train_x[index * batch_size : (index + 1) * batch_size],
                                y: train_y[index * batch_size : (index + 1) * batch_size]
                                },
                        allow_input_downcast = True,
                        on_unused_input='warn'
                  )

    val_model = theano.function(
        [index], 
        [classifier.errors(y),entropy0,classifier.entropy],
        givens={
            x: valid_x[index * batch_size: (index + 1) * batch_size],
            y: valid_y[index * batch_size: (index + 1) * batch_size]
            },
        allow_input_downcast=True,
        on_unused_input='warn'
        )
    
    debug_model = theano.function(
        [index], 
        [],
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size]
            },
        allow_input_downcast=True,
        on_unused_input='warn'
        )



    DEBUG = False
    
    if DEBUG == True:

        ############
        #  DEBUG  #
        ############
        for i in range(1):
            outs= debug_model(2)
            for out in outs:
                print out
    else:

        ############
        #Training #
        ############
    
        print 'fine_tune...'
        f = open('log/autocode.txt','wb') 
        epoch = 0
        while epoch < epochs :
            start_time = time.time()
            epoch = epoch + 1
            train_losses = []
            train_cost = []
            kls =[]
            kl2s = []
            vkls = []
            vkl2s =[]
            val_losses =[]
            # random shuffle (洗牌) the sample ，SGD
            for minibatch_index in rng.permutation(range(n_train_batches)):
                cost_index,train_errors,kl,kl2 = fine_tune_model( minibatch_index) 
                train_losses.append(train_errors) 
                train_cost.append(cost_index)
                kls.append(kl)
                kl2s.append(kl2)
            for i in xrange(n_valid_batches):
                val_loss,vkl,vkl2= val_model(i)
                val_losses.append(val_loss)
                vkls.append(vkl)
                vkl2s.append(vkl2)
            
            val_perf   = 1- numpy.mean(val_losses)
            train_perf = 1- numpy.mean(train_losses)
            cost_index = numpy.mean(train_cost)
            kl         = numpy.mean(kls)
            vkl        = numpy.mean(vkls)
            kl2        = numpy.mean(kl2s)
            vkl2       = numpy.mean(vkl2s)
            
            
            f.write('%3i,%.2f,%.2f,%.2f\n' % (epoch,val_perf*100, train_perf*100, cost_index*100)) 
            print ('epoch: %3i, time: %.2f, valperf: %.2f%%, trainperf: %.2f%%, cost: %.2f%%, kl: %f, vkl:%f, kl2: %f, vkl2:%f' 
                    % (epoch, time.time()-start_time, val_perf*100, train_perf*100, cost_index*100,kl,vkl,kl2,vkl2))
            if cost_index < 0.01 and epoch >25:
                break 
        f.close()
            
   

if __name__ == "__main__":
    train()
