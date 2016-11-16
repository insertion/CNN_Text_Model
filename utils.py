# -*- coding: utf-8 -*-
'''
    some utils for process data
'''
import numpy as np
import re
from   collections import defaultdict

def clean_en_str(string):
    # 正则化处理字符串,
    # 把每个string转化成小写字符
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    data_folder ：文件的名字
    cv          : 分成几份
    """
   
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    
    revs = []
    vocab = defaultdict(float)
    # 所有的key的value 都是float类型的，就算不存在的key,被初始化为0，这样防止了访问不存在key时出现的error提示
    # 存储每个word出现的次数
    with open(pos_file, "rb") as f:
        for line in f:       
            sentence = line[:].strip()
            if clean_string:
                orig_rev = clean_en_str(sentence)
            else:
                orig_rev = sentence.lower()
            words = set(orig_rev.split())
            # split默认参数是空格
            for word in words:
                vocab[word] += 1
            datum  = {
                      "y"         : 1, # 1 代表是pistive的词
                      "text"      : orig_rev,                             
                      "num_words" : len(orig_rev.split()),
                      "split"     : np.random.randint(0,cv) # 随机划分成十份中的一份
                      }
            revs.append(datum)
    
    with open(neg_file, "rb") as f:
        for line in f:       
            sentence = line[:].strip()
            if clean_string:
                orig_rev = clean_en_str(sentence)
            else:
                orig_rev = sentence.lower()
            words = set(orig_rev.split())
            # split默认参数是空格
            for word in words:
                vocab[word] += 1
            datum  = {
                      "y"         : 0, # 0 代表是pistive的词
                      "text"      : orig_rev,                             
                      "num_words" : len(orig_rev.split()),
                      "split"     : np.random.randint(0,cv) # 随机划分成十份中的一份
                      }
            revs.append(datum)
    # 把每个句子和词汇表返回        
    return revs, vocab

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}

    with open(fname, "rb") as f:
        header = f.readline()
        # 第一行 里面保存的是词典的大小 和 词向量的维数 
        # '3000000 300'
        vocab_size, vector_size = map(int, header.split())
        # map是什么函数
        # map(function, sequence) ：对sequence中的item依次执行function(item)，见执行结果组成一个List返回
        # map()函数接收两个参数，一个是函数，一个是序列，map将传入的函数依次作用到序列的每个元素，并把结果作为新的list返回。
        '''
        map(f, iterable)
        基本上等于：
                    [f(x) for x in iterable]
        '''
        binary_len = np.dtype('float32').itemsize * vector_size
        # In [31]: np.dtype('float32').itemsize
        # Out[31]: 4
        for line in xrange(vocab_size):
            # xrange 用法与 range 完全相同，所不同的是生成的不是一个list对象，而是一个生成器。
            # 要生成很大的数字序列的时候，用xrange会比range性能优很多，因为不需要一上来就开辟一块很大的内存空间
            # xrange则不会直接生成一个list，而是每次调用返回其中的一个值
            word = []
            while True:
                # 实际上，每行由 word 和 其对应的词向量组成 以空格分割
                # 但是因为以二进制方式保存，不能readline，除了第一行
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    # 把字符列表组成字符
                    break
                if ch != '\n':
                    word.append(ch)      
            if word in vocab:
                # 如果是词汇表中的词，则把它的词向量保存下来
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                # np.fromstring : A new 1-D array initialized from raw binary or text data in a string
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
            # 返回一个 size = 300 的 array 

def get_w_idx(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    把词向量字典转化成，一个 向量矩阵 和 一个索引map
    这样只用索引就可以表示sentence,节约了内存
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    # 一个字典，里面保存了每个词的词向量在 W 中的index
    W = np.zeros(shape=(vocab_size, k), dtype='float32')
    # 把所有的词向量放在一起形成一个矩阵,每行是一个词向量            
    i = 0
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def get_idx_from_sent(sent, word_idx_map, input_h=56,pad_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x   = []
    pad = pad_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    # 对于句子中的每个词
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
            # x 表示句子中词的index list
    while len(x) < input_h + 2 * pad:
        x.append(0)
        #在句子最后pad zero
    return x
def make_idx_data_cv(sentences, word_idx_map, cv, max_l=56, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in sentences:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, filter_h) 
        sent.append(rev["y"])
        if rev["split"] == cv:      
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.asarray(train,dtype="int")
    test  = np.asarray(test,dtype="int")
    return [train, test]  



""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""




def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(
                        X, 
                        img_shape, 
                        tile_shape, 
                        tile_spacing = (0, 0),
                        scale_rows_to_unit_interval = True,
                        output_pixel_vals           = True
                        ):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :param X: a 2-D array in which every row is a flattened image.

    :param img_shape: the original shape of each image

    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats 像数

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array
   