# -*- coding: utf-8 -*-
import utils,sys,cPickle

if __name__ == '__main__' :

    data_folder = ["data/rt-polarity.pos","data/rt-polarity.neg"]

    print "loading data ..."
    sentences,vocab = utils.build_data_cv(data_folder,cv=10,clean_string = True)
    print "data loaded!"
    print "number of sentences: " + str(len(sentences))
    print "vocab size: "          + str(len(vocab))

    print "\nloading word2vec vectors..."
    if len(sys.argv) <= 1:
        print "error : please input path of word2vec bin file"
        sys.exit()
    w2v_file = sys.argv[1]  
    w2v = utils.load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))

    utils.add_unknown_words(w2v, vocab)
    vectors,word_idx_map = utils.get_w_idx(w2v)
    #print word_idx_map['good'] # 12004
    print "generate rand vectors"
    rand_vecs = {}
    utils.add_unknown_words(rand_vecs,vocab)
    rand_vectors,word_idx_map2 = utils.get_w_idx(rand_vecs)
    #print word_idx_map2['good'] # 4911
    # ranvec 产生的map和word2vec的map并不相同，所以不能用word_idx_map2 覆盖 word_idx_map
    cPickle.dump([sentences, vectors,rand_vectors, word_idx_map, vocab], open("dataset.pkl", "wb"))
    print "dataset created!"