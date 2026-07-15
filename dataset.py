import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
import torch
import sklearn.preprocessing as P
import scipy.sparse

def csr_to_torchsparse(x, gpu = False):
    assert scipy.sparse.isspmatrix_csr(x), "x must be a SciPy CSR matrix"
    crow_indices = torch.LongTensor(x.indptr) 
    col_indices = torch.LongTensor(x.indices)
    values = torch.DoubleTensor(x.data)
    size = torch.Size(x.shape)  
    ret = torch.sparse_csr_tensor(crow_indices=crow_indices, col_indices=col_indices, values=values, size=size)
    if gpu:
        ret = ret.cuda()
    return ret

def load_data(dataset, use_tfidf, normalize, sublinear = False):
    data_tr = scipy.sparse.load_npz("data/x_train_"+dataset+".npz")
    data_te = scipy.sparse.load_npz("data/x_test_"+dataset+".npz")
    vocab = pickle.load(open("data/vocab_"+dataset+".p", "rb"))
    vocab_size = len(vocab)    
    data_tr=data_tr[data_tr.getnnz(1) > 0]
    data_te=data_te[data_te.getnnz(1) > 0]
    
    if use_tfidf == True:
        tfidf = TfidfTransformer(sublinear_tf=sublinear)
        data_tr = tfidf.fit_transform(data_tr)
        data_te = tfidf.transform(data_te)
        
    elif normalize == True:
        data_tr = P.normalize(data_tr)
        data_te = P.normalize(data_te)
    
    num_tr = data_tr.shape[0]
    #--------------print the data dimentions--------------------------
    print('Dim Training Data',data_tr.shape)
    print('Dim Test Data',data_te.shape)
    
    return (data_tr, data_te, vocab, vocab_size, num_tr)