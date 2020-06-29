import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
import torch.nn.functional as F
import torch

def load_data(use_tfidf, normalize, sublinear = False, dataset):
    train_set = pickle.load(open("data/x_train_"+dataset+".p", "rb"))
    test_set = pickle.load(open("data/x_test_"+dataset+".p", "rb"))
    vocab = pickle.load(open("data/vocab_"+dataset+".p", "rb"))
    vocab_size = len(vocab)    
    data_tr = train_set.toarray()
    data_tr=data_tr[data_tr.sum(axis=-1) > 0]
    data_te = test_set.toarray()
    data_te=data_te[data_te.sum(axis=-1) > 0]
    
    if use_tfidf == True:
        tfidf = TfidfTransformer(sublinear_tf=sublinear)
        data_tr = np.array(tfidf.fit_transform(data_tr).todense())
        data_te = np.array(tfidf.transform(data_te).todense())
    
    num_tr = data_tr.shape[0]
    #--------------print the data dimentions--------------------------
    print('Dim Training Data',data_tr.shape)
    print('Dim Test Data',data_te.shape)
    #--------------make tensor datasets-------------------------------
    tensor_tr = torch.tensor(data_tr).float()
    tensor_te = torch.tensor(data_te).float()
    if normalize == True:
        tensor_tr = F.normalize(tensor_tr)
        tensor_te = F.normalize(tensor_te)
    
    return (data_tr, data_te, tensor_tr, tensor_te, vocab, vocab_size, num_tr)