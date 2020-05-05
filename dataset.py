import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
import torch.nn.functional as F
import torch

def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def sparse_to_numpy(x, vocab_size):
    out = np.zeros((len(x), vocab_size), dtype=int)
    for i, doc in enumerate(x):
        for id, count in doc.items():
            out[i][id] = count
    return out

def load_20news(use_tfidf = False, normalize = True, sublinear = False):
    train_set = pickle.load(open("data/20news/train_set.p", "rb"))
    test_set = pickle.load(open("data/20news/test_set.p", "rb"))
    vocab = pickle.load(open("data/20news/vocab.p", "rb"))
    vocab_size = len(vocab)    
    data_tr = sparse_to_numpy(train_set, vocab_size)
    data_te = sparse_to_numpy(test_set, vocab_size)
    
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

def load_20news_5k(use_tfidf = False, normalize = True, sublinear = False):
    train_set = pickle.load(open("data/20news5k/x_train.p", "rb"))
    test_set = pickle.load(open("data/20news5k/x_test.p", "rb"))
    vocab = pickle.load(open("data/20news5k/vocab.p", "rb"))
    vocab_size = len(vocab)    
    data_tr = train_set.toarray()
    data_te = test_set.toarray()
    
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

def load_20news_full(use_tfidf = False, normalize = True):
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    data_tr = np.load('data/20news_clean/train.txt.npy', encoding = 'bytes')
    data_te = np.load('data/20news_clean/test.txt.npy', encoding = 'bytes')
    vocab = pickle.load(open('data/20news_clean/vocab.pkl','rb'))
    vocab_size=len(vocab)
    np.load = np_load_old
    #--------------convert to one-hot representation------------------
    data_tr = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
    data_te = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_te if np.sum(doc)!=0])
    
    if use_tfidf == True:
        tfidf = TfidfTransformer()
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

def load_20news_diff():
    with open('data/20News-diff/diff.train.tfidf1.data') as file:
      for i, line in enumerate(file):
          tokens = line.split()
          if i == 0:
              vocab_size = int(tokens[0])
              num_tr = int(tokens[1])
              data_tr = np.zeros((num_tr, vocab_size))
          elif i == num_tr + 1:
              categories = float(tokens[0])
          else:
              for token in tokens[2:]:
                  temp = token.split(':')
                  idx = int(temp[0])
                  val = float(temp[1])
                  data_tr[i-1][idx] = val
                  
    with open('data/20News-diff/diff.test.tfidf1.data') as file:
      for i, line in enumerate(file):
          tokens = line.split()
          if i == 0:
              num_te = int(tokens[1])
              data_te = np.zeros((num_te, vocab_size))
          elif i == num_te + 1:
              pass
          else:
              for token in tokens[2:]:
                  temp = token.split(':')
                  idx = int(temp[0])
                  val = float(temp[1])
                  data_te[i-1][idx] = val
                
    vocab = {}
    with open('data/20News-diff/diff.voc') as file:
        for i, line in enumerate(file):
            vocab[line.strip()] = i
    
    tensor_tr = torch.tensor(data_tr).float()
    tensor_te = torch.tensor(data_te).float()
    
    return data_tr, data_te, tensor_tr, tensor_te, vocab, vocab_size, num_tr