import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
import torch
import sklearn.preprocessing as P
import scipy.sparse
import torch.nn.functional as F

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


def get_block_diag_data_batches_all_chains(tensor_tr, S, M):
    device = tensor_tr.device
    num_tr, vocab_size = tensor_tr.shape
    torch_indices = torch.stack([torch.randperm(num_tr, device = device)[:S] for x in range(M)])
    S = torch_indices.shape[-1]
    indices = torch_indices.reshape(-1)
    full_crow = tensor_tr.crow_indices()
    batch_starts = full_crow[indices]
    batch_counts = full_crow[indices + 1] - batch_starts
    batch_crow = F.pad(batch_counts,(1,0)).cumsum(dim=0)
    batch_total_nnz = batch_crow[-1].item()
    batch_original_indices = (torch.repeat_interleave(batch_starts - batch_crow[:-1], batch_counts, output_size=batch_total_nnz)
                              + torch.arange(batch_total_nnz, device = device))
    batch_col_indices = tensor_tr.col_indices()[batch_original_indices]
    batch_values = tensor_tr.values()[batch_original_indices]
    # extraction done, now make it block diagonal
    batch_col_indices_offsets = (torch.repeat_interleave(torch.arange(M, device = device)*vocab_size, S, output_size=M*S)
                                 .repeat_interleave(batch_counts, output_size=batch_total_nnz))
    batch_col_indices += batch_col_indices_offsets
    
    return (torch.sparse_csr_tensor(crow_indices=batch_crow, col_indices=batch_col_indices, 
                                   values=batch_values, size=(M*S, M*vocab_size)), 
            torch_indices)