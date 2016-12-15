#!/usr/local/bin/python3

import argparse
import tokenize
import numpy as np

########################################################################

def load_data(data_path):
    return np.load(data_path)

def create_zipf_index(data):
    unique, counts = np.unique(data, return_counts=True)
    zipf_idxs = unique[np.argsort(counts)[::-1]]
    inv_zipf_idxs = np.argsort(zipf_idxs)
    return zipf_idxs, inv_zipf_idxs, counts

def zipf(data, zipf_idxs):
    return zipf_idxs[data]

def unzipf(zipfed, inv_zipf_idxs):
    return inv_zipf_idxs[zipfed]

def unknownize(zipfed, cutoff):
    return np.minimum(zipfed, cutoff-1)

########################################################################

def sample_oov_words(zipf_idxs, num_oov_words, cutoff_left, cutoff_right):
    return np.random.choice(
        zipf_idxs[cutoff_left:cutoff_right], 
        replace=False, size=num_oov_words)

def sample_cbow_batch(seqs, context_size):
    offset = int(context_size/2)
    ctxidxs = []
    tgtidxs = []
    for i in range(offset, seqs.shape[1]-offset):
        ctxidxs += [i-j for j in range(offset, 0, -1)] 
        ctxidxs += [i+j for j in range(1,offset+1)]
        tgtidxs += [i]
    seqidx = np.array(ctxidxs, np.int32)
    tgtidxs = np.array(tgtidxs, np.int32)
    return seqs[:,ctxidxs].reshape(-1, context_size), seqs[:,tgtidxs].reshape(-1)
    
def sample_seqs(seqs, num_sampled):
    return np.random.choice(np.arange(seqs.shape[0]), replace=False, size=num_sampled)

########################################################################

def to_sequences(data, len_seq=20):
    num_seq = len_seq*int(len(data)/len_seq)
    return data[:num_seq].reshape(-1, len_seq)

def containing_sequences(widxs, seqs, return_negation=False):
    match = np.in1d(seqs[:,0], widxs)
    for i in range(1, seqs.shape[1]):
        match = match | np.in1d(seqs[:,i], widxs)
    if return_negation:
        return np.where(match)[0], np.where(~match)[0] 
    else:
        return np.where(match)[0]
 
########################################################################

def lookup_word(widx, vocab_path):
    w = None
    fp = open(vocab_path)
    for i, line in enumerate(fp):
        if i == widx:
            w = line[:-1]
            break
    fp.close()
    return w

def lookup_index(w):
    idx = None
    fp = open(vocab_path)
    for i, line in enumerate(fp):
        if line[:-1] == w:
            idx = i
            break
    fp.close()
    return idx

def lookup_sequence(sidx, seqs, vocab_path):
    result = []
    for widx in seqs[sidx]:
        result.append(lookup_word(widx, vocab_path))
    return result

########################################################################

def sample_oov_seqs(seqs, zipf_idxs, num_oov_words, cutoff_left, cutoff_right):
    oov_widxs = sample_oov_words(zipf_idxs, num_oov_words, cutoff_left, cutoff_right)
    return oov_widxs, containing_sequences(oov_widxs, seqs, return_negation=True)

def random_split(a, ratio):
    split = int(a.shape[0] * ratio)
    idxs = np.random.permutation(np.arange(a.shape[0]))
    return a[idxs[:split]], a[idxs[split:]]
    
def make_train_test_split(oov_seqs, iv_seqs, train_ratio=0.8):
    iv_seqs_train, iv_seqs_test = random_split(iv_seqs, train_ratio)
    oov_seqs_train, oov_seqs_test = random_split(oov_seqs, train_ratio)
    return oov_seqs_train, oov_seqs_test, iv_seqs_train, iv_seqs_test

########################################################################

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="path to the numpy index")
    parser.add_argument("vocab_path", type=str, help="path to the vocabulary")

    args = parser.parse_args()
    
    index = load_data(args.data_path)
    

if __name__ == '__main__':
    main()