import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _get_contexts(data, oov_widxs, zipf_idxs, context_size, vocabulary_size):
    n = int(context_size / 2)
    idxs = np.array([i for i in range(2*n+1) if i != n])
    sorted_oov = np.sort(oov_widxs)
    for i in range(0, data.shape[0]-idxs.shape[0]):
        h = hash(tuple(data[idxs+i]))
        w = data[i+n]
        pos = np.searchsorted(sorted_oov, zipf_idxs[w])
        if zipf_idxs[w] < vocabulary_size:
            if pos < len(sorted_oov) and zipf_idxs[w] != sorted_oov[pos]:
                yield (h, w, False)
            else:
                yield (h, w, True)
        
        if i % 100000 == 0:
            print("\rprocessed {0}/{1}".format(i, len(data)), end="")

def _get_contexts_dataframe(data, oov_widxs, zipf_idxs, context_size, vocabulary_size):
    df = pd.DataFrame(_get_contexts(data, oov_widxs, zipf_idxs, context_size, vocabulary_size))
    df.columns=["context", "word", "oov"]
    df = df.set_index(df.context)
    return df           
            
def _get_oov_iv_merge(df):
    return pd.merge(df[ df.oov ], df[ ~df.oov ], on="context")

def _find_closest(w, counts, merged, df, V, smoothing, lamb):
    sel = df[ df.word == w ]

    contexts = np.sort(sel.context.unique())
    candidates = np.sort(merged[ merged.word_x == w ].word_y.unique())
    
    if len(candidates) == 0:
        return False, np.random.randint(V)
    
    counts_w = sel.groupby(["context"]).context.count().values
    
    ps_w = (counts_w + lamb) / (counts[w] + smoothing)
    
    contexts_inv = { ctx: i for i, ctx in enumerate(contexts) }
    candidates_inv = { ctx: i for i, ctx in enumerate(candidates) }
    
    test = df[ (df.context.isin(contexts)) & (df.word.isin(candidates)) ]
    
    iv_contexts = test.context.unique()
    groups = test.groupby(["word", "context"])
    
    counts_vs = np.zeros((len(contexts), len(candidates)))
    for i in range(len(groups.grouper.result_index.labels[0])):
        v_label = groups.grouper.result_index.labels[0][i]
        ctx_label = groups.grouper.result_index.labels[1][i]
        v = groups.grouper.result_index.levels[0][v_label]
        ctx = groups.grouper.result_index.levels[1][ctx_label]
        j = candidates_inv[v]
        k = contexts_inv[ctx]
        counts_vs[k,j] += 1 
    
    ps_vs = (counts_vs + lamb) / ( counts[candidates] + smoothing)
    
    kl_loss = (ps_w * np.log(ps_w / ps_vs.T)).T
    
    v_min = np.argmin(kl_loss.sum(axis=0))
    return True, candidates[v_min]

class ContextSimilarityIndex:
    
    def __init__(self, data, counts, oov_widxs, zipf_idxs, context_size, vocab_size, lamb=1e-10):
        self.vocab_size_ = vocab_size 
        self.lamb_ = lamb
        self.smoothing_ = ( np.power(lamb, 1/(context_size)) * vocab_size)**(context_size)
        
        self.counts_ = counts
        self.df_ = _get_contexts_dataframe(data, oov_widxs, zipf_idxs ,context_size, vocab_size)
        self.merge_ = _get_oov_iv_merge(self.df_)
    
    def find_closest(self, widx):
        return _find_closest(widx, self.counts_, 
                             self.merge_, 
                             self.df_, 
                             self.vocab_size_, 
                             self.smoothing_,
                             self.lamb_)
