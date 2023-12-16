import numpy as np
from nltk.metrics import jaccard_distance
import pandas as pd
# IMport partial
from functools import partial
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk import ngrams
brown_ic = wordnet_ic.ic('ic-brown.dat')
from nltk import download
import nltk

download('averaged_perceptron_tagger')
download('wordnet')
download('omw-1.4')
download('punkt')


import textdistance

def dice_sim(x, y):
    x = set(x)
    y = set(y)
    x_n_y = x.intersection(y)
    try:
        return 2*len(x_n_y) / (len(x) + len(y))
    except ZeroDivisionError:
        return 0

def overlap_sim(x, y):
    x = set(x)
    y = set(y)
    x_n_y = x.intersection(y)
    try:
        return len(x_n_y) / min(len(x), len(y))
    except ZeroDivisionError:
        return 0

def cosine_sim(x, y):
    x = set(x)
    y = set(y)
    x_n_y = x.intersection(y)

    if len(x_n_y) == 0:
        return 0
    
    try:
        return len(x_n_y) / np.sqrt(len(x)*len(y))
    except ZeroDivisionError:
        return 0

def jaccard_sim(x, y):
    '''
    Get the Jaccard similarity between two sets
    '''
    x = set(x)
    y = set(y)
    try:
        return 1 - jaccard_distance(x, y)
    except ZeroDivisionError:
        return 0


def get_similarities(X, d_sim=None):
    '''
    Get the similarities between two sentences
    '''
    df_sim = pd.DataFrame()
    if d_sim is None:
        d_sim = {"jaccard": jaccard_sim, "dice": dice_sim, 
                  "overlap": overlap_sim, "cosine": cosine_sim}
        
    for name, sim in d_sim.items():
        df_sim[name] = X.apply(lambda x: sim(x["sent1"], x["sent2"]), axis=1)
        
    return df_sim


def get_path_similarity(syn1, syn2):
    return syn1.path_similarity(syn2)

def get_lch_similarity(syn1, syn2):
    try:
        return syn1.lch_similarity(syn2)
    except:
        return None
    
def get_wup_similarity(syn1, syn2):
    try:
        return syn1.wup_similarity(syn2)
    except:
        return None
    
def get_lin_similarity(syn1, syn2):
    try:
        return syn1.lin_similarity(syn2, brown_ic)
    except:
        return None
    
def longest_common_substring(str1, str2):
    return textdistance.lcsstr.similarity(str1, str2)

def longest_common_subsequence(str1, str2):
    return textdistance.lcsseq.similarity(str1, str2)

def greedy_string_tiling(str1, str2):
    return textdistance.ratcliff_obershelp.similarity(str1, str2)

def jaro_similarity(str1, str2):
    return textdistance.jaro.similarity(str1, str2)  # Use jellyfish for Jaro similarity

def jaro_winkler_similarity(str1, str2):
    return textdistance.jaro_winkler.similarity(str1, str2)

def monge_elkan_similarity(str1, str2):
    return textdistance.monge_elkan.similarity(str1, str2)

def levenshtein_distance(str1, str2):
    return textdistance.levenshtein.similarity(str1, str2)


def get_textdistance_similarities(X, d_sim=None):
    '''
    Get the similarities between two sentences
    '''
    if d_sim is None:
        d_sim = {"lcstr": longest_common_substring, "lcseq": longest_common_subsequence, 
                  "gst": greedy_string_tiling, "jaro": jaro_similarity, 
                  "jaro_w": jaro_winkler_similarity, "monge": monge_elkan_similarity, 
                  "levenshtein": levenshtein_distance}

    df_sim = pd.DataFrame()
    for name, sim in d_sim.items():
        df_sim[name] = X.apply(lambda x: sim(x["sent1"], x["sent2"]), axis=1)

    return df_sim

def get_word_ngrams(text, n, tokenize=True):
    if tokenize:
        words = nltk.word_tokenize(text)
    else:
        words = text
    return list(ngrams(words, n))

def get_ngrams_similarity(x, y, sim, n=2, tokenize=True):
    ngrams1 = get_word_ngrams(x, n, tokenize=tokenize)
    ngrams2 = get_word_ngrams(y, n, tokenize=tokenize)
    return sim(ngrams1, ngrams2)

def get_ngrams_similarities(X, d_sim=None, nmax=5, tokenize=True):
    '''
    Get the similarities between two sentences
    '''
    if d_sim is None:
        d_sim = {"jaccard": jaccard_sim, "dice": dice_sim, 
                  "overlap": overlap_sim, "cosine": cosine_sim}

    df_sim = pd.DataFrame()
    for n in range(1, nmax+1):
        for name, sim in d_sim.items():
            df_sim[name + f"_n_{n}"] = X.apply(lambda x: get_ngrams_similarity(x["sent1"], x["sent2"], sim, n=n, tokenize=tokenize), axis=1)
        
    return df_sim

def calculate_idf(term, documents):
    document_frequency = sum(1 for document in documents if term in document)
    if document_frequency == 0:
        return 0.0
    else:
        return np.log(len(documents) / document_frequency)
    
def compare_synsets(ls_ls_syns1, ls_ls_syns2, documents=None, similarity=get_wup_similarity, idf=True):

    d_out = {}
    for w, ls_syns1 in ls_ls_syns1:
        d_out[w] = []
        for _, ls_syns2 in ls_ls_syns2:
            for syn1 in ls_syns1:
                for syn2 in ls_syns2:
                    out = similarity(syn1, syn2) if syn1.pos() == syn2.pos() else 0
                    out = 0 if out is None else out
                    d_out[w].append(out)
                    # print(syn1, syn2, out)
        
        if d_out[w] != []:
            if idf:
                d_out[w] = calculate_idf(w, documents) * np.max(d_out[w])
            else:
                d_out[w] = np.max(d_out[w])
        else:
            d_out[w] = 0

    return d_out

def get_sem_similarities(X, d_sim=None, idf=True, documents=None):
    '''
    Get the similarities between two sentences
    '''
    if d_sim is None:
        d_sim = {"wup": get_wup_similarity, "path": get_path_similarity, 
                  "lch": get_lch_similarity, "lin": get_lin_similarity}

    df_sim = pd.DataFrame()
    for name, sim in d_sim.items():
        if idf:
            sim_fw = X.apply(lambda x: sum(compare_synsets(x.iloc[0], x.iloc[1], documents, similarity=sim).values()), axis=1)
            sim_bw = X.apply(lambda x: sum(compare_synsets(x.iloc[1], x.iloc[0], documents, similarity=sim).values()), axis=1)
        else:
            sim_fw = X.apply(lambda x: sum(compare_synsets(x.iloc[0], x.iloc[1], idf=idf, similarity=sim).values()), axis=1)
            sim_bw = X.apply(lambda x: sum(compare_synsets(x.iloc[1], x.iloc[0], idf=idf, similarity=sim).values()), axis=1)
        
        df_sim[name] = (sim_fw + sim_bw) / 2

    return df_sim