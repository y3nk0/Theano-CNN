import numpy as np
import _pickle as cPickle
from collections import defaultdict
import sys, re
import pandas as pd

def build_data_cv(data_file, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    y_data = []

    vocab = defaultdict(float)
    with open(data_file, "r",encoding='latin1') as f:
        for line in f:
            rev = []
            rev.append(line[2:].strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1

            y_data.append(int(line[0]))

            datum  = {"y":int(line[0]),
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)

    num_classes = len(set(y_data))

    return revs, vocab, num_classes

def build_data_split(data_folder, clean_string=True):
    """
    Loads data for pre-split datasets.
    """

    train_file = data_folder[0]
    test_file = data_folder[1]
    dev_file = data_folder[2]

    train_revs = []
    test_revs = []
    dev_revs = []

    y_data = []

    vocab = defaultdict(float)
    with open(train_file, "r", encoding='latin1') as f:
        for line in f:
            rev = []
            rev.append(line[2:].strip())
            if clean_string:
                if "stsa" in train_file:
                    orig_rev = clean_str_sst(" ".join(rev))
                elif "TREC" in train_file:
                    orig_rev = clean_str(" ".join(rev),True)
                else:
                    orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1

            y_data.append(int(line[0]))

            datum = {"y":int(line[0]),
                      "text": orig_rev,
                      "num_words": len(orig_rev.split())}
            train_revs.append(datum)

    if "TREC" not in dev_file:
        with open(dev_file, "r", encoding='latin1') as f:
            for line in f:
                rev = []
                rev.append(line[2:].strip())
                if clean_string:
                    if "stsa" in dev_file:
                        orig_rev = clean_str_sst(" ".join(rev))
                    else:
                        orig_rev = clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1

                y_data.append(int(line[0]))

                datum = {"y":int(line[0]),
                          "text": orig_rev,
                          "num_words": len(orig_rev.split())}
                dev_revs.append(datum)

    with open(test_file, "r", encoding='latin1') as f:
        for line in f:
            rev = []
            rev.append(line[2:].strip())
            if clean_string:
                if "stsa" in test_file:
                    orig_rev = clean_str_sst(" ".join(rev))
                elif "TREC" in test_file:
                    orig_rev = clean_str(" ".join(rev),True)
                else:
                    orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()

            y_data.append(int(line[0]))

            datum = {"y":int(line[0]),
                      "text": orig_rev,
                      "num_words": len(orig_rev.split())}
            test_revs.append(datum)

    num_classes = len(set(y_data))

    if len(dev_revs)>0:
        return train_revs, test_revs, dev_revs, vocab, num_classes
    return train_revs, test_revs, vocab, num_classes

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}

    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format(fname, binary=True)  # C binary format

    for word in vocab:
        if word in model.wv.vocab:
            word_vecs[word]=model[word]

    del model

    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
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
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

if __name__=="__main__":

    # dataset = "../data/rt-polarity.all"
    #dataset = "../data/custrev.all"
    #dataset = "../data/mpqa.all"
    #dataset = "../data/subj.all"

    # dataset = "../data/stsa.binary"
    # dataset = "../data/stsa.fine"
    dataset = "../data/TREC"

    dataset_split = "split"

    w2v_file = "../GoogleNews-vectors-negative300.bin"

    print("loading data...")
    if dataset_split=="cv":
        revs, vocab, num_classes = build_data_cv(dataset, cv=10, clean_string=True)
        max_l = np.max(pd.DataFrame(revs)["num_words"])
        print("number of sentences: " + str(len(revs)))
    else:
        data_folder = [dataset+".train",dataset+".test",dataset+".dev"]
        if 'TREC' in dataset:
            train_revs, test_revs, vocab, num_classes = build_data_split(data_folder, clean_string=True)
        else:
            train_revs, test_revs, dev_revs, vocab, num_classes = build_data_split(data_folder, clean_string=True)

        max_l = np.max(pd.DataFrame(train_revs)["num_words"])

    print("data loaded!")

    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    print("loading word2vec vectors...")
    w2v = load_bin_vec(w2v_file, vocab)
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)

    if dataset_split=="cv":
        cPickle.dump([revs, W, W2, word_idx_map, vocab, max_l, num_classes], open("mr.p", "wb"))
    else:
        if 'TREC' in dataset:
            cPickle.dump([train_revs, test_revs, W, W2, word_idx_map, vocab, max_l, num_classes], open("mr_split.p", "wb"))
        else:
            cPickle.dump([train_revs, dev_revs, test_revs, W, W2, word_idx_map, vocab, max_l, num_classes], open("mr_split.p", "wb"))

    print("dataset created!")
