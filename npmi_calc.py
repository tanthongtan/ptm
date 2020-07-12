# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import re
import math
import os
import itertools
import pickle
from statistics import mean

phrase_split_pattern = re.compile(r'-|_')

# corpus_vocab  # number of words
# inv_index  # word_id: [doc_id1, doc_id2...]
# corpus_size  # number of documents


def get_docs_from_index(w, inv_index):
    wdocs = set()
    if re.search(phrase_split_pattern, w):
        # this is to handle the phrases in NYT corpus, without which we will have 50% of the words considered OOV.
        wdocs = intersecting_docs(w, inv_index)
    elif w in inv_index:
        wdocs = inv_index[w]
    return wdocs


def intersecting_docs(phrase, inv_index):
    words = re.split(phrase_split_pattern, phrase)
    intersect_docs = set()
    for word in words:
        if not word in inv_index:
            # if any of the words in the phrase is not the corpus, the phrase also is not in the corpus
            return set()
        if not intersect_docs:
            intersect_docs.update(inv_index[word])
        else:
            intersect_docs.intersection_update(inv_index[word])
    return intersect_docs


def get_pmi(docs_1, docs_2, corpus_size):
    assert len(docs_1)
    assert len(docs_2)
    small, big = (docs_1, docs_2) if len(docs_1) < len(docs_2) else (docs_2, docs_1)
    intersect = small.intersection(big)
    pmi = 0.0
    npmi = 0.0
    if len(intersect):
        pmi = math.log(corpus_size) + math.log(len(intersect)) - math.log(len(docs_1)) - math.log((len(docs_2)))
        npmi = -1 * pmi / (math.log(len(intersect)) - math.log(corpus_size))

    return pmi, npmi


def get_idf(w, inv_index, corpus_size):
    n_docs = len(get_docs_from_index(w, inv_index))
    return math.log(corpus_size / (n_docs + 1.0))


def test_pmi(inv_index, corpus_size):
    word_pairs = [
        ["apple", "ipad"],
        ["monkey", "business"],
        ["white", "house"],
        ["republican", "democrat"],
        ["china", "usa"],
        ["president", "bush"],
        ["president", "george_bush"],
        ["president", "george-bush"]
    ]
    pmis = []
    for pair in word_pairs:
        w1docs = get_docs_from_index(pair[0], inv_index)
        w2docs = get_docs_from_index(pair[1], inv_index)
        assert len(w1docs)
        assert len(w2docs)
        pmi, _ = get_pmi(w1docs, w2docs, corpus_size)
        assert pmi > 0.0
        print("Testing PMI: w1: {}  w2: {}  pmi: {}".format(pair[0], pair[1], pmi))
        pmis.append(pmi)
    assert pmis[0] > pmis[1]  # pmi(apple, ipad) > pmi(monkey, business)



def get_topic_pmi(wlist, inv_index, corpus_size, max_words_per_topic):
    num_pairs = 0
    pmi = 0.0
    npmi = 0.0
    # compute topic coherence only for first 10 word in each topic.
    wlist = wlist[:max_words_per_topic]
    for (w1, w2) in itertools.combinations(wlist, 2):
        w1docs = get_docs_from_index(w1, inv_index)
        w2docs = get_docs_from_index(w2, inv_index)
        if len(w1docs) and len(w2docs):
            word_pair_pmi, word_pair_npmi = get_pmi(w1docs, w2docs, corpus_size)
            pmi += word_pair_pmi
            npmi += word_pair_npmi
            num_pairs += 1
    if num_pairs:
        pmi /= num_pairs
        npmi /= num_pairs
    return pmi, npmi, num_pairs


def calc_pmi_for_all_topics(topics, ref_corp):
    
    inv_index, corpus_size = pickle.load(open('iics_' + ref_corp + '.p','rb'))
    for i, (k,v) in enumerate(inv_index.items()):
        inv_index[k] = set(v)
    
    pmis = []
    npmis = []
    for wlist in topics:
        use_N_words = len(wlist)  # use full list
        pmi, npmi, _ = get_topic_pmi(wlist, inv_index, corpus_size, use_N_words)
        # print(npmi, pmi)
        # print(wlist)
        pmis.append(pmi)
        npmis.append(npmi)
        
    mean_pmi = mean(pmis)
    mean_npmi = mean(npmis)

    return pmis, mean_pmi, npmis, mean_npmi