# -*- coding: utf-8 -*-
import codecs
import json
import numpy as np
import random

def train_test_data_load(fp, args):
    if 'customize' not in fp:
        with open(fp+'/paths.json','r') as f:
            paths = json.load(f)
        with open(fp+'/paths_index_direct.json','r') as f:
            paths_index_direct = json.load(f)
    else:
        with open(fp+'/path_customized.json','r') as f:
            paths = json.load(f)
        paths_index_direct = []

    w2v = np.loadtxt(fp+'/w2v_bert.txt')

    with open(fp+'/training.json','r') as f:
        train = json.load(f)

    #with open(fp+'/dev3.json','r') as f:
    #    dev = json.load(f)

    with open(fp+'/test_hyn.json','r') as f:
        test = json.load(f)

    with open(fp+'/parameters.json','r') as f:
        parameters = json.load(f)

    with open(fp+'/lemma_inverted_index.json','r') as f:
        lemma_inv_index = json.load(f)
    inv_paths_index = {}
    for x in paths_index_direct:
        y, z = paths_index_direct[x]
        inv_paths_index[(y,z)] = x
        inv_paths_index[(z,y)] = x

    tr = path_to_fix_length(train, args.path_len)
    test_path = path_to_fix_length_test(train, args.path_len)
    print(len(train),len(tr), len(test_path))

    return tr, test_path, test, paths, w2v, paths_index_direct, parameters, inv_paths_index, lemma_inv_index, 

def feature_load(fp, args):
    nfd_norm = np.loadtxt(fp+'/nfd_norm.txt')
    gene_diff = np.loadtxt(fp+'/gene_diff.txt')
    suffix = np.loadtxt(fp+'/Suffix.txt')
    contain = np.loadtxt(fp+'/Contains.txt')
    ld = np.loadtxt(fp+'/LD.txt')
    lcs = np.loadtxt(fp+'/LCS.txt')
    ends = np.loadtxt(fp+'/Ends.txt')
    return nfd_norm, gene_diff, suffix, contain, ld, lcs, ends

def context_gnn_load(fp):
    context = np.loadtxt(fp + '/w2v_gnn.txt')
    with open(fp+'/w2v_gnn_idx.txt','r') as f:
        context_idx = json.load(f)
    return context, context_idx


def path_to_fix_length(train_data, length = 3):
    train_path = []
    test_path = []
    visited_path = []
    for x in train_data:
        data_len = x["len"]
        attached = x["attach"]
        for i in range(data_len - length + 1): # 0~3, 1~4
            #print(i)
            attached_ = {idx: (attached[idx]-i) for idx in attached if (attached[idx]-i)>=0 and (attached[idx]-i)<length }
            if len(attached_) == 0:
                continue
            if i > 0 and x["path"][i-1][0] not in attached_:
                attached_[ str(x["path"][i-1][0]) ] = 0
            if x["path"][i : (i+length)] not in visited_path:
                if len(attached_) > 0:
                    visited_path.append(x["path"][i : (i+length)])
                    train_path.append({"path": x["path"][i : (i+length)], "neg": x["neg"] , "len": length, "attach": attached_})
            else:
                for idx,z in enumerate(train_path):
                    if train_path[idx]["path"] == x["path"][i : (i+length)]:
                        for neg_ in x["neg"]:
                            if neg_ not in train_path[idx]["neg"] and neg_ not in train_path[idx]["attach"]:
                                train_path[idx]["neg"].append(neg_)
                        for att_ in attached_:
                            if att_ not in train_path[idx]["attach"]:
                                train_path[idx]["attach"][att_] = attached_[att_]
                            #print(att_)
    return train_path#, test_path

def path_to_fix_length_test(train_data, length = 3):
    train_path = []
    test_path = []
    visited_path = []
    for x in train_data:
        data_len = x["len"]
        attached = x["attach"]
        #if 1219 in [z[0] for z in x["path"]]:
        #	print(data_len, x["path"], attached)
        for i in range(data_len - length + 1): # 0~3, 1~4
            #print(i)
            attached_ = {idx: (attached[idx]-i) for idx in attached if (attached[idx]-i)>=0 and (attached[idx]-i)<length }
            #if len(attached_) == 0:
            #    continue
            if i > 0 and x["path"][i-1][0] not in attached_:
                attached_[ str(x["path"][i-1][0]) ] = 0
            if x["path"][i : (i+length)] not in visited_path:
                #if len(attached_) > 0:
                visited_path.append(x["path"][i : (i+length)])
                train_path.append({"path": x["path"][i : (i+length)], "neg": x["neg"] , "len": length, "attach": attached_})
            else:
                for idx,z in enumerate(train_path):
                    if train_path[idx]["path"] == x["path"][i : (i+length)]:
                        for neg_ in x["neg"]:
                            if neg_ not in train_path[idx]["neg"] and neg_ not in train_path[idx]["attach"]:
                                train_path[idx]["neg"].append(neg_)
                        for att_ in attached_:
                            if att_ not in train_path[idx]["attach"]:
                                train_path[idx]["attach"][att_] = attached_[att_]
                            #print(att_)
    #print(train_path)
    return train_path#, test_path
