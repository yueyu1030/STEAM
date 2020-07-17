# -*- coding: utf-8 -*-
import codecs
import json
import numpy as np
import random

def train_test_data_load(fp):
    if 'customize' not in fp:
        with open(fp+'/paths.json','r') as f:
            paths = json.load(f)
        with open(fp+'/paths_index_direct.json','r') as f:
            paths_index_direct = json.load(f)
    else:
        with open(fp+'/path_customized.json','r') as f:
            paths = json.load(f)
        paths_index_direct = []
    w2v = np.loadtxt(fp+'/w2v.txt')

    '''
    with open(fp+'/lemma_index_customized.json','r') as f:
        customize_idx = json.load( f)

    with open(fp+'/lemma_inverted_index_customized.json','r') as f:
        customize_inverted_idx = json.load(f)
    '''

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

    return train, test, paths, w2v, paths_index_direct, parameters, inv_paths_index, lemma_inv_index