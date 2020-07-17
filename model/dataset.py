import torch
from torch.utils.data import Dataset
import json
import numpy as np


def open_json(fp):
    with open(fp, 'r') as f:
        data = json.load(f)
    return data

def write_json(fp, data):
    with open(fp, 'r') as f:
        json.dump(data, f)

class Taxonomy(Dataset):
    #pulationDataset(Dataset):

    def __init__(self, synonym, neg_sample,
                type, inv_paths_index,
                train, path):
        self.path = path
        #self.embedding = w2v
        self.train = train
        self.neg_sample = neg_sample
        self.synonym = synonym
        self.inv_paths_index = inv_paths_index
        #self.embedding = []
        self.inner_paths = []
        self.train_path = []
        self.poslabels = []
        self.build_trainset()
        self.len = len(self.train)

    def get_path_from_terms(self, src, tgt):
        paths = []
        #for s in src:
        #    for t in tgt:
        s = src[0]
        t = tgt[0]
        if (s,t) in self.inv_paths_index:
            paths.append(self.inv_paths_index[(s,t)])
        return paths

    def get_path_from_terms_with_synonym(self, src, tgt):
        paths = []
        for s in src:
            for t in tgt:
                if (s,t) in self.inv_paths_index:
                    paths.append(self.inv_paths_index[(s,t)])
        return paths

    #def get_path_from_neg(self,):

    def get_path_from_attach_terms(self, attach, original, length):
        #attach: a number
        #original: a 
        path = []
        if self.synonym:
            for o in original:
                if (attach, o[0]) in self.inv_paths_index:
                    path.append([self.inv_paths_index[(attach, o[0])]])
                else:
                    path.append(None)
        else:
            for o in original:
                path_tmp = []
                for x in o:
                    if (attach, x) in self.inv_paths_index:
                        path_tmp.append(self.inv_paths_index[(attach, o[0])])      
                path.append(path_tmp if len(path_tmp)>0 else None)
        #print(path, len(path), length)
        assert len(path) == length
        return path     
        
    def build_trainset(self):
        ## decode path information
        ## input: path id
        ## output: 
        ##  1) the path information for all node pairs in a path
        ##  2) the embedding id information
        ##  3) direct interacted paths
        synonym = self.synonym
        if not synonym:
            terms = [y[:1] for y in x["path"] for x in self.train]

        for i in self.train:
            tmp = []
            if synonym:
                for j in range(i["len"]-1):
                    tmp.extend(self.get_path_from_terms(i["path"][j], i["path"][j+1]))
            else:
                for j in range(i["len"]-1):
                    tmp.extend(self.get_path_from_terms(i["path"][j], i["path"][j+1]))
            self.inner_paths.append(tmp)

        for i in self.train:
            pos_path = {} #{id: path_id}
            neg_path = {} #{id: path_id}
            label = {}
            #print(i["attach"], i["neg"])
            #assert 0
            for x in i["attach"]:
                pos_path[x] = self.get_path_from_attach_terms(int(x), i["path"], i["len"])
                label[x] = i["attach"][x]
            for x in i["neg"]:
                neg_path[x] = self.get_path_from_attach_terms(x, i["path"], i["len"])
            self.train_path.append({"pos": pos_path, "neg": neg_path})
            self.poslabels.append(label)


    def __getitem__(self, i):
        return  {
                    "path": self.train[i]["path"] ,
                    "pos": self.train_path[i]["pos"] ,
                    "neg": self.train_path[i]["neg"] ,
                    "inner_path": self.inner_paths[i],
                    "label": self.poslabels[i]
                }

    def __len__(self):
        return self.len

        ## decode embedding information


        ## decode 

class Taxonomy_test(Dataset):
    #pulationDataset(Dataset):

    def __init__(self, synonym, inv_paths_index,
            taxo, path, test_hyn):
        self.path = path
        #self.embedding = w2v
        self.synonym = synonym
        self.inv_paths_index = inv_paths_index
        #self.embedding = []
        self.label = {}
        self.train_path = {}
        self.taxo = taxo
        self.inner_paths = self.taxo.inner_paths
        self.len = len(self.taxo)
        self.pos_term = []
        self.test_hyn = test_hyn
        self.parent = {}
        self.taxo_path = [self.taxo[i]["path"] for i in range(len(self.taxo))]
        self.build_trainset()
        self.find_parent()
        #print(self.parent)

    def find_parent(self):
        for i in range(len(self.taxo)):
            for p in range(len(self.taxo[i]["path"])-1):
                #print(self.taxo[i]["path"][p],self.taxo[i]["path"][p+1])
                self.parent[self.taxo[i]["path"][p][0]] = self.taxo[i]["path"][p+1][0]


    def get_path_from_terms(self, src, tgt):
        paths = []
        s = src[0]
        t = tgt[0]
        if (s,t) in self.inv_paths_index:
            paths.append(self.inv_paths_index[(s,t)])
        return paths

    def get_path_from_terms_with_synonym(self, src, tgt):
        paths = []
        for s in src:
            for t in tgt:
                if (s,t) in self.inv_paths_index:
                    paths.append(self.inv_paths_index[(s,t)])
        return paths

    def get_path_from_attach_terms(self, attach, original, length):
        #attach: a number
        #original: a 
        path = []
        if self.synonym:
            for o in original:
                if (attach, o[0]) in self.inv_paths_index:
                    path.append([self.inv_paths_index[(attach, o[0])]])
                else:
                    path.append(None)
        else:
            for o in original:
                path_tmp = []
                for x in o:
                    if (attach, x) in self.inv_paths_index:
                        path_tmp.append(self.inv_paths_index[(attach, o[0])])      
                path.append(path_tmp if len(path_tmp)>0 else None)
        #print(path, len(path), length)
        assert len(path) == length
        return path     
        
    def build_trainset(self):
        ## decode path information
        ## input: path id
        ## output: 
        ##  1) the path information for all node pairs in a path
        ##  2) the embedding id information
        ##  3) direct interacted paths
        synonym = self.synonym
        l = self.len
        for j in range(l):
            self.pos_term.append( [int(x[0]) for x in self.taxo[j]["path"]] )

        for i in self.test_hyn:
            pos_path = [] #{id: path_id}
            term_id = int(i)
            gold_pos = self.test_hyn[i]["attach"]
            #assert 0
            for j in range(l):
                pos_path.append( self.get_path_from_attach_terms(term_id, self.taxo[j]['path'],\
                    len(self.taxo[j]['path'])) )
            self.label[term_id] = gold_pos
            self.train_path[term_id] = pos_path
    
    def expand_test_set(self, new_term, new_term_parent, path_length):
        self.parent[new_term] = new_term_parent
        new_path = []
        start_ = new_term
        for _ in range(path_length):
            new_path.append(start_)
            start_ = self.parent[start_]
        # new path is the generated path, we add them to the test set
        self.pos_term.append(new_path)
        self.len += 1
        add_inner_path = []
        for _ in range(path_length-1):
            add_inner_path.extend(self.get_path_from_terms(new_path[i], new_path[i+1]))
        self.inner_paths.append(add_inner_path)

        self.taxo_path.append([[x] for x in new_path])
        for i in self.test_hyn:
            pos_path = [] #{id: path_id}
            term_id = int(i)
            gold_pos = self.test_hyn[i]["attach"]
            for j in range(l):
                pos_path.append( self.get_path_from_attach_terms(term_id, self.taxo[j]['path'],\
                    len(self.taxo[j]['path'])) )

            self.label[term_id] = gold_pos
            self.train_path[term_id] = pos_path

    def __len__(self):
        return self.len