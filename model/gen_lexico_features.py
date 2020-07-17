from difflib import SequenceMatcher
import numpy as np
import numpy as np
import csv
import json
from collections import Counter
import copy
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

plt.switch_backend('agg')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='environment_eurovoc_en', help='Dataset')
parser.add_argument('--tr', type=float, default=0.4, help='Ratio')
args = parser.parse_args()


dataset = args.dataset #'environment_eurovoc_en'
tr = args.tr
root = 'data_%s_%.1f/'%(dataset, tr)
TAXO_NODE_INFO_PATH = root+ 'taxo_node_info.json'
LEMMA_INFO_PATH = root+ 'lemma_index.json'

with open(TAXO_NODE_INFO_PATH, 'r') as f:
    taxo_node_info = json.load(f)

with open(LEMMA_INFO_PATH, 'r') as f:
    lemma_info = json.load(f)

lemma_key = copy.deepcopy(list(lemma_info.keys()))
for x in lemma_key:
    lemma_info[x.lower()] = lemma_info[x]

num_of_terms = np.max(list(lemma_info.values())) + 1
print(num_of_terms)

ld= np.zeros([num_of_terms, num_of_terms])
lcs = np.zeros([num_of_terms, num_of_terms])
ends = np.zeros([num_of_terms, num_of_terms])
contains = np.zeros([num_of_terms, num_of_terms])
suffix = np.zeros([num_of_terms, num_of_terms])


term_cnt = np.zeros(num_of_terms)

def Capitalization(x, y, lower2original):
    if x in lower2original:
        x = lower2original[x]
    if y in lower2original:
        y = lower2original[y]
    if x[0].isupper() and y[0].isupper():
        return 0
    elif x[0].isupper() and not y[0].isupper():
        return 1
    elif not x[0].isupper() and y[0].isupper():
        return 2
    elif not x[0].isupper() and not y[0].isupper():
        return 3


def Endswith(x, y):
    return int(y.endswith(x)) #{0, 1}


def Contains(x, y):
    return int(x in y) #{0, 1}


def Suffix_match(x, y):
    k = 7
    for i in range(k):
        if x[-i - 1:] != y[-i - 1:]:
            return i
    return k  #[0, 7]


def LCS(x, y):
    match = SequenceMatcher(None, x, y).find_longest_match(0, len(x), 0, len(y))
    res = 2.0 * match.size / (len(x) + len(y))  # [0, 1]
    return res#int(round(res, 1) * 10)  # [0, 10]


def LD(x, y):
    res = 2.0 * (len(x) - len(y)) / (len(x) + len(y))  # (-2,2)
    return res#int(round(res, 1) * 10 + 20)  # [0, 40]


def normalized_freq_diff(hypo2hyper, x, y):
    if x not in hypo2hyper or y not in hypo2hyper[x] or hypo2hyper[x][y] == 0:
        a = 0
    else:
        a = float(hypo2hyper[x][y]) / max(hypo2hyper[x].values())
    if y not in hypo2hyper or x not in hypo2hyper[y] or hypo2hyper[y][x] == 0:
        b = 0
    else:
        b = float(hypo2hyper[y][x]) / max(hypo2hyper[y].values())
    res = a - b  # [-1, 1]
    # return res
    return int(res * 10) + 10  # [0, 20]


def generality_diff(hyper2hypo, x, y):
    if x not in hyper2hypo:
        b = 0
    else:
        b = np.log(1 + len([i for i in hyper2hypo[x] if hyper2hypo[x][i] != 0]))
    if y not in hyper2hypo:
        a = 0
    else:
        a = np.log(1 + len([i for i in hyper2hypo[y] if hyper2hypo[y][i] != 0]))
    res = a - b  # (-7.03, 7.02)
    # return res
    return int(res) + 7  # [0, 14]


def get_all_features(x, y, sub_feat=False, hypo2hyper=None, hyper2hypo=None, lower2original=None):
    feat = {}
    #feat['Capitalization'] = Capitalization(x, y, lower2original)
    feat['Endswith'] = Endswith(x, y)
    feat['Contains'] = Contains(x, y)
    feat['Suffix_match'] = Suffix_match(x, y)
    feat['LCS'] = LCS(x, y)
    feat['LD'] = LD(x, y)
    if sub_feat:
        if hypo2hyper is None or hyper2hypo is None:
            print('features.py: hypo2hyper not loaded')
            exit(-2)
        feat['Freq_diff'] = normalized_freq_diff(hypo2hyper, x, y)
        feat['General_diff'] = generality_diff(hyper2hypo, x, y)
    return feat


def main():
    for x in tqdm(lemma_info):
        x_idx = int(lemma_info[x])
        x_term = x 
        for y in lemma_info:
            y_idx = int(lemma_info[y])#int(y)
            y_term = y
            if x_idx != y_idx:
                feat = get_all_features(x_term, y_term)
                lcs[x_idx, y_idx] = feat["LCS"]
                ld[x_idx, y_idx] = feat["LD"]
                ends[x_idx, y_idx] = feat["Endswith"]
                suffix[x_idx, y_idx] = feat["Suffix_match"]
                contains[x_idx, y_idx] = feat["Contains"]
    save_data(lcs, ld, ends, suffix, contains, root)
    #draw_taxi_score(taxo_node_info, lcs, ld)
    draw_taxi_score(taxo_node_info, lcs, ld, 1)
    draw_taxi_score(taxo_node_info, lcs, ends, 2)
    draw_taxi_score(taxo_node_info, lcs, contains, 3)
    draw_taxi_score(taxo_node_info, lcs, suffix, 4)

    
def save_data(lcs, ld, ends, suffix, contains, root):
    np.savetxt( root+'LCS.txt', lcs, fmt = '%.4f')
    np.savetxt(root+'LD.txt' , ld, fmt = '%.4f')
    np.savetxt( root+'Ends.txt', ends, fmt = '%.4f')
    np.savetxt(root+'Suffix.txt' , suffix, fmt = '%.4f')
    np.savetxt( root+'Contains.txt', contains, fmt = '%.4f')
    #np.savetxt(root+'LD.txt' , LD, fmt = '%.4f')

def draw_taxi_score(taxo_node_info, gene_diff, nfd_norm, idx = 0):
    plt.figure(figsize = [15.5,9], dpi = 150)
    pair = []
    for i in tqdm(taxo_node_info):
        if taxo_node_info[i]["syn"] == 1:
            continue
        x  = int(lemma_info[taxo_node_info[i]["term"][0].lower()])
        '''
        '''
        
        if taxo_node_info[i]["parent"]!=[]:
            y = int(lemma_info[taxo_node_info[taxo_node_info[i]["parent"]]["term"][0].lower()])#int(taxo_node_info[i]["parent"])
            coordinate = [gene_diff[x,y], nfd_norm[x,y]]
            pair.append(coordinate)
            plt.scatter(coordinate[0], coordinate[1], c='r', marker='o')
            #for z in tqdm(range(gene_diff.shape[0])):
            for j in tqdm(taxo_node_info):
                #print(j, taxo_node_info[j]["term"])
                if taxo_node_info[j]["term"]!=[]:
                    z = int(lemma_info[taxo_node_info[j]["term"][0].lower()])
                    if z!=y:
                        coordinate = [gene_diff[x,z], nfd_norm[x,z]]
                        if coordinate[0] != 0 and coordinate[1] != 0:# and [coordinate[0], coordinate[1]] not in pair:
                            plt.scatter(coordinate[0], coordinate[1], c='g', marker='^')
    plt.tight_layout()
    plt.savefig(root+'feature_%s_%d.png'%(dataset, idx))

if __name__ == '__main__':
	main()


