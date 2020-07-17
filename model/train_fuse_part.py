import time
import random
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import Taxonomy, Taxonomy_test
from tqdm import tqdm
from model_fuse import model#,model2
from layers_path import Transformer
from torch.autograd import Variable
import argparse
import json
from utils_path import *
from test_fuse import testing, testing_f1

#import torchsnooper
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--optimizer', type = str, default='Adam', help='Optimizer.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=70, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=2e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=60, help='Number of hidden units.')
#parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--cudaid', type=int, default=1, help='Cuda ID (0~3)')
parser.add_argument('--maxpath', type=int, default=200, help='Maximum number of dependency paths')
parser.add_argument('--batchsize', type=int, default=1, help='size of batch')
parser.add_argument('--layers', type=int, default=3, help='size of layers')
#parser.add_argument('--dev_epoch', type=int, default=2, help='Interval epoch for validation')
parser.add_argument('--test_epoch', type=int, default=20, help='Interval epoch for test')
parser.add_argument('--path_len', type=int, default=3, help='Length of term paths')
parser.add_argument('--fp', type=str, default='../data_train_test_path', help='Length of term paths')
parser.add_argument('--attach_path', type=int, default=1, help='Length of term paths')
parser.add_argument('--savepath', type=str, default='path', help='Length of term paths')
parser.add_argument('--encode_dep', type=str, default='rnn', help='Length of term paths')
parser.add_argument('--encode_prop', type=str, default='rnn', help='Length of term paths')
parser.add_argument('--method', type=str, default='our', help='Length of term paths')
parser.add_argument('--minus_only', type=int, default=1, help='Length of term paths')
parser.add_argument('--with_attn', type=int, default=1, help='use attn or not in lstm')
parser.add_argument('--model_name', type=str, default='123', help='use attn or not in lstm')
parser.add_argument('--lambda1', type=float, default=0.2, help='use attn or not in lstm')
parser.add_argument('--lambda2', type=float, default=0.2, help='use attn or not in lstm')
parser.add_argument('--load_gcn', type=int, default=0, help='use gcn or not')
parser.add_argument('--load_score', type=int, default=1, help='use gcn or not')
parser.add_argument('--load_emb', type=int, default=1, help='use distributed feature or not')
parser.add_argument('--load_dep', type=int, default=1, help='use contextual features or not')
parser.add_argument('--load_pat', type=int, default=1, help='use linguistic patterns or not')
parser.add_argument('--taxi_feature', type=int, default=1, help='use features extracted from TAXI or not')

args = parser.parse_args()
args.attach_path = False if args.attach_path == 0 else True
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cudaid = 'cuda:%d'%(args.cudaid)
print(args,args.cudaid)

def random_choice(p, max_val= 50):
    candid = [x for x in range(len(p))]
    p = np.array(p) / np.sum(p)
    index = [np.random.choice(candid, p = p.ravel()) for _ in range(max_val)]
    val = [1 for _ in range(max_val)]
    return index, val

def write_json(fp, data):
    with open(fp, 'w') as f:
        json.dump(data, f)

def optimizer_select(model, args):
    embedding_params = list(map(id, model.term_embedding.parameters()))
    base_params = filter(lambda p:id(p) not in embedding_params, model.parameters())

    if (args.optimizer == 'sgd'):
        optimizer = optim.SGD(
            [
                {'params': base_params},
                {'params': model.term_embedding.parameters(), 'lr': 2e-3}
            ],  
            lr = args.lr,
            weight_decay = args.weight_decay,
            momentum = 0.9
    )
    elif (args.optimizer == 'Adam'):
        optimizer = optim.Adam(
             [
                {'params': base_params},
                {'params': model.term_embedding.parameters(), 'lr': 5e-4}
            ],  
            lr = args.lr,
            weight_decay = args.weight_decay
    )
    else:
        raise NotImplementedError
    return optimizer

def importance_sample():
	pass

def train(models, taxo, batchsize, dep_path, device, optimizer, neg_sample_ratio = 1, compound = False, synonym = True, innerpath = False):
    #for batch_idx, train_dict in enumerate(taxo):
    #print(len(taxo))
    #print(epoch)
    running_loss = 0.0
    running_num = 0
    lnum = [0 for _ in range(args.path_len+1)]
    for i in taxo:
        for pos_sample in i["pos"]:
            z = int(i["label"][pos_sample])
            lnum[z] += 1
    lnum[-1] = int(np.sum(lnum[:-1])*neg_sample_ratio)
    lnum = torch.FloatTensor([np.max(lnum)/x for x in lnum]).to(device)
    #print(lnum)
    loss = nn.CrossEntropyLoss(weight = lnum)
    loss2 = nn.MSELoss()
    loss_dict = {}
    for i in range(int(len(taxo)/batchsize)):
        #print(i, batchsize)   
        l = 0 
        running_num = 0
        for j in range(i*batchsize, min((i+1)*batchsize, len(taxo)-1)):
            labels = []
            pred = []
            embed_preds = []
            taxi_preds = []
            paths_preds = []
            #print(len(taxo[j]["pos"]), len(taxo[j]["neg"]))
            target_path = taxo[j]["path"] # [[term1], [term21, term22], [...], [...]]
            inner_path = taxo[j]["inner_path"] # []
            if innerpath:
                inner = []
                freq = []
                for z in inner_path:
                    inner.extend(dep_path[z[0]]["path"])
                    freq.extend(dep_path[z[0]]["freq"])
                #print(inner,freq)
                if len(inner) > 100:
                    sample_idx, sample_freq = random_choice(freq, max_val= 100)
                    inner = [inner[x] for x in sample_idx]
                    freq = sample_freq
            else:
                inner = []
                freq = []
            for pos_sample in taxo[j]["pos"]:
                label = taxo[j]["label"][pos_sample]
                pos_path = taxo[j]["pos"][pos_sample]
                if pos_path == [None, None, None]:
                    if np.random.rand() > 0.5:
                        #print('skip')
                        continue
                #print(inner, freq)
                dependency_paths = []
                dependency_freq = []
                for y in pos_path:
                    if y == None:
                        dependency_paths.append([])
                        dependency_freq.append([])
                    else: #no path with 2 ids
                        #if len(y)>1:
                        #    print(len(y),y)
                        if len(dep_path[y[0]]["path"]) > 100: # sample 50 paths if there is too much dependency path
                            sample_idx, sample_freq = random_choice(dep_path[y[0]]["freq"], max_val= 100)
                            #print(dep_path[xx]["path"], dep_path[xx]["freq"], sample_idx, sample_freq)
                            dependency_paths.append([dep_path[y[0]]["path"][x] for x in sample_idx])
                            dependency_freq.append(sample_freq)
                        else:
                            dependency_paths.append(dep_path[y[0]]["path"])
                            dependency_freq.append(dep_path[y[0]]["freq"])
                embed, paths_pred, emb_pred, taxi_pred = models(target_path = target_path, inner_path = inner, inner_freq = freq, attach = int(pos_sample), \
                    attach_path = dependency_paths, attach_freq = dependency_freq, device = device, compound = compound)
                pred.append(embed)
                labels.append(label)
                embed_preds.append(emb_pred)
                paths_preds.append(paths_pred)
                taxi_preds.append(taxi_pred)
                #print(embed.shape)
            if 1:
                have_path = []
                no_path = []
                for x in taxo[j]["neg"]:
                    num_path = len([z for z in taxo[j]["neg"][x] if z!= None])
                    if num_path == 0:
                        no_path.append(x)
                    else:
                        have_path.append(x)
                #print(len(no_path), len(have_path))

                #neg_samples = random.sample(list(taxo[j]["neg"]), int(neg_sample_ratio*len(taxo[j]["pos"])))
                if len(have_path) >= int(neg_sample_ratio*len(taxo[j]["pos"])):
                    neg_samples = random.sample(have_path, int(neg_sample_ratio*len(taxo[j]["pos"])))
                else:
                    neg_samples = have_path + random.sample(no_path, int(neg_sample_ratio*len(taxo[j]["pos"]))-len(have_path))
                #print(len(taxo[j]["pos"]),len(neg_samples))
                for neg_sample in neg_samples:
                    label = len(target_path)
                    neg_path = taxo[j]["neg"][neg_sample]
                    dependency_paths = []
                    dependency_freq = []
                    for y in neg_path:
                        if y == None:
                            dependency_paths.append([])
                            dependency_freq.append([])
                        else: #no path with 2 ids
                            #if len(y)>1:
                            #    print(len(y),y)
                            if len(dep_path[y[0]]["path"])>100: # sample 50 paths if there is too much dependency path
                                sample_idx, sample_freq = random_choice(dep_path[y[0]]["freq"], max_val= 100)
                                #print(dep_path[xx]["path"], dep_path[xx]["freq"], sample_idx, sample_freq)
                                dependency_paths.append([dep_path[y[0]]["path"][x] for x in sample_idx])
                                dependency_freq.append(sample_freq)
                            else:
                                dependency_paths.append(dep_path[y[0]]["path"])
                                dependency_freq.append(dep_path[y[0]]["freq"])
                    embed , paths_pred, emb_pred, taxi_pred = models(target_path = target_path, inner_path = inner, inner_freq = freq, attach = int(neg_sample), \
                    attach_path = dependency_paths, attach_freq = dependency_freq, device = device, compound = compound)
                    #print(embed)
                    #embed = F.softmax(embed, dim = 1)
                    #print(embed)
                    embed_preds.append(emb_pred)
                    paths_preds.append(paths_pred)
                    taxi_preds.append(taxi_pred)
                    pred.append(embed)
                    labels.append(label)
            #print('labels', labels)
            labels = torch.LongTensor(labels).to(device)
            pred, embed_preds, paths_preds, taxi_preds = torch.cat(pred, dim = 0), torch.cat(embed_preds, dim = 0), torch.cat(paths_preds, dim = 0), torch.cat(taxi_preds, dim = 0)
            #print(F.softmax(embed_preds, dim = 1))
            #print(paths_preds.shape,embed_preds.shape, pred.shape,taxi_preds.shape)
            try:
                if self.load_dep and self.load_emb:
                    l += loss(pred, labels) + args.lambda1 * (loss(embed_preds, labels) + loss(paths_preds, labels)) \
                    + args.lambda2 * loss2(F.softmax(embed_preds, dim = 1), F.softmax(paths_preds, dim = 1))
                if self.load_dep and self.load_pat:
                    l += loss(pred, labels) + args.lambda1 * (loss(paths_preds, labels) + loss(taxi_preds, labels)) \
                    + args.lambda2 * loss2(F.softmax(paths_preds, dim = 1), F.softmax(taxi_preds, dim = 1)) 
                if self.load_pat and self.load_emb:
                    l += loss(pred, labels) + args.lambda1 * (loss(embed_preds, labels) + loss(taxi_preds, labels)) \
                    + args.lambda2 * loss2(F.softmax(embed_preds, dim = 1),  F.softmax(taxi_preds, dim = 1))
            except NameError:
                l = loss(pred, labels) \
                    + args.lambda1 * (loss(embed_preds, labels) + loss(paths_preds, labels) + loss(taxi_preds, labels)) \
                    + args.lambda2 * (loss2(F.softmax(embed_preds, dim = 1),  F.softmax(taxi_preds, dim = 1)) \
                    + loss2(F.softmax(paths_preds, dim = 1), F.softmax(taxi_preds, dim = 1)) 
                    + loss2(F.softmax(embed_preds, dim = 1), F.softmax(paths_preds, dim = 1))) 
            
            running_loss = l.item()
            running_num += 1

            pbar.update(1)
            pbar.set_description('Fuse:%d/%d l1:%.2f l2:%.2f wd:%.3f Loss: %.4f' % (epoch+1, \
             int(args.epochs), args.lambda1, args.lambda2, args.weight_decay,running_loss/running_num))
            if epoch % 10 == 0:
                loss_dict['_'.join([str(x[0]) for x in target_path])] = l.item()
        optimizer.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(models.parameters(), 5.0)
        optimizer.step()
    if epoch % 10 == 0:
        write_json('../log_result/loss_dict_%d_%s.json'%(epoch,args.model_name), loss_dict)
        


if __name__ == "__main__":
    print("\n[STEP 1] : Training")
    batchsize = args.batchsize
    loss = []
    
    fp = args.fp
    train_path, test_path, test, paths, w2v, paths_index_direct, parameters, inv_paths_index, lemma_inv_index = train_test_data_load(fp, args)
    embed_dim = w2v.shape[-1]
    nfd_norm, gene_diff, suffix, contain, ld, lcs, ends = feature_load(fp, args)
    embed_context = 50
    if args.load_gcn:
        context, context_idx = context_gnn_load(fp)
        embed_context = context.shape[-1]
        if args.load_score:
            args.gnn_score = np.loadtxt(fp+'/score_gnn.txt')

    if 'science_wordnet' in args.model_name:
        dataset = 'science_wordnet_en'
    elif 'env' in args.model_name and 'eurovoc' in args.model_name:
        dataset = 'science_wordnet_en'
    elif 'food_wordnet' in args.model_name:
        dataset = 'science_wordnet_en'
    else:
        dataset = 'unknown'

    with open('../data_train_test_path/idx_cnts_%s.json'%(dataset),'r') as f:
        idx_cnt = json.load(f)
    print(len(test_path))
    max_len = -1
    for i in paths:
        for j in paths[i]["path"]:
            if len(j) > max_len:
                #print(len(j))
                max_len = len(j)

    taxo = Taxonomy(synonym=True, neg_sample=5, type='train', inv_paths_index = inv_paths_index, 
                    train = train_path, path = paths)
    taxo_t = Taxonomy(synonym=True, neg_sample=5, type='train', inv_paths_index = inv_paths_index, 
                    train = test_path, path = paths)
    taxo_test = Taxonomy_test(synonym = True, inv_paths_index = inv_paths_index,\
    						 taxo = taxo_t, path = paths, test_hyn= test)
    #train_loader = torch.utils.data.DataLoader(taxo,  batch_size=batchsize, shuffle=True)
    #print([x["pos"] for x in taxo.train_path])
    model = model(args = args, path_length = args.path_len, hidden_size = args.hidden, num_layers = 1, num_terms = w2v.shape[0], parameters = parameters,  \
     embed_dim= embed_dim, embed_context = embed_context, dropout = args.dropout, bidirection = False, taxi_feature = args.taxi_feature) #, attach_path= args.attach_path)
    if args.load_gcn:
        model.init_embed(w2v, context)
    else:
        model.init_embed(w2v)

    if args.taxi_feature:
        model.insert_taxi_feature(nfd_norm, gene_diff)
        model.insert_string_feature(suffix, contain, ld, lcs, ends)

    model = model.to(args.cudaid)
    optimizer = optimizer_select(model, args)

    #train(models = model, taxo = taxo, device = args.cudaid, batchsize = 10, dep_path = paths)

    with tqdm(total=args.epochs * int(len(taxo)/batchsize)) as pbar:
        for epoch in range(args.epochs):
            if epoch % 5 == 4:
                model.eval()

                #testing(models = model, taxo_test = taxo_test, device = args.cudaid, tes = test, \
                #    dep_path = paths, epoch = epoch, args = args)
                testing_f1(models = model, taxo_test = taxo_test, device = args.cudaid, test_syn = [], test_hyn = test, dep_path = paths, \
                    epoch = epoch, args = args,  inv_paths_index = inv_paths_index, idx_cnt = idx_cnt, metric= 'mean', is_softmax = True, \
                    compound = False, synonym = True, innerpath = False)

            model.train()
            train(models = model, taxo = taxo, device = args.cudaid, optimizer = optimizer, batchsize = 32, dep_path = paths)
            #train(models = model, epoch = epoch, batchsize = batchsize, pbar = pbar, valacc=valacc, attach_path= args.attach_path, )
    #all_paths = [train_set[str(x)]["triple"] for x in range(len(train_set))]
    print(args)