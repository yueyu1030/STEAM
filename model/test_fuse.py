import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence 
import numpy as np 
from tqdm import tqdm
import json
import scipy
from scipy import stats

def random_choice(p, max_val= 50):
    candid = [x for x in range(len(p))]
    p = np.array(p) / np.sum(p)
    index = [np.random.choice(candid, p = p.ravel()) for _ in range(max_val)]
    val = [1 for _ in range(max_val)]
    return index, val

def getrank(a, idx=0):
    return len(a)-list(np.array(a).argsort()).index(idx)

def write_json(fp, data):
    with open(fp, 'w') as f:
        json.dump(data, f)

def testing(models, taxo_test, device, test_syn, test_hyn, dep_path, epoch, args, is_softmax = False, compound = False, synonym = True, innerpath= False):
    models.eval() 
    y_true = []
    y_pred = []
    score_pred = {}
    rank = []
    correct = 0 
    total = 0
    score_save = {}
    for t in tqdm(test_hyn):
        score_t = {}
        score_tmp = []
        num_of_paths = {}
        test_node = int(t)
        attach_pos = int(test_hyn[t]["attach"])
        relation_type = test_hyn[t]["type"]
        for i in range(len(taxo_test)):
            target_path = taxo_test.taxo[i]["path"]
            inner_path = taxo_test.inner_paths[i]
            train_path = taxo_test.train_path[test_node][i]
            pos_term = taxo_test.pos_term[i]

            if innerpath:
                inner = []
                freq = []
                for z in inner_path:
                    inner.extend(dep_path[z[0]]["path"])
                    freq.extend(dep_path[z[0]]["freq"])
                if len(inner) > 100:
                    sample_idx, sample_freq = random_choice(freq, max_val= 100)
                    inner = [inner[x] for x in sample_idx]
                    freq = sample_freq
            else:
                inner = []
                freq = []
            dependency_paths = []
            dependency_freq = []
            for y in train_path:

                if y == None:
                    dependency_paths.append([])
                    dependency_freq.append([])
                elif len(dep_path[y[0]]["path"]) > 100: # sample 50 paths if there is too much dependency path
                    sample_idx, sample_freq = random_choice(dep_path[y[0]]["freq"], max_val= 100)
                    #print(dep_path[xx]["path"], dep_path[xx]["freq"], sample_idx, sample_freq)
                    dependency_paths.append([dep_path[y[0]]["path"][x] for x in sample_idx])
                    dependency_freq.append(sample_freq)
                else:
                    dependency_paths.append(dep_path[y[0]]["path"])
                    dependency_freq.append(dep_path[y[0]]["freq"])
            score, embed_preds, paths_preds, taxi_preds = models(target_path = target_path, inner_path = inner, inner_freq = freq, attach = test_node, \
                    attach_path = dependency_paths, attach_freq = dependency_freq, device = device, compound = compound)
            
            if is_softmax:
                #print(score)
                score = F.softmax(score, dim = 1)
                #print(score)
            score = score.detach().cpu().numpy().reshape(-1)
            #print(score)
            score_tmp.append({str(int(x)): '%.3f'%y for (x,y) in zip(pos_term, score)})
            #print(score.shape[0], len(pos_term))
            for e,s in zip(pos_term, score):
                if e not in score_t:
                    score_t[e] = [s]
                else:
                    score_t[e].append(s)
        l = len(score_t)
        score_save[t] = score_tmp
        if attach_pos in score_t:
            ranklst = [ np.mean(score_t[attach_pos]) ]
            for r in score_t:
                if r != attach_pos:
                    ranklst.append( np.mean(score_t[r]) )
            rank_ = getrank(ranklst, idx = 0)
            if rank_ == 1:
                correct += 1
            total += 1
            rank.append(rank_)
        score_pred[t] = score_t

    write_json('../log_result/%s_%s_%d_score.json'%(args.encode_dep, args.encode_prop, epoch), score_save)
    if synonym:
        pass
    print('\n=> Testing: Mean Rank:%.2f/%d, Acc = %.3f'%(np.mean(rank), l, float(correct/total)))
    
def testing_f1(models, taxo_test, device, test_syn, test_hyn, dep_path, epoch, args, inv_paths_index, idx_cnt, metric= 'mean', is_softmax = False, compound = False, synonym = True, innerpath= False):
    models.eval() 
    wum = np.loadtxt('%s/wum.txt'%(args.fp))
    y_true = []
    y_pred = []
    score_pred = {} # score all score for nodes [format: attach_term:{all score dict} ]
    score_one = {}  # score all score for nodes [format: attach_term:{overall score with the attach node} ]
    score_best = {} # [format: attach_term:{best score with the attach node} ]
    score_save_dict = {}
    attach_term_list = [] # attached node in test set in the taxo
    to_attached_number = len(test_hyn)
    rank = []
    wup = []
    correct = 0 
    total = 0
    total_rank = 0
    score_save = {}
    rank_save = {}
    path_save = {}
    cnt_save = {}
    score_test_save = {}
    target_paths = [taxo_test.taxo[i]["path"] for i in range(len(taxo_test))]
    inner_paths = [taxo_test.inner_paths[i] for i in range(len(taxo_test))]
    train_paths = {int(x):taxo_test.train_path[int(x)] for x in test_hyn}
    pos_terms = [taxo_test.pos_term[i] for i in range(len(taxo_test))]
    num_of_paths={}

    attach_pos_gold = {int(t):int(test_hyn[t]["attach"]) for t in test_hyn}
    for t in tqdm(test_hyn):
        score_t = {} # all score
        one_score = {} #one score(do average)
        score_tmp = []
        test_node = int(t)
        num_paths = {}
        attach_pos = int(test_hyn[t]["attach"])
        relation_type = test_hyn[t]["type"]
        for i in range(len(taxo_test)):
            target_path = taxo_test.taxo[i]["path"]
            inner_path = taxo_test.inner_paths[i]
            train_path = taxo_test.train_path[test_node][i]
            pos_term = taxo_test.pos_term[i]
            #print(target_path, inner_path, pos_term)

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
            dependency_paths = []
            dependency_freq = []
            for i, y in enumerate(train_path):
                num_paths[target_path[i][0]] = len(dep_path[y[0]]["path"]) if y!=None else 0
                if y == None:
                    dependency_paths.append([])
                    dependency_freq.append([])
                elif len(dep_path[y[0]]["path"])>100: # sample 50 paths if there is too much dependency path
                    sample_idx, sample_freq = random_choice(dep_path[y[0]]["freq"], max_val= 100)
                    #print(dep_path[xx]["path"], dep_path[xx]["freq"], sample_idx, sample_freq)
                    dependency_paths.append([dep_path[y[0]]["path"][x] for x in sample_idx])
                    dependency_freq.append(sample_freq)
                else:
                    dependency_paths.append(dep_path[y[0]]["path"])
                    dependency_freq.append(dep_path[y[0]]["freq"])
            score, embed_preds, paths_preds, taxi_preds = models(target_path = target_path, inner_path = inner, inner_freq = freq, attach = test_node, \
                    attach_path = dependency_paths, attach_freq = dependency_freq, device = device, compound = compound)
            if is_softmax:
                score = F.softmax(score, dim = 1)
            score = score.detach().cpu().numpy().reshape(-1)
            embed_preds, paths_preds, taxi_preds = embed_preds.detach().cpu().numpy().reshape(-1),paths_preds.detach().cpu().numpy().reshape(-1),taxi_preds.detach().cpu().numpy().reshape(-1)
			
            score_tmp.append({str(int(x)): 'o:%.3f,emb:%.3f,dep:%.3f,taxi:%.3f'%(y, y1, y2, y3) for (x, y, y1, y2, y3) in zip(pos_term, score, embed_preds, paths_preds, taxi_preds)})
            '''
            we calculate the score of all terms for term t here (e is the term in train taxo)
            '''
            for e,s in zip(pos_term, score):
                if e not in score_t:
                    score_t[e] = [s]
                else:
                    score_t[e].append(s)
        l = len(score_t)
        #optim_pos = 0
        if attach_pos in score_t:
            ranklst = [ scipy.stats.gmean(score_t[attach_pos]) if is_softmax else np.mean(score_t[attach_pos]) ]
            optim_pos = attach_pos
            maxval = ranklst[0]
            #print('##########', num_paths[attach_pos], '#########')
            for r in score_t:
                if r != attach_pos:
                    ranklst.append( scipy.stats.gmean(score_t[r]) if is_softmax else np.mean(score_t[r]) )
                    if ranklst[-1] > maxval:
                        optim_pos = r
                        maxval = ranklst[-1]
            if optim_pos == attach_pos:
                wup.append(1)
            else:
                wup.append(wum[optim_pos, attach_pos])

            rank_ = getrank(ranklst, idx = 0)
            if rank_ == 1:

                correct += 1
            total_rank += 1
            rank.append(rank_)

            #score_pred[t] = np.mean(score_t[attach_pos])
            score_save[t] = score_tmp#
            score_test_save[t] = np.float(np.mean(score_t[attach_pos]))
            rank_save[t] = rank_
            path_save[t] = num_paths[attach_pos]
            cnt_save[t] = [idx_cnt[str(attach_pos)] if str(attach_pos) in idx_cnt else 0, idx_cnt[str(test_node)] if str(test_node) in idx_cnt else 0]

        score_pred[t] = score_t
        num_of_paths[test_node] = num_paths
        '''
        For all nodes, we calculate the mean score and get the highest score with the corresponding location
        '''
        for r in score_t:
            if metric =='mean':
                one_score[r] = np.float(scipy.stats.gmean(score_t[r])) if is_softmax else np.float(np.mean(score_t[r]))
            else:
                one_score[r] = np.max(score_t[r])
        '''
        Get maximum score for all nodes, decide the place to insert the node
        '''
        best_val, best_attach = get_max_from_dict(one_score) # attach: is the node in original taxo
        score_one[t] = one_score
        score_best[t] = {'best_val': best_val, 'attach': best_attach} # t: term in test set
    acc3 = getacc(rank, 3)
    acc5 = getacc(rank, 5)
    mrr = np.mean([1/x for x in rank])
    wup_score = np.mean(wup)
    if 'science_wordnet' in args.model_name:
        dataset = 'science_wordnet_en'
    elif 'env' in args.model_name and 'eurovoc' in args.model_name:
        dataset = 'environment_eurovoc_en'
    elif 'food_wordnet' in args.model_name:
        dataset = 'food_wordnet_en'
    else:
        dataset = 'unknown'
    if args.encode_prop in ['linear', 'none'] and epoch>1 and args.layers == 2:
        np.savetxt('../log_result/param_layer1_%d_%s.txt'%(epoch, args.model_name), models.gc2.mlp.dense0.weight.detach().cpu().numpy(), fmt = '%.5f')
        np.savetxt('../log_result/param_layer2_%d_%s.txt'%(epoch, args.model_name), models.gc2.mlp.dense1.weight.detach().cpu().numpy(), fmt = '%.5f')
    if epoch >= 0:
        write_json('../data_train_test_path/path_cnt_%s.json'%(dataset), num_of_paths)
        write_json('../log_result/rank_test_%d_%s.json'%(epoch, args.model_name), rank_save)
        write_json('../log_result/score_test_%d_%s.json'%(epoch, args.model_name), score_test_save)
        write_json('../log_result/score_%d_%s.json'%(epoch, args.model_name), score_save)
        write_json('../log_result/cnt_test_%d_%s.json'%(epoch, args.model_name), cnt_save)
        write_json('../log_result/path_test_%d_%s.json'%(epoch, args.model_name), path_save)
        import time
    with open('../log_result/%s.txt'%(args.model_name), 'a+') as f:
        f.write('Time: %s @ Mean Rank:%.2f/%d, Acc: %.4f, Acc@3:%.4f Acc@5:%.4f mrr:%.4f wup:%.4f '%(time.ctime(), np.mean(rank), l, float(correct/total_rank),acc3,acc5,mrr,wup_score))
        f.write('\n')
    print('@ Mean Rank:%.2f/%d, Acc: %.4f, Acc@3:%.4f Acc@5:%.4f mrr:%.4f wup:%.4f '%( np.mean(rank), l, float(correct/total_rank),acc3,acc5,mrr,wup_score))
    print('=> Testing: Mean Rank:%.2f/%d, Acc = %.3f'%(np.mean(rank), l, float(correct/total_rank)))
    # score_one={}(given item, output score), score_pred={}(given item, output all scores)
    '''
    attach the node in the test set and design new paths
    ## this case consider all nodes / this setting is not considered in evaluation part --> we only consider leaf nodes 
    '''
    correct = 0
    ranks = []
    wums = []
    item_, attach_ = select_node_to_attach(score_best) #Item: term in the test, attach_: term in the taxo 
    item_, attach_ = int(item_), int(attach_)
    total += 1
    tmp_rank = 1
    if attach_pos_gold[item_] == attach_:
        correct += 1
        ranks.append(1)
        wums.append(1)
    else:
        gold_position =attach_pos_gold[item_]
        if gold_position in score_one[str(item_)]:
            for term in score_one[str(item_)]:
                if score_one[str(item_)][term] > score_one[str(item_)][gold_position]:
                    tmp_rank += 1
            ranks.append(tmp_rank)
        wums.append(wum[gold_position, attach_])

    print('attach %d to %d, gold %d'%( int(item_), int(attach_), int(attach_pos_gold[item_])) )
    '''
    update the score one by one
    '''
    taxo_test.parent[item_] = attach_
    attach_term_list.append(item_)

    attach_path = []
    node = item_
    for _ in range(args.path_len):
        attach_path.append(node)
        if node not in taxo_test.parent:
            break
        node = taxo_test.parent[node]
    to_attached_number -= 1
    while to_attached_number > 0:
        if len(attach_path) == args.path_len:
            target_path = [[x] for x in attach_path]
            inner_path = []
            for _ in range(args.path_len-1):
                inner_path.extend(get_path_from_terms(target_path[_], target_path[_+1], inv_paths_index))
            #train_path = 
            pos_term = attach_path
            for t in test_hyn:
                if t in attach_term_list:
                    continue
                score_t = {}
               	test_node = int(t)
                train_path = get_path_from_attach_terms(t, target_path, args.path_len, inv_paths_index, synonym)
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
                dependency_paths = []
                dependency_freq = []
                for y in train_path:
                    if y == None:
                        dependency_paths.append([])
                        dependency_freq.append([])
                    elif len(dep_path[y[0]]["path"])>100: # sample 50 paths if there is too much dependency path
                        sample_idx, sample_freq = random_choice(dep_path[y[0]]["freq"], max_val= 100)
                        #print(dep_path[xx]["path"], dep_path[xx]["freq"], sample_idx, sample_freq)
                        dependency_paths.append([dep_path[y[0]]["path"][x] for x in sample_idx])
                        dependency_freq.append(sample_freq)
                    else:
                        dependency_paths.append(dep_path[y[0]]["path"])
                        dependency_freq.append(dep_path[y[0]]["freq"])
                score, embed_preds, paths_preds, taxi_preds = models(target_path = target_path, inner_path = inner, inner_freq = freq, attach = test_node, \
                        attach_path = dependency_paths, attach_freq = dependency_freq, device = device, compound = compound)
                if args.encode_prop == 'attn':
                    s=np.around(attn.astype(np.float64), decimals=2)
                #print(list([list(x) for x in s[1]]))
                if is_softmax:
                    score = F.softmax(score, dim = 1)
                score = score.detach().cpu().numpy().reshape(-1)
                for e,s in zip(pos_term, score):
                    if e not in score_t:
                        score_t[e] = [s]
                    else:
                        score_t[e].append(s)
                for attach_item in score_t:
                    if attach_item in score_pred[t]:
                        score_pred[t][attach_item].extend(score_t[attach_item])
                    else:
                        score_pred[t][attach_item] = score_t[attach_item]
                    if metric =='mean':
                        score_one[t][attach_item] =  scipy.stats.gmean(score_pred[t][attach_item]) if is_softmax else np.mean(score_pred[t][attach_item])
                    else:
                        score_one[t][attach_item] = np.max(score_pred[t][attach_item])

                best_val, best_attach = get_max_from_dict(score_one[t]) # attach: is the node in original taxo
                #score_one[t] = one_score
                score_best[t] = {'best_val': best_val, 'attach': best_attach} # t: term in test set
        item_, attach_ = select_node_to_attach(score_best, attach_term_list) #Item: term in the test, attach_: term in the taxo 
        item_, attach_ = int(item_), int(attach_)
        total += 1
        tmp_rank = 1
        if attach_pos_gold[item_] == attach_:
            correct += 1
            wums.append(1)
            ranks.append(1)
        else:
            gold_position = attach_pos_gold[item_]
            #print([gold_position], score_one[str(item_)].keys() )
            if gold_position in score_one[str(item_)]:
                for term in score_one[str(item_)]:
                    if score_one[str(item_)][term] > score_one[str(item_)][gold_position]:
                        tmp_rank += 1
                ranks.append(tmp_rank)
            wums.append(wum[gold_position, attach_])
        taxo_test.parent[item_] = attach_
        attach_term_list.append(item_)

        attach_path = []
        node = item_
        for _ in range(args.path_len):
            attach_path.append(node)
            if node not in taxo_test.parent:
                break
            node = taxo_test.parent[node]
        #print(item_,attach_, attach_pos_gold[item_])
        #print('attach %d to %d, gold %d'%( int(item_), int(attach_), int(attach_pos_gold[item_])) )
        if (int(item_), int(attach_)) in inv_paths_index:
            dep_num1 = len(dep_path[inv_paths_index[(int(item_), int(attach_))]]["path"])
        else:
            dep_num1 = 0
        if (int(item_), int(attach_pos_gold[item_]) ) in inv_paths_index:
            dep_num2 = len(dep_path[inv_paths_index[(int(item_), int(attach_pos_gold[item_]))]]["path"])
        else:
            dep_num2 = 0
        to_attached_number -= 1

    #write_json('../log_result/%s_%s_%s_%d_score.json'%(args.model_name, args.encode_dep, args.encode_prop, epoch), score_)
    if synonym:
        pass
    #print(ranks,wums)
    mrr = np.mean([1/x for x in ranks])
    acc3 = getacc(ranks, 3)
    acc5 = getacc(ranks, 5)
    print('=> Testing: Mean Rank:%d/%d, Acc = %.4f, Acc@3 = %.4f, Acc@5 = %.4f, wum = %.4f, mrr = %.4f'%(correct, total, float(correct/len(ranks)), acc3,acc5,np.mean(wums), mrr))

def getacc(rank, k):
    return len([x for x in rank if x<=k])/len(rank)
def get_max_from_dict(one_score):
    best_val = -10000
    best_attach = -10000
    for z in one_score:
        if one_score[z] > best_val:
            best_val = one_score[z]
            best_attach = z
    return best_val, best_attach

def select_node_to_attach(score_best, attached = []):
    best_val = -10000
    best_attach = -100000
    best_item = -10000
    for item in score_best:
        if int(item) not in attached:
            if score_best[item]["best_val"] > best_val:
                best_val = score_best[item]["best_val"]
                best_attach = score_best[item]["attach"]
                best_item = item
        else:
            pass
            #print(item, attached)
    return best_item, best_attach

def get_path_from_terms_with_synonym(src, tgt, inv_paths_index):
    paths = []
    for s in src:
        for t in tgt:
            if (s,t) in inv_paths_index:
                paths.append(inv_paths_index[(s,t)])
    return paths

def get_path_from_attach_terms(attach, original, length, inv_paths_index, synonym):
    #attach: a number
    #original: a 
    path = []
    if synonym:
        for o in original:
            if (attach, o[0]) in inv_paths_index:
                path.append([inv_paths_index[(attach, o[0])]])
            else:
                path.append(None)
    else:
        for o in original:
            path_tmp = []
            for x in o:
                if (attach, x) in inv_paths_index:
                    path_tmp.append(inv_paths_index[(attach, o[0])])      
            path.append(path_tmp if len(path_tmp)>0 else None)
    assert len(path) == length
    return path  

def get_path_from_terms(src, tgt, inv_paths_index):
    paths = []
    s = src[0]
    t = tgt[0]
    if (s,t) in inv_paths_index:
        paths.append(inv_paths_index[(s,t)])
    return paths   