import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#HIDDEN_DIM = 60
#LEMMA_DIM = 50
#POS_DIM = 4
#DEP_DIM = 5
#DIR_DIM = 1

class LinearEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, dropout = 0.4):
        super(LinearEncoder, self).__init__()
        
        self.num_layers = num_layers
        #self.bidirection = bidirection
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = ((self.input_size +self.output_size) // 4)
        self.dropout = dropout
        self.mlp = nn.Sequential()
        if self.num_layers == 1:
            self.mlp.add_module("dropout0", nn.Dropout(self.dropout))
            self.mlp.add_module("dense0", nn.Linear(self.input_size, self.output_size))
        
        elif self.num_layers == 2:
            self.mlp.add_module("dense0", nn.Linear(self.input_size, self.hidden_size))
            self.mlp.add_module("dropout0", nn.Dropout(self.dropout))
            self.mlp.add_module("relu0", nn.ReLU())
            self.mlp.add_module("dense1", nn.Linear(self.hidden_size, self.output_size))
        elif self.num_layers > 2:
            self.mlp.add_module("dense0", nn.Linear(self.input_size, self.hidden_size))
            self.mlp.add_module("dropout0", nn.Dropout(self.dropout))
            for i in range(1, self.num_layers-1):
                self.mlp.add_module("dense%d"%(i), nn.Linear(self.hidden_size, self.hidden_size))
                self.mlp.add_module("dropout%d"%(i), nn.Dropout(self.dropout))
                self.mlp.add_module("relu%d"%(i), nn.ReLU())
            self.mlp.add_module("dense%d"%(self.num_layers-1), nn.Linear(self.hidden_size, self.output_size))
        #print(self.mlp)
    def forward(self, x):
        x = self.mlp(x)
        #print(self.mlp.dense0.weight.detach().cpu())
        return x.squeeze(0)



class RNNEncoder(nn.Module):
    def __init__(self, term_embedding, input_size, hidden_size, path_length, path_param, num_layers=1, dropout = 0.2, bidirection = False, pos_dim = 4, dep_dim = 5, dir_dim = 1):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirection = bidirection
        self.lemma_embed = term_embedding  # lemma
        self.path_param = path_param
        self.path_length = path_length
        self.pos_embed = nn.Embedding(int(path_param["pos"]), pos_dim)     # part of speech 
        self.dep_embed = nn.Embedding(int(path_param["dep"]), dep_dim)  # dependency label
        self.dir_embed = nn.Embedding(int(path_param["dir"]), dir_dim)      # dependency direction
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first=True, bidirectional = self.bidirection, dropout = self.dropout)
        self.attn = AttentionLayer(input_size = self.hidden_size, hidden_size = self.hidden_size * 2, atten_size = self.path_length)

    def forward(self, x, batch_size, attach_freq, device, idx, attn = False, initial = True):
        # Set initial states
        #print(attn)
        if not initial:
            pass
            #h0 = torch.zeros(self.num_layers * (2 if self.bidirection else 1), x.size(0), self.hidden_size).to(device) # 2 for bidirection 
            #c0 = torch.zeros(self.num_layers * (2 if self.bidirection else 1), x.size(0), self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.num_layers * (2 if self.bidirection else 1), batch_size, self.hidden_size).to(device) # 2 for bidirection 
            c0 = torch.zeros(self.num_layers * (2 if self.bidirection else 1), batch_size, self.hidden_size).to(device)            
        # permute transpose 
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*bidirection)
        out, batch_size = pad_packed_sequence(out, batch_first=True)
        out = out[[x for x in range(batch_size.size(0))], batch_size - 1].contiguous() # size: num_of_sentences * feature_dim
        #print(out, out.shape, attach_freq)

        attach_freq = torch.FloatTensor(attach_freq).reshape(-1, 1).to(device)
        out = out * attach_freq / torch.sum(attach_freq)
        if not attn:
            out = torch.sum(out, dim = 0) #.unsqueeze(0) # 1 * feature_dim
        else:
            out = self.attn(out, idx)
        # Decode the hidden state of the last time step
        #out = self.fc(out[-1, :]) #out: tensor of shape (seq_length, num_classes)
        return out

    def padding(self, path, device):
        # path: a list with different length 
        # e.g [ [[1,2,3,4],[1,2,4,5],[2,3,4,5]], [[1,2,3,4],[5,6,2,3],[3,4,5,3],[2,3,4,5]] ]
        embed_dep = []
        for p in path:
            p = torch.LongTensor(np.array(p).T).to(device)
            lemma, pos, dep, drc = p[0], p[1], p[2], p[3]
            #p = torch.LongTensor(p)
            lemma_emb = self.lemma_embed(lemma)
            pos_emb = self.pos_embed(pos)
            dep_emb = self.dep_embed(dep)
            dir_emb = self.dir_embed(drc)
            embed_dep.append( torch.cat([lemma_emb,pos_emb,dep_emb,dir_emb], dim = 1) )
        embed_dep.sort(key=lambda x: len(x), reverse=True)
        data_length = torch.LongTensor([len(embed) for embed in embed_dep]).to(device)
        embed_dep = pad_sequence(embed_dep, batch_first = True)
        embed_dep = pack_padded_sequence(embed_dep, data_length, batch_first = True)
        return embed_dep

class AttentionFuseLayer(nn.Module):

    def __init__(self, input_size_path, input_size_emb, input_feat_emb, input_pred_emb, hidden_size, atten_size):
        super(AttentionFuseLayer, self).__init__()
        self.hidden_size = hidden_size
        #self.output_size = output_size    
        self.gamma = 5
        self.atten_size = atten_size
        self.input_pred_emb = input_pred_emb
        self.attn1 = nn.Linear(input_size_path, self.hidden_size)
        self.attn2 = nn.Linear(input_size_emb, self.hidden_size)
        self.attn3 = nn.Linear(input_feat_emb, self.hidden_size)
        self.attn4 = nn.Linear(input_pred_emb, input_pred_emb*2)

        self.weight = nn.Parameter(torch.FloatTensor(input_pred_emb*2, self.atten_size))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, pred, x_path, x_emb, x_feat):
        #print(pred.shape,x_path.shape )
        pred_soft = F.softmax(pred, dim = 1)
        
        #print('pred',pred,  torch.mean(pred_soft, dim = 0))
        pred_shift = pred_soft - torch.mean(pred_soft, dim = 0)
        pred_shift = torch.exp(-self.gamma * pred_shift)
        y4 = self.attn4(pred_shift.reshape(1, -1))

        alpha = F.softmax(torch.mm(y4, self.weight), dim = 0).reshape(-1, 1) # 3 * 1
        #print(alpha, pred)
        out = torch.sum(pred * alpha, dim = 0) # dim = 0 equal to mean pooling, we tried to use attention here (dim=1) but not found significant performance gain 
        
        #out = y1+y2+y3
        return out.view(-1)


class AttentionLayer(nn.Module):

    def __init__(self, input_size, hidden_size, atten_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        #self.output_size = output_size    
        self.atten_size = atten_size
        self.attn = nn.Linear(input_size, self.hidden_size)
        self.weight = nn.Parameter(torch.FloatTensor(self.hidden_size, self.atten_size))
        nn.init.xavier_uniform_(self.weight)
        #print('weight:',self.weight)

    def forward(self, x, i):
        y = (self.attn(x)) # num_term * hidden_size
        #print('y:',y)
        #i =  torch.FloatTensor([i])
        alpha = F.softmax(torch.mm(y, self.weight[:,i].reshape(-1,1))/0.01, dim = 0)
        #print('alpha:',alpha, alpha.shape)
        #print(alpha.detach().cpu().reshape(-1)) #self.attn.weight.detach().cpu())
        out = torch.sum(x * alpha, dim = 0)
        #print(out.shape, out)
        return out


