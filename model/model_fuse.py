import torch 
import torch.nn as nn
from layers_path import  RNNEncoder, LinearEncoder, AttentionFuseLayer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class model(nn.Module):
    def __init__(self, args, path_length= 3, hidden_size = 16, num_layers=1, num_terms = 599, parameters = [], embed_dim = 50, embed_context = 50, num_classes = 4, dropout = 0.3, bidirection = False, taxi_feature = False):
        super(model, self).__init__()
        self.path_length = path_length
        self.hidden_size = hidden_size # output frature size of RNN
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.embed_context = embed_context
        self.pos_dim = 4 # args.posdim
        self.dep_dim = 5 # args.depdim
        self.dir_dim = 1 # args.dirdim
        self.hierar_size = 3 # up same and down
        self.path_param = parameters
        self.output_dep_dim = 60
        self.attn_dim = 128
        self.num_terms = num_terms
        self.term_embedding = nn.Embedding(self.num_terms, self.embed_dim)
        self.context_embedding = nn.Embedding(self.num_terms, self.embed_context)
        self.vacant_path = nn.Embedding(1, self.hidden_size)
        self.num_classes = num_classes
        self.dropout = dropout
        self.encode_dep = args.encode_dep #'rnn' or 'transform'
        self.encode_prop = args.encode_prop # 'rnn' or 'selfattn' or None
        self.method = args.method
        self.atten = False
        self.taxi_feature = taxi_feature
        #self.minus_only = args.minus_only
        self.feature_dim = 50
        self.use_context = args.load_gcn
        self.load_score = args.load_score
        if args.load_score:
            self.gnn_score = args.gnn_score
        
        self.rnn_input_size = embed_dim + self.pos_dim + self.dep_dim + self.dir_dim

        #if self.encode_dep == 'rnn':
        self.gc1 = RNNEncoder(term_embedding = self.term_embedding, input_size = self.rnn_input_size, hidden_size = self.hidden_size, path_length = self.path_length,\
            path_param = self.path_param, num_layers = num_layers, dropout = self.dropout, bidirection = False, pos_dim = 4, dep_dim = 5, dir_dim = 1)

        if self.use_context:
            linear_input_size = self.path_length * self.hidden_size + (self.path_length )* (self.embed_context) + self.embed_dim
            path_input_size =  self.path_length * self.hidden_size 
            if args.load_score:
                emb_input_size = self.path_length
            else:
                emb_input_size = self.path_length * self.embed_context + self.embed_dim 

        else:
            linear_input_size = self.path_length * self.hidden_size + (self.path_length )* self.embed_dim
            path_input_size =  self.path_length * self.hidden_size
            emb_input_size = self.path_length * self.embed_dim

        
        self.gc2 = LinearEncoder(input_size = path_input_size, output_size = self.feature_dim, num_layers=args.layers, dropout = self.dropout)
        self.gc3 = LinearEncoder(input_size = emb_input_size, output_size = self.feature_dim, num_layers=args.layers, dropout = self.dropout)
        self.gc4 = LinearEncoder(input_size = 7 * self.path_length, output_size = self.feature_dim// 4, num_layers=args.layers, dropout = self.dropout)
        self.pred2 = LinearEncoder(input_size = self.feature_dim, output_size = self.path_length + 1, num_layers=args.layers, dropout = self.dropout)
        self.pred3 = LinearEncoder(input_size = self.feature_dim, output_size = self.path_length + 1, num_layers=args.layers, dropout = self.dropout)
        self.pred4 = LinearEncoder(input_size = self.feature_dim // 4, output_size = self.path_length + 1, num_layers=args.layers, dropout = self.dropout)
        self.attn = AttentionFuseLayer(self.feature_dim, self.feature_dim, self.feature_dim // 4 , (self.path_length + 1)*3, self.path_length + 1, 3)
        print(self.gc1, self.gc2, self.gc3)

    def insert_taxi_feature(self, nfd_norm, gene_diff):
        if self.taxi_feature:
            self.nfd_norm =  nfd_norm
            self.gene_diff = gene_diff

    def insert_string_feature(self, suffix, contain, ld, lcs, ends):
        if self.taxi_feature:
            self.suffix = suffix
            self.contain = contain
            self.ld = ld
            self.lcs = lcs
            self.ends = ends


    def init_embed(self, embed, context = []):
        ### load embed for the term
        assert embed.shape[0] == self.num_terms and embed.shape[1] == self.embed_dim
        self.term_embedding.weight.data.copy_(torch.from_numpy(embed))
        if context != []:
        	self.context_embedding.weight.data.copy_(torch.from_numpy(context))


    def gen_taxi_feature(self, attach, target_path, inverse = False):
        target_term = [x[0] for x in target_path]
        if not inverse:
            return [[   
                        self.nfd_norm[attach, y], 
                        self.gene_diff[attach, y],
                        self.suffix[attach, y],
                        self.contain[attach, y],
                        self.ld[attach, y],
                        self.lcs[attach, y],
                        self.ends[attach, y],
                    ] 
                    for y in target_term]
        else:
            return [[   
                        self.nfd_norm[y, attach], 
                        self.gene_diff[y, attach],
                        self.suffix[y, attach],
                        self.contain[y, attach],
                        self.ld[y, attach],
                        self.lcs[y, attach],
                        self.ends[y, attach],
                    ] 
                    for y in target_term]

    def gen_context_embed(self, target_path, device):
        embed = []
        idx = [t[0] for t in target_path]
        ids = torch.LongTensor(idx).to(device)
        embed = self.context_embedding(ids)

        return embed

    def gen_term_embed(self, target_path, device, synonym = True):
        embed = []
        idx = [t[0] for t in target_path]
        ids = torch.LongTensor(idx).to(device)
        embed = self.term_embedding(ids)

        return embed

    def forward(self, target_path, inner_path, inner_freq, attach, \
                    attach_path, attach_freq, device, compound = True):
        '''
        target_path = [[1,2],[2,3],[3],[4]]
        inner_path = [1,2,3,...(path_id)]
        attach = []
        path = []
        '''
        #print(target_path)
        M = len(target_path)
        # attach node and target_path

        # generate taxi feature for all three term paths
        
        attach_embed = self.term_embedding(torch.LongTensor([attach]).to(device)) # torch.size 1*dim
        if self.use_context:
            embed = self.gen_context_embed(target_path, device)
        else:
            embed = self.gen_term_embed(target_path, device, synonym = False) # term embedding (terms on the path). size: (path_length) * (len_embedding)

        assert len(attach_path) == M
        dep_path_embed = []
        if self.encode_dep != 'none':
            for idx, (freq, path) in enumerate(zip(attach_freq, attach_path)):
                if path == []:
                    out = self.vacant_path(torch.LongTensor([0]).to(device))
                else:
                    if self.encode_dep == 'rnn' :
                        out = self.gc1(self.gc1.padding(path, device), len(path), freq, device, idx, attn = False) 
                    elif self.encode_dep == 'attn':
                        embed_dep, key_padding, src_mask, end_id, batch_id = self.gc1.masking(path, device)
                        out = self.gc1(embed_dep, key_padding, src_mask, end_id, batch_id, freq, device)
                    out = out.unsqueeze(0)
                dep_path_embed.append(out)
            dep_path_embed = torch.cat(dep_path_embed, dim = 0) #[1, path_len*hidden_dim]
        
        dep_path_embed = dep_path_embed.reshape(1, -1)
        z = attach_embed.reshape(1,-1)
        if self.use_context:
            if self.load_score:
                s = [self.gnn_score[attach, z[0]] for z in target_path]
                #print(s)
                y = torch.FloatTensor(s).reshape(1,-1).to(device)
            else:
                y = torch.cat([embed.reshape(1,-1), attach_embed], dim = 1).reshape(1, -1)
        else:
            y = torch.cat([embed - attach_embed.repeat(M, 1)]).reshape(1,-1)
        taxi_pattern = torch.FloatTensor(self.gen_taxi_feature(attach, target_path)).reshape(1,-1).to(device)
        path_feats, emb_feats, taxi_feats = self.gc2(dep_path_embed), self.gc3(y), self.gc4(taxi_pattern)
        paths_pred, emb_pred, taxi_pred = self.pred2(path_feats), self.pred3(emb_feats), self.pred4(taxi_feats)
        pred = torch.cat([paths_pred, emb_pred, taxi_pred], dim = 0).reshape(3, -1)
        pred_ensem = self.attn( pred, path_feats, emb_feats, taxi_feats)

        return pred_ensem.unsqueeze(0), paths_pred.unsqueeze(0), emb_pred.unsqueeze(0), taxi_pred.unsqueeze(0)