import torch 
import torch.nn as nn
from layers import Matcher, RNNEncoder
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class top_model(nn.Module):
    def __init__(self, args, path_length= 3, hidden_size = 16, num_layers=1, num_terms = 599, embed_dim = 50, num_classes = 4, dropout = 0.3, bidirection = False):
        super(model, self).__init__()
        self.path_length = path_length
        self.hidden_size = hidden_size # output frature size of RNN
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.pos_dim = 4 # args.posdim
        self.dep_dim = 5 # args.depdim
        self.dir_dim = 1 # args.dirdim
        self.term_embedding = nn.Embedding(num_terms, self.embed_dim)
        self.num_classes = num_classes
        self.dropout = dropout
        self.encode_dep = 'rnn' #'rnn' or 'transform'
        self.encode_prop = 'rnn' # 'rnn' or 'attn'
        self.atten = False
        
        self.rnn_input_size = embed_dim + self.pos_dim + self.dep_dim + self.dir_dim
        self.dependpath_embedding = torch.empty(1, (num_term - 1) * hidden_size *(2 if bidirection else 1) )
        self.path_embed_size = self.path_length * self.embed_dim + (self.path_length - 1) * self.hidden_size
        if self.encode_prop == 'rnn':
            self.gc1 = RNNEncoder(term_embedding = self.term_embedding, input_size = self.rnn_input_size, self.hidden_size, num_layers, \
                num_classes = self.num_classes, dropout = self.dropout, bidirection = False, pos_dim = 4, dep_dim = 5, dir_dim = 1)
        else:
            self.gc1 = RNNEncoder(term_embedding = self.term_embedding, input_size = self.rnn_input_size, self.hidden_size, num_layers, \
                num_classes = self.num_classes, dropout = self.dropout, bidirection = False, pos_dim = 4, dep_dim = 5, dir_dim = 1)
        if self.encode_dep == 'rnn':
            self.gc1 = None
        else:
            self.gc2 = None
        self.gc2 = Matcher(self.path_embed_size, self.embed_dim, self.num_classes)


    def init_embed(self, embed):
        ### load embed for the term
        assert embed.shape[0] == num_terms and embed.shape[1] == self.embed_dim
        self.term_embedding.weight.data.copy_(torch.from_numpy(embed))

    def gen_term_embed(self, target_path, device, synonym = False):
        embed = []
        for i,term in enumerate(target_path):
            if synonym:
                idx = torch.LongTensor(term).to(device)
                embed_term = torch.mean(self.term_embedding(idx), dim = 0).unsqueeze(0)
            else:
                idx = torch.LongTensor(term[0]).to(device)
                embed_term = self.term_embedding(idx).unsqueeze(0)
            embed.append(embed_term)
            #embed = embed_term if i == 0 else torch.cat([embed, embed_term], dim = 0)
        #assert embed.shape[0] == len(target_path)
        embed = torch.cat(embed, dim = 0)
        return embed

    #def 
    def forward(self, target_path = target_path, inner_path = inner, inner_freq = freq, attach = pos_sample, \
                    attach_path = dependency_paths, attach_freq = dependency_freq, device = device):

    #def forward(self, target_path, inner_path, attach, path, device):
        '''
        target_path = [[1,2],[2,3],[3],[4]]
        inner_path = [1,2,3,...(path_id)]
        attach = []
        path = []
        '''
        M = len(target_path)
        embed = self.gen_term_embed(target_path, device, synonym = False) # term embedding (terms on the path). size: (path_length) * (len_embedding)
        attach_embed = self.term_embedding(torch.LongTensor(attach).to(device)) # torch.size 1*dim
        attach_embed = attach_embed.repeat(M, 1)
        print(embed, attach_embed)

        #target_path = torch.mean(self.term_embedding(torch.LongTensor(inner_path).to(device)), dim = 0)



    def forward(self, term, depend_path, path_terms, device):
        M = len(term)
        assert M == self.path_length
        N = len(path_terms)
        term = torch.LongTensor(term)
        #attach_terms = torch.LongTensor(attach_terms)
        path_terms.to(device)
        term.to(device)

        term_embed = self.term_embedding(term)
        for t in path_terms:
            term = torch.LongTensor(t)





