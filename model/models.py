import torch 
import torch.nn as nn
from layers import Matcher, RNNEncoder
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
#import torchsnooper

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class model(nn.Module):
    def __init__(self, path_length= 3, hidden_size = 16, num_layers=1, num_terms = 599, embed_dim = 50, args=None, dropout = 0.3, bidirection = False, attach_path = False):
        super(model, self).__init__()
        self.path_length = path_length
        self.hidden_size = hidden_size # output frature size of RNN
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.pos_dim = 4 # args.posdim
        self.dep_dim = 5 # args.depdim
        self.dir_dim = 1 # args.dirdim
        self.num_terms = num_terms
        self.term_embedding = nn.Embedding(num_terms, self.embed_dim)
        self.dropout = dropout
        self.rnn_input_size = embed_dim + self.pos_dim + self.dep_dim + self.dir_dim
        if attach_path:
            self.path_embed_size = self.path_length * self.embed_dim + (2 * self.path_length - 1) * self.hidden_size
        else:
            self.path_embed_size = self.path_length * self.embed_dim + (self.path_length - 1) * self.hidden_size
        self.gc1 = RNNEncoder(term_embedding = self.term_embedding, input_size = self.rnn_input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, \
            dropout = 0, bidirection = False, pos_dim = 4, dep_dim = 5, dir_dim = 1)
        self.gc2 = Matcher(self.path_embed_size, self.embed_dim, self.path_length)


    def init_embed(self, embed):
        ### load embed for the term
        assert embed.shape[0] == self.num_terms and embed.shape[1] == self.embed_dim
        self.term_embedding.weight.data.copy_(torch.from_numpy(embed))

    #@torchsnooper.snoop()
    def forward(self, term, depend_path, attach_terms, device, cuda, attach_path = False, paths_attached = None):
        #M = len(term) # length of the path
        #assert M == self.path_length
        N = attach_terms.shape[0]
        #print(term)
        term_embed = self.term_embedding(term) # torch.tensor, size: (M * hidden_dim)
        #print(term_embed)
        attach_embed = self.term_embedding(attach_terms) # torch.tensor, size: (N * hidden_dim)
        term_embed = term_embed.reshape(1,-1) # 1 * (M*dim)

        for i, path in enumerate(depend_path):
            #print(path)
            if path != []:
                padding = self.gc1.padding(path, device, cuda)
                #print(path,padding)
                out = self.gc1(padding, len(path)) # all path's embeddings, size: 1 * feature_dim
                out = out.reshape(1,-1)
            else:
                #self.dependpath_embedding[0, ( i*self.hidden_size) : ((i+1) * self.hidden_size)] = torch.zeros(1, self.hidden_size)
                out = torch.zeros(1,self.hidden_size)
            if cuda != 0:
                out = out.to(device)
            term_embed = torch.cat([term_embed, out], dim = 1)
            #print(term_embed)
        if attach_path: # N candidate items, for each candidate item, there is a vector ro represent them
            #print(paths_attached, len(paths_attached), N)
            i = 0
            for path_attach in paths_attached:
                j = 0
                for p in path_attach:   
                    if p != []:
                        padding = self.gc1.padding(p, device, cuda)
                        #print(path,padding)
                        out = self.gc1(padding, len(p)) # all path's embeddings, size: 1 * feature_dim
                        out = out.reshape(1,-1)
                    else:
                        out = torch.zeros(1,self.hidden_size)
                    if cuda != 0:
                        out = out.to(device)
                    embed_p = out if j == 0 else torch.cat([embed_p, out], dim = 1)
                    j += 1
                embed_path = embed_p if i == 0 else torch.cat([embed_path, embed_p], dim = 0)
                i += 1
        #self.dependpath_embedding = self.dependpath_embedding.to(device)
        #path_rep = torch.cat([term_embed, self.dependpath_embedding], dim = 1) # 1*(M*dim + (M-1)*hidden_size)
        path_rep = term_embed
        #print(embed_path.shape, path_rep.shape)
        #print('path_rep:',path_rep)
        assert path_rep.requires_grad == True 
        path_rep = path_rep.repeat(N,1) # size: (N * (M*dim + (M-1)*hidden_size))
        #print path_rep.shape
        if attach_path:
            path_rep = torch.cat([path_rep, embed_path], dim = 1)
        output = self.gc2(path_rep, attach_embed)
        #print('output:',output)
        return output



class model2(nn.Module):
    def __init__(self, path_length= 3, hidden_size = 16, num_layers=1, num_terms = 599, embed_dim = 50, args=None, dropout = 0.3):
        super(model2, self).__init__()
        self.path_length = path_length
        self.hidden_size = hidden_size # output frature size of RNN
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.pos_dim = 4 # args.posdim
        self.dep_dim = 5 # args.depdim
        self.dir_dim = 1 # args.dirdim
        self.num_terms = num_terms
        self.term_embedding = nn.Embedding(num_terms, self.embed_dim)
        self.dropout = dropout
        self.rnn_input_size = embed_dim + self.pos_dim + self.dep_dim + self.dir_dim
        self.dependpath_embedding = torch.empty(1, (path_length - 1) * hidden_size)
        self.path_embed_size = self.path_length * self.embed_dim + (self.path_length - 1) * self.hidden_size
        #self.gc1 = RNNEncoder(term_embedding = self.term_embedding, input_size = self.rnn_input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, \
            #dropout = 0, bidirection = False, pos_dim = 4, dep_dim = 5, dir_dim = 1)
        #self.gc2 = Matcher(self.path_embed_size, self.embed_dim, self.num_classes)
        self.fc = nn.Linear(self.embed_dim*self.path_length, self.embed_dim)
        self.fc2= nn.Linear(self.embed_dim, 1 + self.path_length)

    def init_embed(self, embed):
        ### load embed for the term
        assert embed.shape[0] == self.num_terms and embed.shape[1] == self.embed_dim
        self.term_embedding.weight.data.copy_(torch.from_numpy(embed))

    #@torchsnooper.snoop()
    def forward(self, term, attach_terms, device):
        #M = len(term) # length of the path
        #assert M == self.path_length
        N = attach_terms.shape[0]
        #term = torch.LongTensor(term)
        #attach_term = torch.LongTensor(attach_terms)
        #term.to(device)
        #attach_terms.to(device)
        #print(term,attach_terms)
        term_embed = self.term_embedding(term) # torch.tensor, size: (M * hidden_dim)
        attach_embed = self.term_embedding(attach_terms) # torch.tensor, size: (N * hidden_dim)
        #for i, path in enumerate(depend_path):
        #    if path != []:
        #        out = self.gc1(self.gc1.padding(path, device), len(path)) # all path's embeddings, size: 1 * feature_dim
            #out, batch_size = pad_packed_sequence(out, batch_first=True, padding_value=0.0) # size: num_of_sentences * max(seq_len) * feature_dim
            #out = out[[x for x in range(batch_size.size(0))], batch_size - 1].contiguous() # size: num_of_sentences * feature_dim
            #out = torch.mean(b, dim = 0) #.unsqueeze(0) # 1 * feature_dim
        #        self.dependpath_embedding[0, ( i*out.size(0)) : ((i+1) * out.size(0))] = out
        #    else:
        #        self.dependpath_embedding[0, ( i*self.hidden_size) : ((i+1) * self.hidden_size)] = torch.zeros(1, self.hidden_size)
        term_embed = term_embed.reshape(1,-1) # 1 * (M*dim)
        #self.dependpath_embedding = self.dependpath_embedding.to(device)
        #path_rep = torch.cat([term_embed, self.dependpath_embedding], dim = 1) # 1*(M*dim + (M-1)*hidden_size)
        
        #assert path_rep.requires_grad == True 
        #path_rep = path_rep.repeat(N,1) # size: (N * (M*dim + (M-1)*hidden_size))
        term_repr = self.fc(term_embed).repeat(N, 1)
        #print(term_repr.shape)
        output = self.fc2(term_repr-attach_embed)
        #output = self.gc2(path_rep, attach_embed)
        return output





'''
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirection = False):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirection = bidirection
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional = self.bidirection)
        self.fc = nn.Linear(hidden_size, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * (2 if self.bidirection else 1), x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers * (2 if self.bidirection else 1), x.size(0), self.hidden_size).to(device)
        # permute transpose 
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item())))
'''