import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from earlystopping import EarlyStopping
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

tr = pd.read_csv('./data/transactions.csv')

def convert_to_seconds(date_str):
    if date_str[1] == ' ':
        total_days = int(date_str[0])
        date_str = date_str[2:]
    if date_str[2] == ' ':
        total_days = int(date_str[:2])
        date_str = date_str[3:]
    if date_str[3] == ' ':
        total_days = int(date_str[:3])
        date_str = date_str[4:]
        
    total_hours = int(date_str[:2]) if date_str[:2] != '00' else 0
    total_minutes = int(date_str[3:5]) if date_str[3:5] != '00' else 0
    total_seconds = int(date_str[6:8]) if date_str[6:8] != '00' else 0
    return total_days * 24 * 60 * 60 + total_hours * 60 * 60 + total_minutes * 60 + total_seconds

tr['tr_datetime'] = tr['tr_datetime'].apply(lambda x: convert_to_seconds(x))
tr.sort_values(by=['customer_id', 'tr_datetime'], axis=0, inplace=True)
tr.drop(columns=['tr_datetime', 'tr_type', 'term_id'], inplace=True)
tr.reset_index(drop=True, inplace=True)

encoder = OrdinalEncoder()
tr['mcc_code'] = encoder.fit_transform(tr['mcc_code'][:, None])
vocab_size = len(encoder.categories_[0])

ss = StandardScaler()
tr['amount'] = ss.fit_transform(tr['amount'][:, None])

df_mcc = pd.DataFrame(tr.groupby('customer_id')['mcc_code'].apply(list))
df_amount = pd.DataFrame(tr.groupby('customer_id')['amount'].apply(list))
tr_data = pd.merge(df_mcc, df_amount, left_index=True, right_index=True)

tr_data.reset_index(inplace=True)
df_target = pd.read_csv('./data/gender_train.csv')
data = pd.merge(df_target, tr_data, on='customer_id', how='inner')

history_len = 15
unique_cust = data.customer_id.values
num_train_id = 700
num_test_id = 300

train_data = data[:num_train_id]
test_data = data[num_train_id:num_train_id+num_test_id]

train_items = []
for cust_id in train_data.customer_id.values:
    avail_hist = len(train_data.loc[train_data.customer_id == cust_id, 'mcc_code'].values[0])
    for i in range(avail_hist//history_len):
        d = {'customer_id': cust_id, 
             'mcc_code': train_data.loc[train_data.customer_id == cust_id, 'mcc_code'].values[0][i*history_len:(i+1)*history_len],
             'amount': train_data.loc[train_data.customer_id == cust_id, 'amount'].values[0][i*history_len:(i+1)*history_len],
             'gender': train_data.loc[train_data.customer_id == cust_id, 'gender'].values[0]}  
        train_items.append(d)
    
test_items = []
for cust_id in test_data.customer_id.values:
    avail_hist = len(test_data.loc[test_data.customer_id == cust_id, 'mcc_code'].values[0])
    for i in range(avail_hist//history_len):
        d = {'customer_id': cust_id, 
             'mcc_code': test_data.loc[test_data.customer_id == cust_id, 'mcc_code'].values[0][i*history_len:(i+1)*history_len],
             'amount': test_data.loc[test_data.customer_id == cust_id, 'amount'].values[0][i*history_len:(i+1)*history_len],
             'gender': test_data.loc[test_data.customer_id == cust_id, 'gender'].values[0]}        
        test_items.append(d)


train_mcc_items = [train_items[i]['mcc_code'] for i in range(len(train_items))]
train_amount_items = [train_items[i]['amount'] for i in range(len(train_items))]
train_target = [train_items[i]['gender'] for i in range(len(train_items))]

test_mcc_items = [test_items[i]['mcc_code'] for i in range(len(test_items))]
test_amount_items = [test_items[i]['amount'] for i in range(len(test_items))]
test_target = [test_items[i]['gender'] for i in range(len(test_items))]

class TransactinReader(Dataset):
    def __init__(self, mcc_items, amount_items, target):
        super(TransactinReader, self).__init__()
        self.mcc_items = mcc_items
        self.amount_items = amount_items
        self.target = target
        
    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.mcc_items[index, :], self.amount_items[index, :], self.target[index]

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)    
    
 
class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class TransformerTransact(nn.Module):
    def __init__(self, vocab_mcc_size, device):
        super(TransformerTransact, self).__init__()
        self.vocab_mcc_size = vocab_mcc_size
        self.device = device
        self.max_seq_len = 15
        
        ntokens = self.vocab_mcc_size  # size of vocabulary
        emsize = 256  # embedding dimension
        d_hid = 128  # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2  # number of heads in nn.MultiheadAttention
        dropout = 0.2  # dropout probability
        self.transformer = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)
        
        # Input Transformer [seq_len, batch_size]
        # Input mask [seq_len, seq_len]
        # output of Transformer [seq_len, batch_size, ntoken]
        
        self.linear1 = nn.Linear(self.vocab_mcc_size, 64) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 2) 
        
    def forward(self, x_mcc):
        x_perf = self.transformer(x_mcc.transpose(1, 0), torch.ones((x_mcc.shape[1], x_mcc.shape[1])).to(self.device)).transpose(1, 0) # [batch_size, seq_len,  num_tokens]
        x_aver = torch.mean(x_perf, dim=1) # [batch_size, num_tokens]

        x1 = self.linear1(x_aver)
        x1 = self.relu(x1)
        x_final = self.linear2(x1)

        return x_final

def time_memory_consumption(data, path):
    with open(path, 'a') as f:
        f.write(data)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

num_epochs = 300

net = TransformerTransact(vocab_size, device).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
cross_entr_loss = nn.CrossEntropyLoss()

print('Start preparing data')
train_dataset = TransactinReader(np.array(train_mcc_items),
                                 np.array(train_amount_items),
                                 np.array(train_target))


test_dataset = TransactinReader(np.array(test_mcc_items),
                                 np.array(test_amount_items),
                                 np.array(test_target))

train_dataloader = DataLoader(train_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

print('Finish preparing data')

early_stopping = EarlyStopping(patience=3, verbose=True, path='./model_checkpoints/transformer.pth')

i = 0
for epoch in range(1, num_epochs+1):
    net.train(True)
    epoch_train_loss = 0
    for batch_mcc, batch_amount, batch_target in tqdm(train_dataloader):
        batch_mcc, batch_amount, batch_target = \
        batch_mcc.to(device), batch_amount.to(device), batch_target.to(device)
        optimizer.zero_grad()
        predicted_probs = net(batch_mcc.int())
        if i == 0:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         profile_memory=True,
                         record_shapes=True) as prof:
                with record_function("model_training"):
                    predicted_probs = net(batch_mcc.int())
#             print(prof.key_averages().table())
            time_memory_consumption(prof.key_averages().table(), './training_{}.txt'.format('transformer'))
        else:
            predicted_probs = net(batch_mcc.int())
        loss = cross_entr_loss(predicted_probs, batch_target.long())
        epoch_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        i += 1

    print(f'Epoch {epoch}/{num_epochs} || Train loss {epoch_train_loss}')
    net.train(False)
    test_probs = []
    test_labels = []
    y_target = []
    print('Testing...')
    for batch_mcc, batch_amount, batch_target in tqdm(test_dataloader):
        batch_mcc, batch_amount, batch_target = \
        batch_mcc.to(device), batch_amount.to(device), batch_target.to(device)
        predicted_probs = torch.softmax(net(batch_mcc.int()), dim=-1).cpu().detach().numpy()[:, 1]
        predicted_labels = np.argmax(torch.softmax(net(batch_mcc.int()), dim=-1).cpu().detach().numpy(), axis=1)
        test_probs.extend(predicted_probs)
        test_labels.extend(predicted_labels)
        y_target.extend(batch_target.cpu().detach().numpy())

    accuracy_test = accuracy_score(y_target, test_labels)
    scheduler.step(-accuracy_test)
    print(f'Epoch {epoch}/{num_epochs} || Test Accuracy {accuracy_test}')

    early_stopping(-accuracy_test, net)
    if early_stopping.early_stop:
        print('Early stopping')
        break

            
            
            
            
        
    
    
    
    
    
    
    
    
    
    
    
