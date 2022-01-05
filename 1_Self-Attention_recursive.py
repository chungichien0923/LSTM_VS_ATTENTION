# %% [markdown]
# # 匯入函式庫

# %%
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


# %% [markdown]
# # 匯入資料與前處理

# %%
diff_log_data = pd.read_csv('./recursive_linear_residue.csv', index_col=0)
diff_log_data = diff_log_data.astype('float64')
diff_log_data.index = pd.to_datetime(diff_log_data.index)

diff_log_data

# %% [markdown]
# # CPU/GPU、自定義資料集、模型、訓練函數

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
def SetSeed(myseed):
    # Python random module
    random.seed(myseed)
    # Numpy
    np.random.seed(myseed)
    # Torch
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(myseed)
        torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %%
class TimeSeriesDataset(Dataset):
    def __init__(self, X, WindowSize):
        X = np.expand_dims(X, 1)
        self.X = X.astype(np.float64)
        self.WindowSize = WindowSize
        
    def __len__(self):
        return len(self.X) - self.WindowSize

    def __getitem__(self, idx):
        return (self.X[idx:idx+self.WindowSize], self.X[idx+self.WindowSize])
        # return (X = [seqs, features], y)

# %%
class Self_Attention(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.num_layers = num_layers
        
        self.Input_First_HiddenLayer = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)
        self.Query_Key_Value_1 = nn.ModuleList([nn.Linear(1, hidden_size), nn.Linear(1, hidden_size), nn.Linear(1, hidden_size)])

        self.Second_And_Following_HiddenLayer = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1) for i in range(num_layers - 1)])
        self.List_of_Query_Key_Value = nn.ModuleList([nn.ModuleList([nn.Linear(hidden_size, hidden_size) for q_k_v in range(3)]) for i in range(num_layers - 1)])

        self.OutputLayer = nn.Linear(hidden_size, 1)

    def forward(self, input):
        # input.shape = [BatchSize, WindowSize, 1]
        input = input.permute(1, 0, 2)
        # input.shape = [WindowSize, BatchSize, 1]
        Query1 = self.Query_Key_Value_1[0](input)
        Key1 = self.Query_Key_Value_1[1](input)
        Value1 = self.Query_Key_Value_1[2](input)
        hidden, _ = self.Input_First_HiddenLayer(Query1, Key1, Value1)
        # hidden.shape = [WindowSize, BatchSize, HiddenSize]
        if self.num_layers > 1:
            for i, Second_And_Following_HiddenLayer in enumerate(self.Second_And_Following_HiddenLayer):
                Query = self.List_of_Query_Key_Value[i][0](hidden)
                Key = self.List_of_Query_Key_Value[i][1](hidden)
                Value = self.List_of_Query_Key_Value[i][2](hidden)
                hidden, _ = Second_And_Following_HiddenLayer(Query, Key, Value)
        # hidden.shape = [WindowSize, BatchSize, HiddenSize]
        hidden = hidden[-1]
        # hidden.shape = [BatchSize, HiddenSize]
        output = self.OutputLayer(hidden)
        
        return output

# %%
def train_under_config_and_evaluating_at_num_epochs_list(forex_data,
                                                         length_input_sequence,
                                                         num_epochs_list,
                                                         num_hidden_layers,
                                                         num_hidden_sizes,
                                                         batch_sizes,
                                                         device):
    '''
    forex_data,
    length_input_sequence,
    num_epochs_list,
    learning_rate,
    num_hidden_layers,
    num_hidden_sizes,
    batch_sizes,
    device
    '''
    # setseed
    SetSeed(9527)
    
    # dataset_train
    training_data = forex_data.loc['1981-01-01':'2008-12-31']
    training_dataset = TimeSeriesDataset(training_data, length_input_sequence)
    # dataset_valid
    validation_start_index = len(forex_data.loc['1981-01-01':'2008-12-31']) - length_input_sequence
    validation_end_index = len(forex_data.loc['1981-01-01':'2016-12-31'])
    validation_data = forex_data[validation_start_index:validation_end_index]
    validation_dataset = TimeSeriesDataset(validation_data, length_input_sequence)
    
    # dataloader_train
    training_dataloader = DataLoader(training_dataset, batch_size=batch_sizes, shuffle=True)
    # dataloader_valid
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_sizes, shuffle=False)
    
    # model
    model = Self_Attention(num_hidden_layers, num_hidden_sizes).double()
    # criterion & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    # training & evaluating
    min_valid_loss_at_best_epoch = 100000
    min_valid_loss_epoch = 0
    for epoch in tqdm(range(num_epochs_list[-1])):
        # training
        model.to(device)
        model.train()
        for X, y in training_dataloader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            ypred = model(X)
            loss = criterion(ypred, y)
            loss.backward()
            optimizer.step()
            
        # evaluating
        if (epoch + 1) in num_epochs_list:
            model.eval()
            valid_loss = 0
            len_valid = 0
            for X, y in  validation_dataloader:
                len_valid += len(X)
                X, y = X.to(device), y.to(device)
                with torch.no_grad():
                    ypred = model(X)
                    loss = criterion(ypred, y)
                valid_loss += loss.detach().cpu().item() * len(X)
            valid_loss = valid_loss / len_valid
            if valid_loss < min_valid_loss_at_best_epoch:
                min_valid_loss_at_best_epoch = valid_loss
                min_valid_loss_epoch = epoch + 1
    
    return min_valid_loss_epoch, min_valid_loss_at_best_epoch

# %%
def find_optimum_config_under_specific_input_length(forex_data,
                                                    length_input_sequence,
                                                    num_epochs_list,
                                                    num_hidden_layers_list,
                                                    num_hidden_sizes_list,
                                                    batch_sizes_list,
                                                    device=device):
    min_valid_loss = 100000
    min_valid_config_under_specific_input_length = ()
    for num_hidden_layers in num_hidden_layers_list:
        for num_hidden_sizes in num_hidden_sizes_list:
            for batch_sizes in batch_sizes_list:
                print('\nTraining under config:', (num_hidden_layers, num_hidden_sizes, batch_sizes))
                min_valid_loss_epoch, min_valid_loss_at_best_epoch = train_under_config_and_evaluating_at_num_epochs_list(forex_data,
                                                                                                                          length_input_sequence,
                                                                                                                          num_epochs_list,
                                                                                                                          num_hidden_layers,
                                                                                                                          num_hidden_sizes,
                                                                                                                          batch_sizes,
                                                                                                                          device)
                if min_valid_loss_at_best_epoch < min_valid_loss:
                    min_valid_loss = min_valid_loss_at_best_epoch
                    min_valid_config_under_specific_input_length = (num_hidden_layers, num_hidden_sizes, batch_sizes, min_valid_loss_epoch)

                    print('\nvalid_loss improve to',
                            min_valid_loss,
                            'under config:',
                            min_valid_config_under_specific_input_length,
                            '(num_hidden_layers, num_hidden_sizes, batch_sizes, num_epochs)')

    print('\nmin valid loss config under specific input length',
          length_input_sequence,
          'is:',
          min_valid_config_under_specific_input_length,
          '(num_hidden_layers, num_hidden_sizes, batch_sizes, num_epochs)',
          'and the valid loss is:',
          min_valid_loss)
    return min_valid_config_under_specific_input_length, min_valid_loss

# %%
# # configuration
length_input_sequence_list = [5, 10, 20, 60]
num_epochs_list = [5, 10, 15, 20, 25, 30]
num_hidden_layers_list = [1, 2, 3, 4]
num_hidden_sizes_list = [25, 50, 100, 200, 400, 800, 1600]
batch_sizes_list = [16, 32, 64, 128, 256]

# %% [markdown]
# # cad

# %%
cad_data = diff_log_data.iloc[:,0]
print(cad_data.loc['1981-01-01':'2008-12-31'])
print(cad_data.loc['2009-01-01':'2016-12-31'])
print(cad_data.loc['2017-01-01':'2020-12-31'])

# %%
# random walk mse(validation)
se = cad_data.loc['2009-01-01':'2016-12-31'] ** 2
mse = sum(se) / len(se)
mse

# %%
cad_min_valid_loss = 100000
cad_min_valid_length_input_sequence = 0
cad_min_valid_config_under_specific_input_length = ()
for length_input_sequence in tqdm(length_input_sequence_list):
    optimum_config_under_specific_input_length, valid_loss_under_specific_input_length = find_optimum_config_under_specific_input_length(cad_data, length_input_sequence, num_epochs_list, num_hidden_layers_list, num_hidden_sizes_list, batch_sizes_list)
    if valid_loss_under_specific_input_length < cad_min_valid_loss:
        cad_min_valid_loss = valid_loss_under_specific_input_length
        cad_min_valid_length_input_sequence = length_input_sequence
        cad_min_valid_config_under_specific_input_length = optimum_config_under_specific_input_length

# %% [markdown]
# # aud

# %%
aud_data = diff_log_data.iloc[:,1]
print(aud_data.loc['1981-01-01':'2008-12-31'])
print(aud_data.loc['2009-01-01':'2016-12-31'])
print(aud_data.loc['2017-01-01':'2020-12-31'])

# %%
# random walk mse(validation)
se = aud_data.loc['2009-01-01':'2016-12-31'] ** 2
mse = sum(se) / len(se)
mse

# %%
aud_min_valid_loss = 100000
aud_min_valid_length_input_sequence = 0
aud_min_valid_config_under_specific_input_length = ()
for length_input_sequence in tqdm(length_input_sequence_list):
    optimum_config_under_specific_input_length, valid_loss_under_specific_input_length = find_optimum_config_under_specific_input_length(aud_data, length_input_sequence, num_epochs_list, num_hidden_layers_list, num_hidden_sizes_list, batch_sizes_list)
    if valid_loss_under_specific_input_length < aud_min_valid_loss:
        aud_min_valid_loss = valid_loss_under_specific_input_length
        aud_min_valid_length_input_sequence = length_input_sequence
        aud_min_valid_config_under_specific_input_length = optimum_config_under_specific_input_length

# %% [markdown]
# # gbp

# %%
gbp_data = diff_log_data.iloc[:,2]
print(gbp_data.loc['1981-01-01':'2008-12-31'])
print(gbp_data.loc['2009-01-01':'2016-12-31'])
print(gbp_data.loc['2017-01-01':'2020-12-31'])

# %%
# random walk mse(validation)
se = gbp_data.loc['2009-01-01':'2016-12-31'] ** 2
mse = sum(se) / len(se)
mse

# %%
gbp_min_valid_loss = 100000
gbp_min_valid_length_input_sequence = 0
gbp_min_valid_config_under_specific_input_length = ()
for length_input_sequence in tqdm(length_input_sequence_list):
    optimum_config_under_specific_input_length, valid_loss_under_specific_input_length = find_optimum_config_under_specific_input_length(gbp_data, length_input_sequence, num_epochs_list, num_hidden_layers_list, num_hidden_sizes_list, batch_sizes_list)
    if valid_loss_under_specific_input_length < gbp_min_valid_loss:
        gbp_min_valid_loss = valid_loss_under_specific_input_length
        gbp_min_valid_length_input_sequence = length_input_sequence
        gbp_min_valid_config_under_specific_input_length = optimum_config_under_specific_input_length

# %%
print('cad optimum config:', cad_min_valid_length_input_sequence, ',', cad_min_valid_config_under_specific_input_length, 'length_input_sequence, (num_hidden_layers, num_hidden_sizes, batch_sizes, num_epochs).')
print('aud optimum config:', aud_min_valid_length_input_sequence, ',', aud_min_valid_config_under_specific_input_length, 'length_input_sequence, (num_hidden_layers, num_hidden_sizes, batch_sizes, num_epochs).')
print('gbp optimum config:', gbp_min_valid_length_input_sequence, ',', gbp_min_valid_config_under_specific_input_length, 'length_input_sequence, (num_hidden_layers, num_hidden_sizes, batch_sizes, num_epochs).')