import os
import copy
import math
import random
import numpy as np
import pandas as pd
# from scipy.stats import norm

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

#==============================================================================

# 创建一个 GRU 模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        nn.init.constant_(self.linear.weight, 0.0)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        output, hn = self.gru(x)
        out = self.linear(output[:, -1, :])
        return out

#==============================================================================
# 创建损失函数对象
loss_fn = nn.MSELoss()
# 创建数据加载器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)

#==============================================================================

def fit_GRU(data_x, data_y, hidden_size, output_size, num_layers, num_epochs, batch_size, lr, train_prop, set_seed):
    torch.manual_seed(set_seed)
    np.random.seed(set_seed)
    # 转换输入数据为 PyTorch 张量
    X = torch.tensor(data_x, dtype = torch.float32)
    Y = torch.tensor(data_y, dtype = torch.float32).reshape(-1, 1)
    # 划分训练集和验证集
    random.seed(set_seed)
    train_index = random.sample(range(data_y.size), math.floor(data_y.size * train_prop))
    valid_index = list(set(range(data_y.size)) - set(train_index))
    x_train, y_train, x_valid, y_valid = X[train_index], Y[train_index], X[valid_index], Y[valid_index]
    train_iter = load_array((x_train, y_train), batch_size = batch_size)
    # 创建 GRU 模型
    model = GRUModel(data_x.shape[-1], hidden_size, output_size, num_layers)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-2)
    # 早停策略，防止过拟合  
    best_valid = 1e8
    epoch_valids = []   
    for epoch in range(num_epochs):
        batch_valids = []
        for x_batch, y_batch in train_iter:
            model.train()
            loss = loss_fn(model(x_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                valid_loss = loss_fn(model(x_valid), y_valid).detach().numpy().item()
            batch_valids.append(valid_loss)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_dict = copy.deepcopy(model.state_dict())
        
        print("epoch {}:".format(epoch), np.mean(batch_valids))
        epoch_valids.append(np.mean(batch_valids))
        if len(epoch_valids) >= 15 and np.mean(epoch_valids[-5:]) >= np.mean(epoch_valids[-15:-5]):
            break
    
    model.load_state_dict(best_dict)
    model.eval()
    return model

#==============================================================================

# 训练与预测
def gruLearn(series, pastLen, split_n):
    scale_mean = series[:split_n].mean()
    scale_std = series[:split_n].std()
    series_scaled = (series - scale_mean) / scale_std
    
    data_x = pd.DataFrame()
    data_y = pd.DataFrame(series_scaled, copy=True)
    data_y.columns = ['y']
    for i in range(pastLen, 0, -1):
        data_x[i] = data_y['y'].shift(i)
    data_x = data_x.iloc[pastLen:]
    data_x2 = data_x.values - data_x.mean(axis=1).values.reshape(-1, 1)
    data_x = np.stack((data_x.values, data_x2 ** 2, data_x2 ** 3, data_x2 ** 4), axis=2)
    data_y = data_y.iloc[pastLen:].values.reshape(-1, 1)
    
    train_x = data_x[:split_n]
    train_y = data_y[:split_n]
    test_x = data_x[split_n:]
    test_y = data_y[split_n:]
    
    hidden_size = 8
    num_layers = 1
    output_size = 1
    
    num_epochs = 1000
    batch_size = 256
    lr = 0.001
    train_prop = 0.8

    # 调用fit_GRU函数
    gru = fit_GRU(train_x, train_y, hidden_size, output_size, num_layers, num_epochs, batch_size, lr, train_prop, set_seed=10)
    predict = gru(torch.tensor(test_x, dtype=torch.float32))
    predict = (predict * scale_std + scale_mean).detach().numpy()
    return predict

#==============================================================================

if __name__ == '__main__':
    if not os.path.isdir('gru_results/'):
        os.mkdir('gru_results/')
    
    sp500 = pd.read_csv('SP500_19900102_20221231.csv')[['caldt', 'vwretd']].set_index('caldt')
    
    allDays = sp500.shape[0]
    rollDays = 252
    predictions_list = []
    for i in range(0, allDays, rollDays):
        end_day = allDays - i
        test_day = end_day - rollDays
        # 这里rollDays * 10是训练数据长度，* num部分num 为 参数选择，{10、15、20、25}，对应文件路径{'gru_results/all_predictions1.csv', 'gru_results/all_predictions2.csv', 'gru_results/all_predictions3.csv', 'gru_results/all_predictions4.csv'}
        start_day = test_day - rollDays * 10 - 126
        if start_day < 0:
            continue
        
        series = pd.DataFrame(sp500.iloc[start_day:end_day], copy = True)
        series = series['vwretd'].reset_index(drop = True)

        predictions = gruLearn(series, 126, -rollDays)
        print('\n end day: {}, predictions\n'.format(end_day))
        predictions_list.append(predictions.flatten())
    # 保存所有预测值到一个文件中
    all_predictions_df = pd.DataFrame(np.concatenate(predictions_list, axis=0), columns=['predictions'])
    all_predictions_df['caldt'] = sp500.index[-len(all_predictions_df):]
    all_predictions_df['vwretd'] = sp500.iloc[-len(all_predictions_df):].values
    all_predictions_df = all_predictions_df[['caldt','vwretd','predictions']]
    all_predictions_df.to_csv('gru_results/all_predictions1.csv', index=False)
