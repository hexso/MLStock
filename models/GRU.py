import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch
import pandas as pd
import time
'''
GRU Model을 이용
다음날 고가가 특정 퍼센트 이상을 찾는다.
'''

HIGH_RATE = 15
MODEL_FILE_NAME = 'GRURapid.pt'
COL_DATA = ['open', 'close', 'high']

class GRURapid(nn.Module):

    def __init__(self, device='cpu'):
        super(GRURapid, self).__init__()
        self.fileName = MODEL_FILE_NAME
        self.input_size = 7 #open, high, low, close, volume, change, obv
        self.input_list = ['open', 'high', 'low', 'close', 'volume', 'obv', 'change']
        self.output_size = 1 #indicator
        self.output = 'indicator'
        self.hidden_dim = 128
        self.layer_cnt = 5
        self.window_size = 20
        self.train_rate = 0.9
        self.device = device
        self.epoch_cnt = 100

        self.std_scal = StandardScaler()
        self.gru = nn.GRU(self.input_size, self.hidden_dim, self.layer_cnt).to(device)
        self.hidden_layer = nn.Linear(self.hidden_dim, self.output_size).to(device)
        self.output_layer = self._step_function

        self.origin_col = ['change']

    def forward(self, x):
        h0 = torch.zeros(self.layer_cnt, self.window_size-1, self.hidden_dim).to(self.device).requires_grad_()
        out, hn = self.gru(x, h0.detach())

        out = self.hidden_layer(out[:,-1])
        out = self.output_layer(out)
        return out

    def _step_function(self, x):
        if x < 0:
            return 0
        else:
            return 1

    def _slice_window(self, stock_data):
        data_raw = stock_data
        data = []

        for index in range(len(data_raw) - self.window_size):
            data.append(data_raw[index: index + self.window_size])

        return data

    def _pre_data_process(self, data):
        new_data = data

        for col in data.columns:
            if col not in self.origin_col:
                reshape_data = data[col].values.reshape(-1,1)
                new_data[col] = self.std_scal.fit_transform(reshape_data)

        for col in self.origin_col:
            new_data[col] = data[col]

        #내일 고점 계산
        new_data['change_over_tmw'] = 0
        for idx, val in enumerate(new_data['close']):
            if idx + 1 >= len(new_data):
                break
            close = data['close'][idx]
            high_tmw = data['high'][idx + 1]
            if (high_tmw - close) > (HIGH_RATE/100):
                data['change_over_tmw'][idx] = 1

        #고점의 정확도 계산
        new_data['indicator'] = 0
        for idx, val in enumerate(new_data['change_over_tmw']):
            for i in range(10):
                if idx + i >= len(data):
                    break
                if data['change_over_tmw'][idx + i] == 1:
                    data['indicator'][idx] = 1 - (0.1 * idx)
                    break
        new_data = new_data.drop('change_over_tmw',axis=1)
        new_data = new_data.dropna()

        slice_data_x = self._slice_window(new_data[self.input_list])
        slice_data_y = self._slice_window(new_data[self.output])
        x_slice_data = np.array(slice_data_x)
        y_slice_data = np.array(slice_data_y)

        return x_slice_data, y_slice_data

    def learn(self, model, data):
        x_slice, y_slice = self._pre_data_process(data)

        total_size = len(x_slice)
        train_size = int(total_size * self.train_rate)

        x_train = x_slice[:train_size, :-1]
        x_test = x_slice[train_size:, :-1]

        y_train = y_slice[:train_size, -1]
        y_test = y_slice[train_size:, -1]

        x_train_torch = torch.from_numpy(x_train).type(torch.Tensor).to(self.device)
        x_test_torch = torch.from_numpy(x_test).type(torch.Tensor).to(self.device)
        y_train_torch = torch.from_numpy(y_train).type(torch.Tensor).to(self.device)
        y_test_torch = torch.from_numpy(y_test).type(torch.Tensor).to(self.device)

        loss_function = nn.MSELoss(reduction='mean')
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        hist = np.zeros(self.epoch_cnt)

        start_time = time.time()
        for t in range(self.epoch_cnt):
            y_train_pred = model(x_train_torch)
            loss = loss_function(y_train_pred, y_train_torch.reshape(-1, 1))
            if t % 10 == 0:
                print('Epoch ', t, 'MSE: ', loss.item())
            hist[t] = loss.item()
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        train_time = time.time() - start_time
        print('Training Time : {}'.format(train_time))
