import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series)):
        end_ix = i + n_steps
        if end_ix > len(series)-1:
            break
        seq_x, seq_y = series[i:end_ix], series[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def train_model(model, train_loader, num_epochs, criterion, optimizer):
    for epoch in range(num_epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} Loss: {loss.item():.4f}')

def predict(model, test_inputs, n_steps):
    model.eval()
    for _ in range(n_steps):
        x = torch.tensor(test_inputs[-1]).unsqueeze(0)
        y_test_pred = model(x)
        test_inputs.append(y_test_pred.detach().numpy()[0])
    return test_inputs[n_steps:]

def create_datasets(data, sequence_length=100, batch_size=1, split_frac=0.8):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))
    X, y = prepare_data(data_normalized, sequence_length)
    split_pt = int(split_frac * len(X))
    train_x = torch.tensor(X[:split_pt], dtype=torch.float32)
    train_y = torch.tensor(y[:split_pt], dtype=torch.float32)
    test_x = torch.tensor(X[split_pt:], dtype=torch.float32)
    test_y = torch.tensor(y[split_pt:], dtype=torch.float32)

    train_data = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    return train_loader, test_x, test_y, scaler
