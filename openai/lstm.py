import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(1)

def get_input(max_bits=50):
    return [ np.random.randint(2) for _ in range(max_bits) ]

def gen_sequences(num_sequences, max_bits=50):
    inputs = []
    labels = []
    for i in range(num_sequences):
        inp = get_input(max_bits)
        parity = 1 if np.sum(inp) % 2 != 0 else 0
        inputs.append(inp)
        labels.append(parity)
    return inputs, labels


class ParityDataset(Dataset):
    def __init__(self, num_sequences, seq_len):
        self.data, self.labels = gen_sequences(num_sequences, seq_len)

    def __getitem__(self, index):
        datum, label = self.data[index], self.labels[index]
        return torch.tensor(datum).float(), torch.tensor(label).float()

    def __len__(self):
        return len(self.data)


class ParityModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=2, output_dim=1):
        super(ParityModel, self).__init__()
        print('New LSTM: (input: {}, hidden: {}, output: {}\n'
            .format(input_dim, output_dim, hidden_dim))
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 1, 1)
        lstm_out    = self.run_through_lstm(x)
        # print('LSTM layer output size: {}'.format(lstm_out.size()))
        out         = self.linear_out(lstm_out)
        # print('Linear label output size: {}'.format(out.size()))
        out         = F.log_softmax(out.view(-1, 1), dim=1)
        # print(out.size())
        return out

    def run_through_lstm(self, x):
        # initialize the hidden state.
        # hidden = (torch.randn(1, 1, 2),
        #           torch.randn(1, 1, 2))
        # for i in x:
        #     out, hidden = self.lstm(i.view(-1, 1, 1), hidden)
        print(x.size())
        out, h = self.lstm(x)
        print(out.size())
        # print(out, hidden)
        return out

batch_size = 1
num_sequences = 10000
seq_len = 50
train_dataloader = DataLoader(ParityDataset(num_sequences, seq_len), batch_size)

model = ParityModel(input_dim=1, hidden_dim=2, output_dim=1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)
# loss_function = torch.nn.MSELoss()
loss_function = torch.nn.BCELoss()
# loss_function = torch.nn.NLLLoss()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

running_loss = 0.0
for inx, data in enumerate(train_dataloader):
    inputs, labels = data
    labels = labels.float().to(device=device)

    optimizer.zero_grad()

    out = model(inputs)
    # print(out.size())
    loss = loss_function(out, labels.float())
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if inx % 1000 == 999:    # every 1000 mini-batches...
        # ...log the running loss
        print('training loss', running_loss / 1000)
        running_loss = 0.0
    