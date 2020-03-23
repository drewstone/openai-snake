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
        # parity = torch.tensor([1, 0]) if np.sum(inp) % 2 != 0 else torch.tensor([0, 1])
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
        print('New LSTM: (input: {}, hidden: {}, output: {})\n'
            .format(input_dim, hidden_dim, output_dim))
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 1, 1)
        lstm_out    = self.run_through_lstm(x)
        out         = F.relu(lstm_out)
        # print('LSTM layer output size: {}'.format(lstm_out.size()))
        out         = self.linear_out(out)
        # print('Linear label output size: {}'.format(out.size()))
        return out

    def run_through_lstm(self, x):
        out, h = self.lstm(x)
        return out[-1]

batch_size = 1
num_sequences = 10000
seq_len = 10
train_dataloader = DataLoader(ParityDataset(num_sequences, seq_len), batch_size)

model = ParityModel(input_dim=1, hidden_dim=10, output_dim=1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# loss_function = torch.nn.MSELoss()
loss_function = torch.nn.BCEWithLogitsLoss()
# loss_function = torch.nn.NLLLoss()
# loss_function = torch.nn.CrossEntropyLoss()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

running_loss = 0.0
for inx, data in enumerate(train_dataloader):
    inputs, labels = data
    labels = labels.float().to(device=device)

    optimizer.zero_grad()

    out = model(inputs)
    # print(out, labels)
    loss = loss_function(out, labels.view(1, -1))
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if inx % 1000 == 999:    # every 1000 mini-batches...
        # ...log the running loss
        print('training loss', running_loss / 1000)
        running_loss = 0.0


test_num_seq = 100
test_dataloader = DataLoader(ParityDataset(test_num_seq, seq_len), batch_size)
with torch.no_grad():
    model.eval()
    for inx, data in enumerate(train_dataloader):
        inputs, labels = data
        labels = labels.float().to(device=device)
        out = model(inputs)
        print(out, labels)

