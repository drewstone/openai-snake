import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
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
    def __init__(self, input_dim=1, hidden_dim=2, output_dim=1, batch_size=1):
        super(ParityModel, self).__init__()
        print('New LSTM: (input: {}, hidden: {}, output: {})\n'
            .format(input_dim, hidden_dim, output_dim))
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear_in = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 1, 1)
        out    = self.get_lstm_last_layer(x)
        # out         = F.tanh(out)
        # print('LSTM layer output size: {}'.format(out.size()))
        # out         = self.linear_in(out)
        # print('Linear label output size: {}'.format(out.size()))
        # out         = torch.tanh(out)
        out         = self.linear_out(out)
        # print('Linear label output size: {}'.format(out.size()))
        predictions         = self.sigmoid(out)
        return out, predictions

    def get_lstm_last_layer(self, x):
        hidden = (
            torch.randn(self.batch_size, self.input_dim, self.hidden_dim),
            torch.randn(self.batch_size, self.input_dim, self.hidden_dim))
        out, hidden = self.lstm(x, hidden)
        return out[-1]

def evaluate(model, seq_len, device):
  # evaluate on more bits than training to ensure generalization
  test_loader = DataLoader(
      ParityDataset(num_sequences=5000, seq_len=int(seq_len * 1.5)), batch_size=1)

  is_correct = np.array([])

  for inputs, targets in test_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
      out, predictions = model(inputs)
      is_correct = np.append(is_correct, ((predictions > 0.5) == (targets > 0.5)))

  accuracy = is_correct.mean()
  return accuracy

batch_size = 1
num_sequences = 1000
seq_len = 50
train_dataloader = DataLoader(ParityDataset(num_sequences, seq_len), batch_size)

model = ParityModel(input_dim=1, hidden_dim=2, output_dim=1, batch_size=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = torch.nn.MSELoss()
# loss_function = torch.nn.BCEWithLogitsLoss()
# loss_function = torch.nn.NLLLoss()
# loss_function = torch.nn.CrossEntropyLoss()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

running_loss = 0.0
for epoch in range(1, 5):
    y_true = list()
    y_pred = list()
    for inx, data in enumerate(train_dataloader):
        inputs, labels = data
        labels = labels.float().to(device=device)

        optimizer.zero_grad()

        out, predictions = model(inputs)
        loss = loss_function(out, labels.view(1, -1))
        loss.backward()
        optimizer.step()
        accuracy = ((predictions > 0.5) == (labels > 0.5)).type(torch.FloatTensor).mean()

        if inx % 250 == 249:
            print('step {}, loss {}, accuracy {}'.format(inx, loss.item(), accuracy))

        if inx % 1000 == 999:
            test_accuracy = evaluate(model, seq_len, device)
            print('test accuracy {}'.format(test_accuracy))
            if test_accuracy == 1.0:
                # stop early
                break
    y_true = list()
    y_pred = list()