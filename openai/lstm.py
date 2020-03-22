import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class XORModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(XORModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out    = self.run_through_lstm(x)
        out         = self.linear_out(lstm_out)
        out         = F.log_softmax(out, dim=0)
        return out

    def run_through_lstm(self, x):
        # hidden = (torch.randn(1, 1, 2), torch.randn(1, 1, 2))
        # for i in x:
        #     out, hidden = self.lstm(i.view(-1, 1, 1).long(), hidden)
        out = self.lstm(x.view(-1, 1, 1))
        return out



# inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
def get_input(max_bits):
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

model = XORModel(input_dim=1, hidden_dim=2, output_dim=1)

optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0)
# loss_function = torch.nn.MSELoss()
# loss_function = torch.nn.BCELoss()
loss_function = torch.nn.NLLLoss()

inputs, labels = gen_sequences(100)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
for inx, elt in enumerate(zip(inputs, labels)):
    inps, lbls = torch.tensor(elt[0]).view(1, 1, -1), torch.tensor(elt[1]).view(-1, 1, 1)
    print(lbls, lbls.squeeze(1))
    lbls = lbls.to(device=device, dtype=torch.int64)

    print(inps.size(), lbls.size())
    optimizer.zero_grad()

    out = model(inps)
    loss = loss_function(out, lbls)
    loss.backward()
    optimizer.step()
    step += 1

    if (step > 100):
        break;
    