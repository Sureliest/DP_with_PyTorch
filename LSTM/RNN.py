import torch
import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.w_xh = nn.Linear(input_size, hidden_size)
        self.w_hh = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x, hidden):
        return self.step(x, hidden)

    def step(self, x, hidden):
        h1 = self.w_hh(hidden)
        w1 = self.w_xh(x)
        out = torch.tanh(h1 + w1)
        hidden = self.w_hh.weight
        return out, hidden


rnn = RNN(20, 50)
input = torch.randn(32, 20)
h_0 = torch.randn(32, 50)
seq_len = input.shape[0]
for i in range(seq_len):
    output, hn = rnn(input[i, :], h_0)
    print(output.size(), h_0.size())
