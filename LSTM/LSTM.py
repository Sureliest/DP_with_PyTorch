import torch
import torch.nn as nn



if __name__ == '__main__':
    lstm = nn.LSTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    output, hn = lstm(input, (h0, c0))
    print(output.size(), hn[0].size(), hn[1].size())


    embedding = nn.Embedding(10, 3)
    word_input = torch.LongTensor([[1, 2, 4, 5],[4, 3, 2, 9]])
    word_output = embedding(input)
    print(output.size())