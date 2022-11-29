import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, number_levels, number_of_stimuli, number_hidden, NUMBER_OF_CHANNELS, dropout, bidirectional):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=NUMBER_OF_CHANNELS,
                          hidden_size=number_hidden,
                          num_layers=number_levels,
                          dropout=dropout,
                          bidirectional=bidirectional)
        out_dim = 2 * number_hidden if bidirectional else number_hidden
        self.cl = nn.Linear(in_features=out_dim, out_features=number_of_stimuli, bias=True)

    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)
        gru_out, _ = self.gru(inputs)
        print(gru_out.shape, "gru_out")
        print(torch.mean(gru_out, dim=1).shape, "mean")
        output = self.cl(torch.mean(gru_out, dim=1))
        return output
