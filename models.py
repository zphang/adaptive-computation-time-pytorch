from torch import nn
import torch.nn.functional as F


class ParityModel(nn.Module):
    def __init__(self, config):
        super(ParityModel, self).__init__()
        self.rnn_size = config.parity_rnn_size
        self.rnn = nn.RNN(
            input_size=config.parity_input_size,
            hidden_size=self.rnn_size,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.rnn_size, 1)

    def forward(self, x):
        self.rnn.flatten_parameters()
        rnn_out, hidden = self.rnn(x.unsqueeze(1))
        return F.sigmoid(self.fc1(rnn_out).squeeze(1).squeeze(1))
