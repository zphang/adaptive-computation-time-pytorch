from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np

import models


def test_rnn_and_rnn_from_cell_match():
    i_size = 5
    h_size = 6

    x = np.random.normal(size=(1, 10, i_size))
    x_var = Variable(torch.Tensor(x))

    rnn1 = nn.RNN(
        input_size=i_size,
        hidden_size=h_size,
        batch_first=True,
    )
    rnn2_cell = nn.RNNCell(
        input_size=i_size,
        hidden_size=h_size,
    )
    _copy_rnn_params(rnn1, rnn2_cell)
    rnn2 = models.RNNFromCell(rnn2_cell, batch_first=True)

    outputs1, hiddens1 = rnn1(x_var)
    outputs2, hiddens2 = rnn2(x_var)

    assert outputs1.equal(outputs2)
    assert hiddens1.equal(hiddens2)


def test_lstm_and_lstm_from_cell_match():
    i_size = 5
    h_size = 6

    x = np.random.normal(size=(1, 10, i_size))
    x_var = Variable(torch.Tensor(x))

    lstm1 = nn.LSTM(
        input_size=i_size,
        hidden_size=h_size,
        batch_first=True,
    )
    lstm2_cell = nn.LSTMCell(
        input_size=i_size,
        hidden_size=h_size,
    )
    _copy_rnn_params(lstm1, lstm2_cell)
    lstm2 = models.RNNFromCell(lstm2_cell, batch_first=True)

    outputs1, (h1, c1) = lstm1(x_var)
    outputs2, (h2, c2) = lstm2(x_var)

    assert outputs1.equal(outputs2)
    assert h1.equal(h2)
    assert c1.equal(c2)


def _copy_rnn_params(rnn_from, rnn_to):
    rnn_to.weight_ih.data = rnn_from.weight_ih_l0.data
    rnn_to.weight_hh.data = rnn_from.weight_hh_l0.data
    rnn_to.bias_ih.data = rnn_from.bias_ih_l0.data
    rnn_to.bias_hh.data = rnn_from.bias_hh_l0.data
