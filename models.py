import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class ParityRNNModel(nn.Module):
    IS_ACT = False

    def __init__(self, config):
        super(ParityRNNModel, self).__init__()
        self.rnn_size = config.parity_rnn_size
        if config.parity_rnn_type == "RNN":
            rnn_class = nn.RNN
        elif config.parity_rnn_type == "LSTM":
            rnn_class = nn.LSTM
        else:
            raise KeyError("rnn_class")

        self.rnn = rnn_class(
            input_size=config.parity_input_size,
            hidden_size=self.rnn_size,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.rnn_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, x, compute_ponder_cost=True):
        rnn_out, hidden = self.rnn(x.unsqueeze(1))
        return self.fc1(rnn_out).squeeze(1).squeeze(1), None


class ParityACTModel(nn.Module):
    IS_ACT = True

    def __init__(self, config):
        super(ParityACTModel, self).__init__()
        self.rnn_size = config.parity_rnn_size
        if config.parity_rnn_type == "RNN":
            rnn_cell_class = nn.RNNCell
        elif config.parity_rnn_type == "LSTM":
            rnn_cell_class = nn.LSTMCell
        else:
            raise KeyError("rnn_class")
        rnn_cell = rnn_cell_class(
            input_size=config.parity_input_size + 1,  # +1 for flag
            hidden_size=self.rnn_size,
        )
        self.rnn = ACTFromCell(
            rnn_cell=rnn_cell,
            max_ponder=config.act_max_ponder,
            epsilon=config.act_epsilon,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.rnn_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, x, compute_ponder_cost=True):
        rnn_out, hidden, ponder_cost = self.rnn(
            input_=x.unsqueeze(1),
            compute_ponder_cost=compute_ponder_cost,
        )
        return self.fc1(rnn_out).squeeze(1).squeeze(1), ponder_cost


class LogicRNNModel(nn.Module):
    IS_ACT = False

    def __init__(self, config):
        super(LogicRNNModel, self).__init__()
        self.rnn_size = config.logic_rnn_size
        if config.logic_rnn_type == "RNN":
            rnn_class = nn.RNN
        elif config.logic_rnn_type == "LSTM":
            rnn_class = nn.LSTM
        else:
            raise KeyError("rnn_class")

        self.rnn = rnn_class(
            input_size=config.logic_input_size,
            hidden_size=self.rnn_size,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.rnn_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, x, compute_ponder_cost=True):
        rnn_out, hidden = self.rnn(x.unsqueeze(1))
        return self.fc1(rnn_out).squeeze(1).squeeze(1), None


class LogicACTModel(nn.Module):
    IS_ACT = True

    def __init__(self, config):
        super(LogicACTModel, self).__init__()
        self.rnn_size = config.logic_rnn_size
        if config.logic_rnn_type == "RNN":
            rnn_cell_class = nn.RNNCell
        elif config.logic_rnn_type == "LSTM":
            rnn_cell_class = nn.LSTMCell
        else:
            raise KeyError("rnn_class")
        rnn_cell = rnn_cell_class(
            input_size=config.logic_input_size + 1,  # +1 for flag
            hidden_size=self.rnn_size,
        )
        self.rnn = ACTFromCell(
            rnn_cell=rnn_cell,
            max_ponder=config.act_max_ponder,
            epsilon=config.act_epsilon,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.rnn_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, x, compute_ponder_cost=True):
        rnn_out, hidden, ponder_cost = self.rnn(
            input_=x.unsqueeze(1),
            compute_ponder_cost=compute_ponder_cost,
        )
        return self.fc1(rnn_out).squeeze(1).squeeze(1), ponder_cost


class RNNFromCell(nn.Module):
    def __init__(self, rnn_cell, batch_first=False):
        super(RNNFromCell, self).__init__()
        self.rnn_cell = rnn_cell
        self.batch_first = batch_first
        self._is_lstm = isinstance(self.rnn_cell, nn.LSTMCell)

    def forward(self, input_, hx=None):
        if self.batch_first:
            # Move batch to second
            input_ = input_.transpose(0, 1)
        if hx is None:
            hx = Variable(input_.data.new(
                input_.size(1), self.rnn_cell.hidden_size
            ).zero_())
            if self._is_lstm:
                hx = [hx, hx]

        hx_list = []
        for input_row in input_:
            hx = self.rnn_cell(input_row, hx)
            hx_list.append(hx)

        if self._is_lstm:
            all_hx = [
                torch.stack([_[0] for _ in hx_list]),
                torch.stack([_[1] for _ in hx_list]),
            ]
            hx = hx[0].unsqueeze(0), hx[1].unsqueeze(1)
        else:
            all_hx = torch.stack(hx_list)
            hx = hx.unsqueeze(0)

        if self.batch_first:
            # Move batch to first
            if self._is_lstm:
                all_hx = all_hx[0].transpose(0, 1)
            else:
                all_hx = all_hx.transpose(0, 1)

        return all_hx, hx


class ACTFromCell(nn.Module):
    def __init__(self, rnn_cell, max_ponder=100, epsilon=0.01,
                 batch_first=False):
        super(ACTFromCell, self).__init__()
        self.rnn_cell = rnn_cell
        self.batch_first = batch_first
        self.max_ponder = max_ponder
        self.epsilon = epsilon

        self._is_lstm = isinstance(self.rnn_cell, nn.LSTMCell)
        self.ponder_linear = nn.Linear(self.rnn_cell.hidden_size, 1)

    def forward(self, input_, hx=None, compute_ponder_cost=True):
        if self.batch_first:
            # Move batch to second
            input_ = input_.transpose(0, 1)
        if hx is None:
            hx = Variable(input_.data.new(
                input_.size(1), self.rnn_cell.hidden_size
            ).zero_())
            if self._is_lstm:
                cx = hx

        # Pre-allocate variables
        time_size, batch_size, input_dim_size = input_.size()
        accum_h = Variable(input_.data.new(batch_size).zero_())
        accum_hx = Variable(input_.data.new(
            input_.size(1), self.rnn_cell.hidden_size
        ).zero_())
        if self._is_lstm:
            accum_cx = Variable(input_.data.new(
                input_.size(1), self.rnn_cell.hidden_size
            ).zero_())
        selector = input_.data.new(batch_size).byte()
        hx_list, cx_list = [], []
        step_count = Variable(input_.data.new(batch_size).zero_())
        ponder_cost = 0
        ponder_times = []

        # For each t
        for input_row in input_:

            accum_h = accum_h.zero_()
            accum_hx = accum_hx.zero_()
            selector = selector.fill_(1)

            if self._is_lstm:
                accum_cx = accum_cx.zero_()
            step_count = step_count.zero_()
            input_row_with_flag = torch.cat([
                input_row,
                Variable(input_row.data.new(batch_size, 1).zero_())
            ], dim=1)
            if compute_ponder_cost:
                step_ponder_cost = Variable(input_.data.new(batch_size).zero_())

            for act_step in range(self.max_ponder):
                input_row_with_flag[:, input_dim_size] = act_step
                idx = bool_to_idx(selector)
                if compute_ponder_cost:
                    step_ponder_cost[idx] = accum_h[idx]

                if self._is_lstm:
                    hx[idx], cx[idx] = self.rnn_cell(
                        input_row_with_flag[idx], (hx[idx], cx[idx]))
                else:
                    hx[idx] = self.rnn_cell(input_row_with_flag[idx], hx[idx])
                accum_hx[idx] += hx[idx]
                h = F.sigmoid(self.ponder_linear(hx[idx]).squeeze(1))
                accum_h[idx] += h
                p = h - (accum_h[idx] - 1).clamp(min=0)
                accum_hx[idx] += p.unsqueeze(1) * hx[idx]
                if self._is_lstm:
                    accum_cx[idx] += p.unsqueeze(1) * cx[idx]
                step_count[idx] += 1
                selector = (accum_h < 1 - self.epsilon).data
                if not selector.any():
                    break

            ponder_times.append(step_count.data.cpu().numpy())
            if compute_ponder_cost:
                ponder_cost -= step_ponder_cost

            hx = accum_hx / step_count.clone().unsqueeze(1)
            hx_list.append(hx)

            if self._is_lstm:
                cx = accum_cx / step_count.clone().unsqueeze(1)
                cx_list.append(cx)

        if self._is_lstm:
            all_hx = [
                torch.stack(hx_list),
                torch.stack(cx_list),
            ]
            hx, cx = hx.unsqueeze(0), cx.unsqueeze(1)
        else:
            all_hx = torch.stack(hx_list)
            hx = hx.unsqueeze(0)

        if self.batch_first:
            # Move batch to first
            if self._is_lstm:
                all_hx = all_hx[0].transpose(0, 1)
            else:
                all_hx = all_hx.transpose(0, 1)

        return all_hx, hx, {
            "ponder_cost": ponder_cost,
            "ponder_times": ponder_times,
        }

    def reset_parameters(self):
        self.rnn_cell.reset_parameters()
        self.ponder_linear.reset_parameters()
        self.ponder_linear.bias.data.fill_(1)


def bool_to_idx(idx):
    return idx.nonzero().squeeze(1)
