import attr


@attr.s
class Config(object):
    seed = attr.ib(1234)
    batch_size = attr.ib(128 * 16)
    cuda = attr.ib(True)
    learning_rate = attr.ib(0.1 ** 4 * 16)
    test_percentage = attr.ib(0.1 / 0.16)
    train_log = attr.ib(True)
    train_log_interval = attr.ib(10)
    num_epochs = attr.ib(32)

    parity_data_len = attr.ib(100000 * 16)
    parity_input_size = attr.ib(64)
    parity_rnn_size = attr.ib(128)
    parity_rnn_type = attr.ib("RNN")

    act_max_ponder = attr.ib(100)
    act_epsilon = attr.ib(0.01)
    act_ponder_penalty = attr.ib(0.0001)

    logic_data_len = attr.ib(100000 * 16)
    logic_input_size = attr.ib(102)
    logic_rnn_size = attr.ib(128)
    logic_rnn_type = attr.ib("LSTM")
