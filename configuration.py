import attr


@attr.s
class Config(object):
    seed = attr.ib(1234)
    batch_size = attr.ib(512)
    cuda = attr.ib(True)
    learning_rate = attr.ib(0.1 ** 4)
    train_percentage = attr.ib(0.1)
    train_log = attr.ib(True)
    train_log_interval = attr.ib(10)
    num_epochs = 100

    parity_data_len = attr.ib(10000)
    parity_input_size = attr.ib(64)
    parity_difficulty = attr.ib(64)
    parity_rnn_size = attr.ib(128)

    logic_data_len = attr.ib(10000)
