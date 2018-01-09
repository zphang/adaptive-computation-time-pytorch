import attr
import torch
import argparse


def argparse_attr(default=attr.NOTHING, validator=None,
                  repr=True, cmp=True, hash=True, init=True,
                  convert=None, opt_string=None,
                  **argparse_kwargs):
    if opt_string is None:
        opt_string_ls = []
    elif isinstance(opt_string, str):
        opt_string_ls = [opt_string]
    else:
        opt_string_ls = opt_string

    if argparse_kwargs.get("type", None) is bool:
        argparse_kwargs["choices"] = {True, False}
        argparse_kwargs["type"] = _is_true

    return attr.attr(
        default=default,
        validator=validator,
        repr=repr,
        cmp=cmp,
        hash=hash,
        init=init,
        convert=convert,
        metadata={
            "opt_string_ls": opt_string_ls,
            "argparse_kwargs": argparse_kwargs,
        }
    )


def update_parser(parser, class_with_attributes):
    for attribute in class_with_attributes.__attrs_attrs__:
        if "argparse_kwargs" in attribute.metadata:
            argparse_kwargs = attribute.metadata["argparse_kwargs"]
            opt_string_ls = attribute.metadata["opt_string_ls"]
            if attribute.default is attr.NOTHING:
                argparse_kwargs = argparse_kwargs.copy()
                argparse_kwargs["required"] = True
            else:
                argparse_kwargs["default"] = attribute.default
            parser.add_argument(
                f"--{attribute.name}", *opt_string_ls,
                **argparse_kwargs
            )


def read_parser(parser, class_with_attributes, skip_non_class_attributes=False):
    attribute_name_set = {
        attribute.name
        for attribute in class_with_attributes.__attrs_attrs__
    }

    kwargs = dict()
    leftover_kwargs = dict()

    for k, v in vars(parser.parse_args()).items():
        if k in attribute_name_set:
            kwargs[k] = v
        else:
            if not skip_non_class_attributes:
                raise RuntimeError(f"Unknown attribute {k}")
            leftover_kwargs[k] = v

    instance = class_with_attributes(**kwargs)
    if skip_non_class_attributes:
        return instance, leftover_kwargs
    else:
        return instance


def _is_true(x):
    return x == "True"


@attr.s
class Config:

    # Global configuration
    cuda = argparse_attr(
        default=torch.has_cudnn, type=bool,
        help="Whether to use cuda",
    )
    seed = argparse_attr(
        default=1234, type=int,
        help="Seed",
    )
    batch_size = argparse_attr(
        default=128 * 16, type=int,
        help="Batch size for model",
    )

    # ACT configuration
    use_act = argparse_attr(
        default=True, type=bool,
        help="Whether to use ACT",
    )
    act_max_ponder = argparse_attr(
        default=100, type=int,
        help="Maximum number of ponder steps",
    )
    act_epsilon = argparse_attr(
        default=0.01, type=int,
        help="Epsilon margin for halting",
    )
    act_ponder_penalty = argparse_attr(
        default=0.0001, type=float,
        help="Weight for ponder cost",
    )

    # Task configuration
    task = argparse_attr(
        default=None, type=str,
        help="Experiment Task (parity|logic)",
    )

    # Train configuration
    learning_rate = argparse_attr(
        default=0.1 ** 4 * 16, type=float,
        help="Learning rate",
    )
    train_log = argparse_attr(
        default=True, type=bool,
        help="Whether to have verbose training logs",
    )
    train_log_interval = argparse_attr(
        default=10, type=int,
        help="How often to output training log messages",
    )
    num_epochs = argparse_attr(
        default=64, type=int,
        help="Number of training epochs",
    )

    # Test configuration
    test_percentage = argparse_attr(
        default=0.1 / 16, type=float,
        help="Size of test set, as percentage of training set. "
             "For synthetic tasks",
    )

    # Task: Parity configuration
    parity_data_len = argparse_attr(
        default=100000 * 16, type=int,
        help="Samples in training epoch",
    )
    parity_input_size = argparse_attr(
        default=64, type=int,
        help="Size of parity input",
    )
    parity_rnn_size = argparse_attr(
        default=128, type=int,
        help="Hidden size of RNN",
    )
    parity_rnn_type = argparse_attr(
        default="RNN", type=str,
        help="RNN type (RNN/LSTM)",
    )

    # Task: Logic configuration
    logic_data_len = argparse_attr(
        default=10000 * 16, type=int,
        help="Samples in training epoch",
    )
    logic_input_size = argparse_attr(
        default=102, type=int,
        help="Size of logic input",
    )
    logic_rnn_size = argparse_attr(
        default=128, type=int,
        help="Hidden size of RNN",
    )
    logic_rnn_type = argparse_attr(
        default="LSTM", type=str,
        help="RNN type (RNN/LSTM)",
    )

    # Saving
    model_save_path = argparse_attr(
        default=None, type=str,
        help="Folder to save models",
        required=True,
    )
    model_save_interval = argparse_attr(
        default=1, type=int,
        help="Epoch intervals to save model"
    )

    @classmethod
    def parse_configuration(cls, prog=None, description=None):
        parser = argparse.ArgumentParser(
            prog=prog,
            description=description,
        )
        update_parser(
            parser=parser,
            class_with_attributes=cls,
        )
        return read_parser(
            parser=parser,
            class_with_attributes=cls,
        )
