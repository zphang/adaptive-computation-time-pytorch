import numpy as np
import torch
import torch.utils.data


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, *data_list):
        assert len(data_list) > 0
        self.data_length = len(data_list[0])
        for data in data_list[1:]:
            assert len(data) == self.data_length
        self.data_list = data_list

    def __getitem__(self, index):
        return [
            data[index]
            for data in self.data_list
        ]

    def __len__(self):
        return self.data_length


class DataManager:
    @classmethod
    def create_data(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _get_length(cls, config):
        raise NotImplementedError

    @classmethod
    def _get_dataloader(cls, data, batch_size):
        data_x, data_y = data
        return torch.utils.data.DataLoader(
            MultiDataset(data_x, data_y),
            batch_size=batch_size,
            shuffle=True,
        )

    @classmethod
    def create_dataloader(cls, config, mode="train"):
        length = cls._get_length(config)
        if mode == "train":
            pass
        elif mode == "test":
            length = int(config.test_percentage * length)
        else:
            raise KeyError(mode)
        data = cls.create_data(length=length)
        return cls._get_dataloader(data=data, batch_size=config.batch_size)


class ParityDataManager(DataManager):
    @classmethod
    def create_data(cls, length):
        parity_x = np.random.randint(2, size=(length, 64)).astype(
            np.float32) * 2 - 1
        zero_out = np.random.randint(1, 64, size=length)
        for i in range(length):
            parity_x[i, zero_out[i]:] = 0.
        parity_y = (np.sum(parity_x == 1, axis=1) % 2).astype(np.float32)
        return np.expand_dims(parity_x, 1), parity_y

    @classmethod
    def _get_length(cls, config):
        return config.parity_data_len


class LogicDataManager(DataManager):
    LOGIC_TABLE = np.array([
        [[1, 0], [0, 0]],  # NOR
        [[0, 1], [0, 0]],  # Xq
        [[0, 0], [1, 0]],  # ABJ
        [[0, 1], [1, 0]],  # XOR
        [[1, 1], [1, 0]],  # NAND
        [[0, 0], [0, 1]],  # AND
        [[1, 0], [0, 1]],  # XNOR
        [[1, 1], [0, 1]],  # if/then
        [[1, 0], [1, 1]],  # then/if
        [[0, 1], [1, 1]],  # OR
    ])

    @classmethod
    def create_data(cls, length):
        p_and_q = np.random.randint(2, size=(length, 10, 2))
        p_and_q[:, 1:, 1] = 0

        operations = np.random.randint(0, 10, size=(length, 10, 10))
        num_operations = np.random.randint(1, 11, size=(length, 10))
        for i in range(length):
            for t in range(10):
                operations[i, t, num_operations[i, t]:] = -1
        one_hot_operations = np.zeros((length, 10, 100))

        logic_y = np.empty(shape=(length, 10))
        for row_index, (row_p_and_q, row_operations) in enumerate(
                zip(p_and_q, operations)):
            b_0 = row_p_and_q[0, 0]
            for t in range(10):
                for op_index, operation in enumerate(row_operations[t]):
                    if operation == -1:
                        break
                    one_hot_operations[
                        row_index, t, op_index * 10 + operation] = 1

                result = cls._resolve_logic(b_0, row_p_and_q[t, 0],
                                            row_operations[t])
                logic_y[row_index, t] = result
                b_0 = result

        logic_x = np.concatenate([p_and_q, one_hot_operations], axis=2)
        return (
            logic_x.astype(np.float32),
            np.expand_dims(logic_y.astype(np.float32), 2),
        )

    @classmethod
    def _get_length(cls, config):
        return config.logic_data_len

    @classmethod
    def _resolve_logic(cls, p, q, op_list):
        for op in op_list:
            if op == -1:
                break
            p, q = q, cls.LOGIC_TABLE[op][p][q]
        return q


def resolve_data_manager(config):
    if config.task == "parity":
        return ParityDataManager
    elif config.task == "logic":
        return LogicDataManager
    else:
        raise KeyError(config.task)
