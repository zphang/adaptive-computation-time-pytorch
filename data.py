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
        )

    @classmethod
    def create_dataloader(cls, config, mode="train"):
        length = cls._get_length(config)
        if mode == "train":
            pass
        elif mode == "Test":
            length *= config.train_percentage
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
        return parity_x, parity_y

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
        p_and_q = np.random.randint(2, size=(length, 2))
        num_operations = np.random.randint(1, 11, size=length)
        operations = [
            np.random.randint(0, 10, size=b)
            for b in num_operations
        ]
        one_hot_operations = np.zeros((length, 100))

        for row_index, row_operations in enumerate(operations):
            for op_index, row_operation in enumerate(row_operations):
                one_hot_operations[row_index, op_index * 10 + row_operation] = 1

        logic_x = np.hstack([p_and_q, one_hot_operations])
        logic_y = np.array([
            cls._resolve_logic(row_p_and_q[0], row_p_and_q[1], row_operations)
            for row_p_and_q, row_operations in zip(p_and_q, operations)
        ])
        return logic_x, logic_y

    @classmethod
    def _get_length(cls, config):
        return config.logic_data_len

    @classmethod
    def _resolve_logic(cls, p, q, op_list):
        for op in op_list:
            p, q = q, cls.LOGIC_TABLE[op][p][q]
        return q
