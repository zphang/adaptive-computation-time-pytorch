import data
import train
import configuration
import models

data_manager = data.ParityDataManager
config = configuration.Config()

config.parity_data_len = 100000
config.train_log = False
config.cuda = False

model = models.ParityModel(config=config)
if config.cuda:
    model = model.cuda()
model.share_memory()


import multiprocessing as mp
processes = []
for rank in range(8):
    p = mp.Process(target=train.train, args=(rank, config, model, data_manager))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
