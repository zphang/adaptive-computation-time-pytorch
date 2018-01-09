# Adaptive Computation Time

This is an implementation of [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) (Graves, 2016) in PyTorch.

### Introduction

*Adaptive Computation Time* is a drop-in replacement for RNNs structures that allows the model to process multiple time steps on a single input token. More information can be found in the paper, or in this blog post:

### Requirements

* Python 3.6
* PyTorch 0.3.0
* `matplotlib`, `argparse`

### Experiments

I am still in the process of replicating the experiments described in the paper.

- [x] Bit Parity
- [x] Logical Gates
- [ ] Addition
- [ ] Sorting
- [ ] Word Prediction

### Usage

1. Git clone this repository
2. Train/Evaluate the model on a given task/parameter setting:
    * E.g.
    
    ```bash
    python run_train.py \
      --task=parity \
      --use_act=False \
      --model_save_path="outputs/models/parity/rnn"
    ```    
    
    ```bash
    python run_train.py \
      --task=parity \
      --use_act=True \
      --act_ponder_penalty=0.001 \
      --model_save_path="outputs/models/parity/act_0.001"
    ```    