import time

import torch
from thop import profile  #
import tempfile
from torch import nn


class Timer:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self):
        self.t = time.time() - self.t

        if self.verbose:
            print(f"{self.name.capitalize()} | Elapsed time : {self.t:.2f}")


def calc_flops_macs(Model, cfg):
    profile(Model, torch.zeros(1, 1, cfg.n_mels, 100).to(cfg.device))  # -> (6.0 MACs, 3.0 parameters)


def get_size_in_megabytes(model):
    # https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html#look-at-model-size
    with tempfile.TemporaryFile() as f:
        torch.save(model.state_dict(), f)
        size = f.tell() / 2 ** 20
    return size
