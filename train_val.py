from typing import Tuple, Union, List, Callable, Optional
from tqdm import tqdm
from itertools import islice
import pathlib
import dataclasses

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from metrics import count_FA_FR, get_au_fa_fr
from config import TaskConfig


# https://github.com/cxa-unique/Simplified-TinyBERT/blob/main/task_distill_simplified.py inspired
def train_epoch_distillation(model, teacher, opt, loader, log_melspec, device, config: TaskConfig):
    model.train()
    for i, (batch, labels) in tqdm(enumerate(loader), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        opt.zero_grad()

        # run model # with autocast():
        student_logits = model(batch)
        with torch.no_grad():
            teacher_logits = teacher(batch) / config.temperature

        hard_loss = torch.nn.functional.cross_entropy(student_logits, labels, reduction='mean')

        student_likelihood = torch.nn.functional.log_softmax(student_logits / config.temperature, dim=-1)
        targets_prob = torch.nn.functional.softmax(teacher_logits / config.temperature, dim=-1)
        soft_loss = (- targets_prob * student_likelihood).mean()

        loss = config.alpha * soft_loss + (1 - config.alpha) * hard_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        opt.step()






