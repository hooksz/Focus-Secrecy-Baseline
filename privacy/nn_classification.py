
import json
import os
import csv
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
from torch.utils.data import DataLoader
import numpy as np
from privacy.utils import get_zeroshot_predictions


def prepare_data(dataset):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True, timeout=60, num_workers=1, drop_last=True)
    sentences = []
    for bix, data in tqdm(enumerate(dataloader)):
        for i in range(len(data[0])):
            input = data[0][i]
            label = data[1][i]
            fpath = data[2][i]
            exidx = int(data[3][i])
            uid = int(data[4][i])
            entry = { "text": input, "gold": label, "exidx": exidx, "uid": uid}