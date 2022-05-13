
from __future__ import print_function
import warnings
import os
import os.path
import json
import csv
import sys
import h5py
import random
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
import torch

class Sent140(torch.utils.data.Dataset):

    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, data_path, args, dataset='train', transform=None, tokenizer=None):
        
        self.data_file = dataset 
        self.data_path = data_path

        # labels to strings
        self.index2label = {
            0: "negative",
            1: "positive"
        }
        self.model_name = args.model

        # load data and targets
        self.data, self.targets, self.user_ids, self.target2name_map = self.load_meta_data(args, self.data_path, dataset)
        self.modality="nlp"
        self.retrievals = []
        self.prompt_prefix = ""
        self.prompt_choice = args.prompt_choice

    def __getitem__(self, index):
        input, target, uid = self.data[index], int(self.targets[index]), self.user_ids[index]
        fpath= ""

        return input, target, fpath, index, uid
