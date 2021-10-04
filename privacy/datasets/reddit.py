from __future__ import print_function
import warnings
import os
import os.path
import json
import csv
import sys
import h5py
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
import torch
import random
import math

class Reddit():

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

        self.index2label = {}

        # load data and targets
        self.data, self.targets, self.user_ids, self.subreddits, self.target2name_map = self.load_meta_data(args, self.data_path, dataset)
        self.modality="nlp"
        self.retrievals = []

    def __getitem__(self, index):
        input, target = self.data[index], self.targets[index]
        fpa