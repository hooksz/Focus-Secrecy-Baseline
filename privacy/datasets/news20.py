
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
import random
import torch


class News20():

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

    def __init__(self, data_path, partition_path, args, dataset='train'):
        
        self.data_file = dataset 
        self.data_path = data_path
        self.partition_path = partition_path

        # load data and targets
        self.data, self.targets, self.user_ids = self.load_meta_data(args, self.data_path, self.partition_path, dataset)

        if args:
            self.model_name = args.model
        self.retrievals = None
        self.modality = "nlp"

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input, target) where target is index of the target class.
        """

        input, target = self.data[index], self.targets[index]
        fpath= ""
        uid = self.user_ids[index]

        return input, target, fpath, index, uid

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.data_path)))


    def get_prompt(self, input="", label="", incontext={}):
        if "gpt" in self.model_name:
            selected_topics = [value for key, value in self.index2label.items()]
            instruction = "Is the topic "
            for topic in selected_topics[:-1]:
                instruction += f"{topic}, "
            instruction += f"or {selected_topics[-1]}?\n\n"

            clean_input = self.parse_document(input)['lines']
            clean_input = self.clean_lines(clean_input)

            token_text = self.tokenizer(clean_input, return_tensors="pt", truncation=True, padding=True)
            token_text = self.tokenizer.convert_ids_to_tokens(token_text.input_ids[0])
            token_text = token_text[:self.max_length]
            token_text = " ".join(token_text)

            truncated_text = clean_input[:len(token_text)]
            clean_input = f"{instruction}{truncated_text}"
            self.instruction_end = "\nTopic:"
            base_prompt = f"{clean_input}{self.instruction_end}"

        elif "t0" in self.model_name: