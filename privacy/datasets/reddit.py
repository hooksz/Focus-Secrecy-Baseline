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
        fpath= ""
        return input, target, fpath, index

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.data_path)))

    def get_incontext_examples(self, args, training_dataset):
        # for in context random
        if "incontext" in args.prompt_choice:

            # first collect incontext examples using the desired strategy
            if args.prompt_choice == "random_incontext":
                training_users2sents = defaultdict(list)
            elif args.prompt_choice == "random_incontext_noprivacy":
                training_sents = []
                training_users2counts = defaultdict(int)
            else:
                training_users2sents = defaultdict(dict)
            for data, label, uid, subreddit in zip(training_dataset.data, training_dataset.targets, training_dataset.user_ids, training_dataset.subreddits):
                example = {
                    "input": data,
                    "label": label
                }

                # collect by user in aggregate
                if args.prompt_choice == "random_incontext": 
                    training_users2sents[uid].append(example)
                
                # no privacy, combine all user examples
                elif args.prompt_choice == "random_incontext_noprivacy":
                    training_sents.append(example)
                    training_users2counts[uid] += 1

                # collect by user, split by subreddit
                else:
                    if subreddit not in training_users2sents[uid]:
                        training_users2sents[uid][subreddit] = []
                    training_users2sents[uid][subreddit].append(example)

            # next assign in context examples to training examples
            test2examples = []
            if args.prompt_choice in ["random_incontext_noprivacy", "incontext"]:
                random.seed(0)
            for data, uid in zip(self.data, self.user_ids):
                # get the incontext examples
                if args.prompt_choice == "random_incontext_noprivacy":
                    user_count = training_users2counts[uid]
                    user_entry = random.sample(training_sents, min(user_count, args.num_incontext))
                else:
                    train_sents = training_users2sents[uid]
                    if args.prompt_choice == "random_incontext":
                        user_entry = train_sents[0:args.num_incontext]
                    else:
                        user_entry = []
                        leftovers = []
                        total_train = sum([len(lst) for sbr, lst in train_sents.items()])
                        for subreddit, lst in train_sents.items():
                            num = math.ceil((len(lst)/total_train)*(args.num_incontext))
                            user_entry.extend(random.sample(lst, min(len(lst), num)))
                        if len(user_entry) > args.num_incontext:
                            user_entry = random.sample(user_entry, args.num_incontext)
                user_text = [f"{entry['input']} {entry['label']}." for entry in user_entry]

                # clean
                user_text = " ".join(user_text).replace("<PAD> ", "").replace("<PAD>", "")
                user_text = user_text.replace("<EOS>", "")
                user_text = user_text.replace(" . ", " ")
                user_text = user_text.replace("  ", " ")
                user_text = user_text.replace("\n", " ")
                user_text = user_text.replace("\t", " ")

                test2examples.append(f"{user_text}{data}")
            self.data = test2examples


    def get_prompt(self, input="", incontext={}):
        prefix = ""
        base_prompt = f"{prefix}{input}"

        return base_prompt


    def clean_text(self, text):
        text = [t for t  in text if t not in ["<BOS>", "<EOS>"]]