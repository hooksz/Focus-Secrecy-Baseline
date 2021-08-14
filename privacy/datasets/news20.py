
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
            selected_topics = [value for key, value in self.index2label.items()]
            instruction = "What label best describes text ("
            for topic in selected_topics[:-1]:
                instruction += f"{topic} ||| "
            instruction += f"||| {selected_topics[-1]})?"

            clean_input = self.parse_document(input)['lines']
            clean_input = self.clean_lines(clean_input)
            token_text = self.tokenizer(clean_input, return_tensors="pt", truncation=True, padding=True)
            token_text = self.tokenizer.convert_ids_to_tokens(token_text.input_ids[0])
            token_text = token_text[:self.max_length]
            token_text = " ".join(token_text)
            truncated_text = clean_input[:len(token_text)]

            base_prompt = f"Text: {truncated_text} \n{instruction}\n"
        else:
            assert 0, print("Unsupported operation in 20news.")

        return base_prompt


    def parse_document(self, doc):
        key_parts = ["Subject: Re:", "Organization", "Lines"]
        parts_dict = {}
        
        parts = doc.split(key_parts[0])
        parts_dict['from'] = parts[0]
        
        try:
            parts = parts[1].split(key_parts[1])
        except:
            parts = doc.split(key_parts[1])
        parts_dict['subject'] = parts[0]
        
        try:
            parts = parts[1].split(key_parts[2])
        except:
            parts = doc.split(key_parts[2])
        parts_dict['org'] = parts[0]
        
        try:
            parts_dict['lines'] = parts[1][4:]
        except:
            parts = doc.split(key_parts[2])
            try:
                parts_dict['lines'] = parts[1][4:]
            except:
                parts_dict['lines'] = doc
        
        return parts_dict


    def clean_lines(self, lines):
        lines = lines.replace("*", "")
        lines = lines.replace("|", "")
        lines = lines.replace(">", "")
        lines = lines.replace("<", "")
        lines = lines.replace("---", "")
        lines = lines.replace("^", "")
        lines = lines.replace("\t", "")
        
        clean_lines = []
        for wd in lines.split():
            if "@" not in wd and ".com" not in wd:
                clean_lines.append(wd)
                
        lines = " ".join(clean_lines)
        
        ines = lines.replace("   ", " ")
        lines = lines.replace("  ", " ")
        return lines    

    # convert raw label names to label descriptions
    def clean_label(self, label):
        category2word = {
                "comp": "",
                "alt": "",
                "misc": "",
                "sci": "",
                "talk": "",
                "rec": "",
                "soc": "",
                "sport": "",
                "autos": "automobiles",
                "med": "medical",
                "crypt": "cryptography security",
                "mideast": "middle east",
                "sys": "",
                "forsale": "sale",
        }
                
        label_clean = label.replace("religion.christian", "christianity")
        wds = label_clean.split(".")
        clean_wds = []
        for wd in wds:
            if wd in category2word:
                wd = category2word[wd]
                if wd:
                    clean_wds.append(wd)
            else:
                clean_wds.append(wd)
        label_text =  " ".join(clean_wds)
        return label_text


    def get_news20_iserror(self, topics, result, gold):
        error = 0
        result = result.split(".")[0].strip()
        result = result.lower()
        found_topics = [topic for topic in topics if topic in result]
        if not result or gold.lower() not in result.split():
            error = 1
        elif len(found_topics) > 1:
            error = 1
        
        if len(found_topics) > 1:
            if "politics" in found_topics and result != "politics" and gold != "politics" and "politics" in gold and gold in result:
                error = 0    
        result_fix = result + "s" 
        if result_fix == gold:
            error = 0
        elif result == "medicine" and gold == "medical":
            error = 0
        elif result == "cars" and gold == "automobiles":
            error = 0

        return error, result