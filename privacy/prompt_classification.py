
import json
import os
import csv
import sys
import h5py
from collections import defaultdict, Counter
from transformers import pipeline
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import sys


def run_prompt_classification(args, model, dataset):
    result_path = args.result_path
    result_path = f"{result_path}/{args.dataset}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    results = []
    to_encode = []
    batch_labels = []
    batch_size = args.batch_size
    file = 0
    print("\nRunning generation given prompts...")
    model.eval()
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, timeout=60, num_workers=1, drop_last=False)
        for bix, data in tqdm(enumerate(dataloader)):
            for i in range(len(data[0])):
                input = data[0][i]
                label = data[1][i]
                index = data[-1][i]
                incontext=defaultdict(list)
                input = dataset.get_prompt(input=input, incontext={})            
                
                to_encode.append(input)
                batch_labels.append(label)
                if len(to_encode) == batch_size or (len(results) + len(to_encode) == len(dataset)):
                    inputs = dataset.tokenizer(to_encode, return_tensors="pt", truncation=True, padding=True)