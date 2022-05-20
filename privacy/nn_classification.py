
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
            sentences.append(entry)

    index2label = dataset.index2label
    queries_lst = [v for k, v in index2label.items()]
    return sentences, queries_lst


def save_results(args, sentences, predictions, idx2label, queries_text):
    correct = defaultdict(list)
    example2pred = {}
    for i,(sentence, pred) in enumerate(zip(sentences, predictions)):
        text = sentence['text']
        gold = sentence['gold']
        gold = int(gold)
        pred = int(pred)
        correct[gold].append(gold == pred)
        example2pred[i] = {
            "input": text,
            "gold": idx2label[gold],
            "pred": idx2label[pred],
            "uid": sentence['uid'],
            'exidx': sentence['exidx']

        }

    total_crct = []
    cls2acc = {}
    for key, value in correct.items():
        label_name = queries_text[key]
        total_crct.extend(value)
        acc = len([c for c in value if c == True])/len(value)
        cls2acc[label_name] = acc
        print(f"Label: {label_name}, Accuracy: {acc}, for {len(value)} examples.")
    print()
    total_acc = len([c for c in total_crct if c == True])/len(total_crct)
    cls2acc['total'] = total_acc
    print(f"Total: {total_acc}, for {len(total_crct)} examples.")

    if not os.path.exists(f"results_prompting/{args.dataset}/"):
        os.makedirs(f"results_prompting/{args.dataset}/")
    with open(f"results_prompting/{args.dataset}/{args.paradigm}_{args.model}_{args.seed}_example2preds.json", "w") as f:
        json.dump(example2pred, f)
    with open(f"results_prompting/{args.dataset}/{args.paradigm}_{args.model}_{args.seed}_cls2acc.json", "w") as f:
        json.dump(cls2acc, f)


def get_dataset_embeddings(args, sentences, queries_lst, model):
    # Load existing embeddings
    embeddings_dir = f"{args.embeddings_dir}/{args.dataset}/"
    embedding_fname = f'd={args.dataset}-s={args.split}-m={args.model}-seed{args.seed}.pt'
    embedding_path = os.path.join(embeddings_dir, embedding_fname)
    if os.path.exists(embedding_path):