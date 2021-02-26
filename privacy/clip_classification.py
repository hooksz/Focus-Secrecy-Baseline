
import json
import os
import csv
import sys
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
from sklearn.linear_model import LogisticRegression

import torch
from privacy.clip import clip
from PIL import Image
import glob


def classify_embeddings(clip_model,
                        image_embeddings, 
                        text_descriptions,
                        args,
                        temperature=100.):
    
    text_tokens = clip.tokenize(text_descriptions)
    clip_model.to(args.device)
    clip_model.eval()
    with torch.no_grad():
        _image_embeddings = (image_embeddings / 
                             image_embeddings.norm(dim=1, keepdim=True))
        
        text_tokens = text_tokens.to(args.device)
        text_embeddings = clip_model.encode_text(text_tokens).float().cpu()
        _text_embeddings = (text_embeddings / 
                            text_embeddings.norm(dim=1, keepdim=True))
        
        cross = temperature * _image_embeddings @ _text_embeddings.T
        text_probs = cross.softmax(dim=-1)
        _, predicted = torch.max(text_probs.data, 1)
    clip_model.cpu()
    return predicted.cpu().numpy()


def logreg(train_embeddings, train_sentences, test_embeddings):
    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    train_labels = [torch.Tensor([entry['gold']]) for entry in train_sentences]
    train_labels = torch.cat(train_labels).cpu().numpy()
    classifier.fit(train_embeddings, train_labels)
    
    predictions = classifier.predict(test_embeddings)
    return predictions


def save_logreg_results(args, sentences, predictions, queries_text):
    test_labels = [torch.Tensor([entry['gold']]) for entry in sentences]
    test_labels = torch.cat(test_labels).cpu().numpy()
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")

    correct = defaultdict(list)
    example2pred = {}
    for i,(sentence, pred) in enumerate(zip(sentences, predictions)):
        gold = sentence['gold']
        gold = int(gold)
        pred = int(pred)
        correct[gold].append(gold == pred)
        example2pred[i] = {
            "pred": queries_text[pred],
            "gold": queries_text[gold],
            "input": sentence["imgpath"]
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
    with open(f"results_prompting/{args.dataset}/{args.clip_method}_{args.model}_example2preds.json", "w") as f:
        json.dump(example2pred, f)
    with open(f"results_prompting/{args.dataset}/{args.clip_method}_{args.model}_cls2acc.json", "w") as f:
        json.dump(cls2acc, f)


def get_dataset_embeddings(args, dataloader, model, split="test"):
    # File paths
    embeddings_dir = f"{args.embeddings_dir}/{args.dataset}/"
    embedding_fname = f'd={args.dataset}-s={split}-m={args.model}-seed={args.seed}.pt'
    embedding_path = os.path.join(embeddings_dir, embedding_fname)

    # Load existing embeddings
    if os.path.exists(embedding_path):
        print(f'-> Retrieving image embeddings from {embedding_path}!')
        embeddings = torch.load(embedding_path)
        sentences = prepare_data(dataloader)
        return torch.load(embedding_path), sentences

    # Compute the dataset embeddings now
    all_embeddings = []
    model.to(args.device)
    model.eval()
    count = 0
    with torch.no_grad():
        sentences = []
        for ix, data in enumerate(tqdm(dataloader, 
                                       desc=f'Computing CLIP image embeddings for {split} split')):

            if args.dataset != "cifar10":
                inputs, labels, fpath, data_ix = data
            else:
                inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            embeddings = model.encode_image(inputs).float().cpu()
            all_embeddings.append(embeddings)
            inputs = inputs.cpu()