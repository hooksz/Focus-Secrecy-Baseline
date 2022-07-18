import torch
import sys
import os
import logging
import numpy as np
import json
import random
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, DataLoader
import torchvision
from collections import defaultdict
from tqdm import tqdm
from torchvision import datasets, transforms

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoModelForSequenceClassification
from transformers import  GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForCausalLM

import torch
from privacy.clip import clip
from PIL import Image
import glob

from privacy.datasets.celeba import CelebA
from privacy.datasets.sent140 import Sent140
from privacy.datasets.news20 import News20
from privacy.datasets.femnist import FEMNIST
from privacy.datasets.reddit import Reddit

API_MODELS = ["gpt175", "gpt6.7"]


def get_model(args):
    print("Loading model...")
    if "clip" in args.model:
        if args.model == "clip32B":
            clip_variant = "ViTB32"
        elif args.model == "clip16B":
            clip_variant = "ViTB16"
        elif args.model == "clip336":
            clip_variant = "ViTL14"
        elif args.model == "clipres101":
            clip_variant = "RN101"
        else:
            assert 0, print("Unsupported clip variant")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, transform = clip.load(clip_variant, device=device)
        tokenizer = None
    elif args.model == "dpr":
        tokenizer = None
        transform = None
        model = SentenceTransformer(("multi-qa-mpnet-base-dot-v1"))
    elif "t0" in args.model:
        # T0 3Bn Model
        transform = None 
        if args.model == "t0pp":
            t0_variant = "bigscience/T0pp"
        elif args.model == "t03b":
            t0_variant = "bigscience/T0_3B"
        else:
            assert 0, print("Unsupported t0 variant.")
        tokenizer = AutoTokenizer.from_pretrained(t0_variant, cache_dir=args.cache_dir)
        tokenizer.padding_side = "left"
        model = AutoModelForSeq2SeqLM.from_pretrained(t0_variant, cache_dir=args.cache_dir)
    elif "gpt" in args.model:
        transform = None

        if args.model in API_MODELS:
            return None, None, None 

        if args.model == "gpt2.7":
             gpt_variant = 'EleutherAI/gpt-neo-2.7B'
        elif args.model == "gpt1.3":
            gpt_variant = 'EleutherAI/gpt-neo-1.3B'
        elif args.model == "gpt125m":
            gpt_variant = 'EleutherAI/gpt-neo-125M'
        else:
            assert 0, print("Unsupported gpt variant.")

        tokenizer = AutoTokenizer.from_pretrained(gpt_variant, max_token_length=512, cache_dir=args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            gpt_variant, 
            pad_token_id=tokenizer.eos_token_id, 
            cache_dir=args.cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    elif "bert" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
        transform = None
        model = None
    else:
        assert 0, print("Unsupported model.")

    if args.use_gpu:
        model = model.to(args.device)

    return model, transform, tokenizer


def get_dataset(args, split="", transform=None, tokenizer=None):
    print("\nLoading dataset...")
    dataset = args.dataset

    if dataset == "sent140":
        data_prefix = f"{args.public_datasets_prefix}/leaf/data/"
        data_path = f"{data_prefix}/sent140/data/train/all_data_niid_0_keep_0_train_9.json"
        training_dataset = Sent140(data_path, args, dataset="train")

        data_path = f"{data_prefix}/sent140/data/test/all_data_niid_0_keep_0_test_9.json"
        test_dataset = Sent140(data_path, args, dataset="test")
        test_dataset.get_incontext_examples(args, training_dataset)

        training_dataset.tokenizer = tokenizer
        training_dataset.transform = transform
        test_dataset.tokenizer = tokenizer
        test_dataset.transform = transform

    elif dataset == "reddit":
        data_prefix = f"{args.public_datasets_prefix}/leaf/data/"
        data_path = f"{data_prefix}/reddit/data/train/train_data.json"
        training_dataset = Reddit(data_path, args, dataset="train")

        data_path = f"{data_prefix}/reddit/data/test/test_data.json"
        test_dataset = Reddit(data_path, args,