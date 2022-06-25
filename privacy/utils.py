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
        model = AutoModelForSeq2S