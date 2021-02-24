
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
        