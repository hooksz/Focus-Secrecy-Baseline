
import json
import os
import csv
import sys
import h5py
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoModelForSequenceClassification
from transformers import set_seed
from sentence_transformers import SentenceTransformer, util

from privacy.prompt_classification import run_prompt_classification
from privacy.nn_classification import run_similarity_classification
from privacy.clip_classification import run_image_similarity_classification
from privacy.run_api_inference import run_api_inference
from privacy.utils import get_zeroshot_predictions, get_dataset, initialize_run


def get_args():
    parser = argparse.ArgumentParser(description='')

    # Dataset
    parser.add_argument('--seed', type=int, default=0, help="seed")
    parser.add_argument('--dataset', type=str, default='sent140', choices=["sent140", "amazon", "femnist", "20news", "synthetic_20news", "agnews", "celeba", "civilcomments", "mnist", "cifar10", "reddit"])
    parser.add_argument('--model', type=str, choices=["clip32B", "clip16B", "clip336", "clipres101", "t03b", "t0pp", "dpr", "bart", "bert-base-uncased", "gpt2.7", "gpt1.3", "gpt125m", "gpt175", "gpt6.7"])
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--paradigm', type=str, default="prompt", choices=["prompt", "similarity"])
    parser.add_argument('--dataset_size', default=0, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--use_gpu', default=0, type=int, help="Whether to use the GPU")
    parser.add_argument('--client_subsample', default=1, type=float, help="Whether to use the GPU")

    parser.add_argument('--dataset_subsize', default=0, type=int, help="Want to subsample the dataset for faster iteration.")
    
    # For prompting
    parser.add_argument('--prompt_choice', default="instruction_prompt", choices=["use_retrieval", "fixed_prompt", "random_prompt", "instruction_prompt", "random_incontext", "random_incontext_noprivacy", "incontext"],
                        help="Whether to pull in-context examples.")
    parser.add_argument('--batch_size', default=2, type=int, help="Batch size for prompting inference")
    parser.add_argument('--num_incontext', default=0, type=int, help="Number of examples to put in context")
    parser.add_argument('--max_sequence_length', default=128, type=int, help="Max sequence length for prompting inference")
    parser.add_argument('--openai_key', default="", type=str, help="OpenAI inference API key, you can obtain yours here https://openai.com/api/")

    # For similarity search
    parser.add_argument('--normalize_embedding_input', default=False,
                        action='store_true')
    parser.add_argument('--clip_method', default="zeroshot", choices=["zeroshot", "logreg"])

    # For loading Hugging Face models
    parser.add_argument('--cache_dir', 
                        default=f'/cache/',
                        type=str)

    parser.add_argument('--result_path', 
                        default=f'/results_prompting/',
                        type=str)

    parser.add_argument('--embeddings_dir', 
                        default=f'/embeddings/',