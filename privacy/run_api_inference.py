
import json
import pathlib, os
from tqdm import tqdm
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
import openai

def run_api_inference(args, dataset):
    # format examples
    print(f"\nPreparing dataset for API...")
    examples = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, timeout=60, num_workers=1, drop_last=False)
    for bix, data in tqdm(enumerate(dataloader)):
        for i in range(len(data[0])):
            text = data[0][i]
            label = data[1][i]
            index = data[-1][i]
            incontext=defaultdict(list)
            text = dataset.get_prompt(input=text, incontext={})
            examples.append({
                "key": bix,
                "input": text,