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
   