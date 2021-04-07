from __future__ import print_function
import warnings
from PIL import Image
import os
import os.path
import csv
from torchvision import transforms
from tqdm import tqdm
import json

class CelebA():

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

    def __init__(self, root, args, dataset='train', transform=None, target_transform=None, imgview=False):
        
        self.data_file = dataset # 'train', 'test', 'validation'
        self.root = root

      