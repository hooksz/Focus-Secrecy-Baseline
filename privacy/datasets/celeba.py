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

        # this is the path to the data downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 
        self.image_path = f"{args.public_dataset_path}/leaf/data/celeba/img_align_celeba"
        
        self.transform = transform
        self.target_transform = target_transform
        self.path = os.path.join(self.processed_folder, self.data_file)
        self.retrievals = None

        # load data and targets
        self.data, self.targets, self.user_ids = self.load_meta_data(args, self.path)
        self.modality="image"

        self.imgview = imgview

    def __getitem__(self, index):
        imgName, target = self.data[index], int(self.targets[index])
        fpath = os.path.join(self.image_path, imgName)
        img = Image.open(fpath)
        img = self.transform(img)

        return img, target, fpath, index

    def __len__(s