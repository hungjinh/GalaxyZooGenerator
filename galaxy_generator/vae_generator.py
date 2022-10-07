
import os
import time
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from galaxy_generator.base import BaseTrainer
from galaxy_generator.data_kits import data_split, GalaxyZooDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class VAE_Generator(BaseTrainer):

    def __init__(self, config):

        super().__init__(config)
        self._prepare_data()

    def _prepare_data(self):

        self.df = {}
        self.df['train'], self.df['valid'] = data_split(self.file_csv,
                                                        self.f_train, self.f_valid, 0, 
                                                        random_state=self.seed, stats=False)

        self.transform = transforms.Compose([transforms.CenterCrop(self.crop_size),
                                             transforms.Resize(self.input_size),
                                             transforms.RandomRotation(90),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomVerticalFlip(),
                                             transforms.RandomResizedCrop(
                                                self.input_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                                             transforms.ToTensor()
                                            ])

        self.dataset = {}
        self.dataloader = {}
        for key in ['train', 'valid']:
            self.dataset[key] = GalaxyZooDataset(self.df[key], self.dir_image, transform=self.transform, label_tag=self.label_tag)
            self.dataloader[key] = DataLoader(self.dataset[key], batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

        print('\n------ Prepare Data ------\n')
        for key in ['train', 'valid']:
            print(f'Number of {key} galaxies: {len(self.dataset[key])} ({len(self.dataloader[key])} batches)')
    

