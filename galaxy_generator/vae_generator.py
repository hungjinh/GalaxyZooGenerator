
import os
import time
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from galaxy_generator.base import BaseTrainer
from galaxy_generator.data_kits import data_split, GalaxyZooDataset, transforms_DCGAN
from torch.utils.data import DataLoader


class VAE_Generator(BaseTrainer):

    def __init__(self, config):

        super().__init__(config)
        self._prepare_data()

    def _prepare_data(self):

        self.df = {}
        self.df['train'], self.df['valid'] = data_split(self.file_csv,
                                                        self.f_train, self.f_valid, 0, 
                                                        random_state=self.seed, stats=False)

        self.transform = transforms_DCGAN(self.input_size, self.crop_size, self.norm_mean, self.norm_std)

        self.dataset = {}
        self.dataloader = {}
        for key in ['train', 'valid']:
            self.dataset[key] = GalaxyZooDataset(self.df[key], self.dir_image, transform=self.transform, label_tag=self.label_tag)
            self.dataloader[key] = DataLoader(self.dataset[key], batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

        print('\n------ Prepare Data ------\n')
        for key in ['train', 'valid']:
            print(f'Number of {key} galaxies: {len(self.dataset[key])} ({len(self.dataloader[key])} batches)')
