
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

from galaxy_generator.models.vae import VAE
from galaxy_generator.utils import display_layer_dimensions

class VAE_Generator(BaseTrainer):

    def __init__(self, config):

        super().__init__(config)
        self._prepare_data()
        self._build_model(config)

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
    
    def _build_model(self, config):

        self.model = VAE(config).to(self.device)

        print('\n------ Build Model ------\n')
        print('Number of trainable parameters')
        print('Encoder  :', sum(param.numel() for param in self.model.encoder.parameters() if param.requires_grad))
        print('Decoder  :', sum(param.numel() for param in self.model.decoder.parameters() if param.requires_grad))

        print('\n------ Encoder Output Layer Dimensions ------\n')
        display_layer_dimensions(self.model.encoder, (1, self.n_channel, self.input_size, self.input_size))

        print('\n------ Decoder Output Layer Dimensions ------\n')
        display_layer_dimensions(self.model.decoder, (1, self.n_zlatent))
