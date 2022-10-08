
import os
import copy
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

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
class VAE_Generator(BaseTrainer):

    def __init__(self, config):

        super().__init__(config)
        self._prepare_data()
        self._build_model(config)
        self._init_optimizer()

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

    def loss(self, x_hat, x, mu, logvar):
        '''
            Parameters:
                x_hat : generator image
                x     : target image
            
            Note : 
                Use BCE because each pixel in x or x_hat ranges from [0,1].
                KLD : KL divergence = 0.5 * sum(σ^2 - log(σ^2)-1 + μ^2)
                      logvar.exp() = σ^2
        '''

        BCE = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
        return BCE + KLD

    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def _init_storage(self):
        '''initialize storage dictionary and directory to save training information'''

        # ------ create storage directory ------
        print('\n------ Create experiment directory ------\n')
        try:
            os.makedirs(self.dir_exp)
        except (FileExistsError, OSError) as err:
            raise FileExistsError(f'Default save directory {self.dir_exp} already exit. Change exp_name!') from err
        print(f'Training information will be stored at :\n \t {self.dir_exp}\n')
        
        # ------ create 'genImgs' directory under self.dir_exp to store generator images
        self.dir_genImgs = os.path.join(self.dir_exp, 'genImgs')
        os.makedirs(self.dir_genImgs)

        # ------ create 'checkpoints' directory to store 'self.statInfo_{epochID}.pth'
        os.makedirs(self.dir_checkpoints)

        # ------ trainInfo ------
        save_key = ['train_loss', 'valid_loss', 'epoch_train_loss',
                    'epoch_valid_loss', 'lr', 'valid_means', 'valid_logvars', 'valid_labels']
        self.trainInfo = {}
        for key in save_key:
            self.trainInfo[key] = []

    def _save_checkpoint(self, epochID):

        with open(self.file_trainInfo, 'wb') as handle:
            pickle.dump(self.trainInfo, handle)

        self.statInfo = {
            'epoch': epochID,
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        outfile_statInfo = os.path.join(self.dir_checkpoints, f'stateInfo_{epochID}.pth')
        torch.save(self.statInfo, outfile_statInfo)

    def _train_one_epoch(self):

        # ------ Training Loop ------
        self.model.train()

        running_train_loss = 0.0  # to store the sum of loss for 1 epoch
        for id_batch, x in enumerate(self.dataloader['train']):
            #gal_label = x[1]
            x = x[0].to(self.device)
            # <--- forward --->
            x_hat, mu, logvar = self.model(x)
            loss = self.loss(x_hat, x, mu, logvar)
            
            # <--- backward --->
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.trainInfo['train_loss'].append(loss.item()/x.size(0))
            running_train_loss += loss.item()

            self.current_iteration += 1


        avg_epoch_train_loss = running_train_loss/len(self.dataloader['train'].dataset)
        print(f'\t\t avg. train  loss : {avg_epoch_train_loss:.3f}')
        self.trainInfo['epoch_train_loss'].append(avg_epoch_train_loss)

        # ------ Validation Loop ------
        means = []
        logvars = []
        labels = []
        with torch.no_grad():
            self.model.eval() 

            running_valid_loss = 0.0
            for id_batch, x in enumerate(self.dataloader['valid']):
                label = x[1]
                x = x[0].to(self.device)
                x_hat, mu, logvar = self.model(x)
                loss = self.loss(x_hat, x, mu, logvar)

                self.trainInfo['valid_loss'].append(loss.item()/x.size(0))
                running_valid_loss += loss.item()

                means.append(mu.detach())
                logvars.append(logvar.detach())
                labels.append(label.detach())

                # --- show reconstructed images while training ---
                if self.freq_img > 0 and self.current_epoch % self.freq_img == 0 and id_batch == len(self.dataloader['valid'])-2: # Take 2nd to last batch to make the plot. 
                    fig = plt.figure(figsize=(16,8))
                    fig.suptitle(f'EpochID : {self.current_epoch}', fontsize=16)
                    plt.subplot(1, 2, 1)
                    self.display_images(x, self.Ngals)
                    plt.title("Real Galaxies", fontsize=16)
                    plt.subplot(1, 2, 2)
                    self.display_images(x_hat, self.Ngals)
                    plt.title("Reconstructed Galaxies", fontsize=16)

                    file_img = os.path.join(self.dir_genImgs, f'ep{self.current_epoch}.png')
                    fig.savefig(file_img, dpi=self.dpi, transparent=False, facecolor='white')
                    plt.close(fig)
        
        avg_epoch_valid_loss = running_valid_loss/len(self.dataloader['valid'].dataset)
        print(f'\t\t avg. valid  loss : {avg_epoch_valid_loss:.3f}')
        self.trainInfo['epoch_valid_loss'].append(avg_epoch_valid_loss)

        self.trainInfo['valid_means'].append(torch.cat(means))
        self.trainInfo['valid_logvars'].append(torch.cat(logvars))
        self.trainInfo['valid_labels'].append(torch.cat(labels))


    def train(self):

        self._init_storage()
        self.current_epoch = 0
        self.current_iteration = 0
        self.min_valid_loss = float('inf')

        for epochID in range(self.num_epochs):
            self.current_epoch = epochID

            print(f'--- Epoch {epochID+1}/{self.num_epochs} ---')

            since = time.time()
            
            self._train_one_epoch()
            self._save_checkpoint(epochID)

            self.trainInfo['lr'].append(self.scheduler.get_last_lr()[0]) # save lr / epoch
            self.scheduler.step()

            # check if is at the best epoch -> deep copy the model
            if self.trainInfo['epoch_valid_loss'][-1] < self.min_valid_loss:
                self.min_valid_loss = self.trainInfo['epoch_valid_loss'][-1]
                self.trainInfo['best_epochID'] = epochID
                self.trainInfo['best_model_state_dict'] = copy.deepcopy(self.model.state_dict())

            time_elapsed = time.time() - since
            print(f'\tTime: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')

            if epochID - self.trainInfo['best_epochID'] >= self.early_stop_threshold:
                print(f'Early stopping... (Model did not imporve after {self.early_stop_threshold} epochs)')
                break

        print(f'Minimum validation loss {self.min_valid_loss} reached at epoch', self.trainInfo['best_epochID']+1)

    def display_images(self, image_tensor, Ngals, nrow=None):
        '''Display reconstruced galaxy images sampled from validation set
        '''
        if nrow is None:
            nrow = int(np.sqrt(Ngals))
        else:
            nrow = Ngals//nrow

        image_tensor = image_tensor.detach().cpu()
        img_grid = vutils.make_grid(image_tensor[:Ngals], nrow=nrow, padding=1)
        img_grid = img_grid.permute(1, 2, 0)
        plt.axis('off')
        plt.imshow(img_grid)
        plt.tight_layout()

    def gen_galaxy(self, Ngals, epochID, nrow=1):
        ''' Draw fake galaxy images from the trained generator, given epochID

            Return : 
                img_grid : np.array 
                    fake galaxy image that can be displayed with plt.imshow(img_grid). 
        '''

        outfile_statInfo = os.path.join(self.dir_checkpoints, f'stateInfo_{epochID}.pth')
        statInfo = torch.load(outfile_statInfo)

        self.model.load_state_dict(statInfo['model_state_dict'])
        self.model.eval()

        z_random = torch.randn(Ngals, self.n_zlatent, device=self.device)
        fake_gals = self.model.decoder(z_random).detach().cpu()

        img_grid = np.transpose(vutils.make_grid(fake_gals, padding=2, normalize=False, nrow=Ngals//nrow), (1, 2, 0))

        return img_grid