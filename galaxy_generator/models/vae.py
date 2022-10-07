'''Convolutional Variational Autoencoder Network
'''

import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        nc = config.n_channel
        nfe = config.n_filter_E
        nfd = config.n_filter_D
        nz = config.n_zlatent

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nfe, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfe, nfe*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfe*2),
            nn.LeakyReLU(0.2, inplace=True), 

            nn.Conv2d(nfe*2, nfe*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfe*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfe*4, nfe*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfe*8), 
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfe*8, nz*4, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(nz*4, nz*2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(nz, nfd),
            nn.Unflatten(1, (nfd, 1, 1)),

            nn.ConvTranspose2d(nfd, nfd*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nfd*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(nfd*8, nfd*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(nfd*4, nfd*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(nfd*2, nfd, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd),
            nn.ReLU(True),

            nn.ConvTranspose2d(nfd, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        '''
            Note : 
                Work with log(σ^2) to assure that σ>=0.
                And log(var) can range from [-inf, inf].
        '''
        if self.training:
            std = logvar.mul(0.5).exp_()             # σ = exp(1/2 log(σ^2))
            # ε ~ N(0, 1) (dim = std.size())
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)             # out = mu + ε x σ
        else:
            return mu

    def forward(self, x):
        # reshape the encoder output to be (batchsize, 2, nz)
        mu_logvar = self.encoder(x).view(-1, 2, self.nz)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar
