import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SRMD(pl.LightningModule):
    # number of conv layers = 12; given in paper
    def __init__(self, conv_layers=12, channels=18, conv_dim=128, scale_factor=2): 
        super(SRMD, self).__init__()
        
        self.nonlinear_mapping = self.create_layers(conv_layers, channels, conv_dim)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(conv_dim, 3*scale_factor**2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.nonlinear_mapping(x)
        x = self.conv_layer(x)
        return x

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimiser

    def training_step(self, batch, batch_idx):
        # fill training step here
        x, y = next(batch)
        out = self.forward(x)
        loss = F.mse_loss(out, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = next(batch)
        out = self.forward(x)
        val_loss = F.mse_loss(out, y)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = next(batch)
        out = self.forward(x)
        test_loss = F.mse_loss(out, y)
        self.log('test_loss', test_loss)
        return test_loss

    def create_layers(self, conv_layers, channels, conv_dim):
        layers = []
        in_dim = channels
        for i in range(conv_layers):
            conv2d = nn.Conv2d(in_dim, conv_dim, kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(conv_dim)
            relu = nn.ReLU()

            layers += [conv2d, bn, relu]

            in_dim = conv_dim

        return nn.Sequential(*layers)