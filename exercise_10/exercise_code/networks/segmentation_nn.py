"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        # Input size is (n 3 240 240)
        pretrained = models.efficientnet_b0(pretrained=True, progress=True).eval()
        for param in pretrained.parameters():
            param.requires_grad = False
        
        # pretrained encoder from (n 3 240 240) to (n 1280 8 8)
        self.en = pretrained.features 
        
        # Decoder_1 from (n 1280 8 8) to (n 512 17 17)
        self.de1=nn.Sequential(

            nn.ConvTranspose2d(1280, 512, kernel_size=3, stride=2),
        )
        
        # Decoder_2 from (n 512 17 17) to (n 256 35 35)
        self.de2=nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2),
        )
        
        # Decoder_3 from (n 256 35 35) to (n 128 71 71) to (n 64 143 143)
        self.de3=nn.Sequential(
            #nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            #nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            torch.nn.Upsample(size=(240, 240)),
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x=self.en(x)
        x=self.de1(x)
        x=self.de2(x)
        x=self.de3(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
