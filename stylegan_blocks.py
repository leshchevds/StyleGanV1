import random
import torch as th
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils


class AffineStyle(nn.Module):
    '''
    Learned affine transform A, this module is used to transform
    midiate vector w into a style vector
    '''
    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = nn.Linear(dim_latent, n_channel * 2)
        
        # "the biases associated with ys that we initialize to one"
        self.transform.bias.data[:n_channel] = 1
        self.transform.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style
    
    
# AdaIn (AdaptiveInstanceNorm)
class AdaIn(nn.Module):
    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)
        
    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        return self.norm(image) * factor + bias
    

class ScaleNoize(nn.Module):
    '''
    Learned per-channel scale factor, used to scale the noise
    '''
    def __init__(self, n_channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((1, n_channel, 1, 1)))
    
    def forward(self, noise):
        result = noise * self.weight
        return result 
    

class ConstStyleConvBlock(nn.Module):
    '''
    This is the very first block of generator that get the constant value as input
    '''
    def __init__ (self, n_channel, dim_latent, dim_input):
        super().__init__()

        self.constant = nn.Parameter(torch.randn(1, n_channel, dim_input, dim_input))

        self.style1   = AffineStyle(dim_latent, n_channel)
        self.style2   = AffineStyle(dim_latent, n_channel)

        self.noise1   = ScaleNoize(n_channel)
        self.noise2   = ScaleNoize(n_channel)

        self.adain    = AdaIn(n_channel)
        self.lrelu    = nn.LeakyReLU(0.2, inplace=True)
        self.conv     = nn.Conv2d(n_channel, n_channel, kernel_size=3, padding=1)
    
    def forward(self, w, noise):
        
        result = self.constant.repeat(noise.shape[0], 1, 1, 1) + self.noise1(noise)
        result = self.adain(result, self.style1(w))
        result = self.lrelu(result)
        
        result = self.conv(result) + self.noise2(noise)
        result = self.adain(result, self.style2(w))
        result = self.lrelu(result)
        
        return result
    

class StyleConvBlock(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''
    def __init__ (self, in_channel, out_channel, dim_latent):
        super().__init__()
        
        self.style1   = AffineStyle(dim_latent, out_channel)
        self.style2   = AffineStyle(dim_latent, out_channel)

        self.noise1   = ScaleNoize(out_channel)
        self.noise2   = ScaleNoize(out_channel)

        self.adain    = AdaIn(out_channel)
        self.lrelu    = nn.LeakyReLU(0.2, inplace=True)

        self.conv1    = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2    = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
    
    def forward(self, image, w, noise):
        # Upsample: Proxyed by generator
        # result = nn.functional.interpolate(previous_result, scale_factor=2, mode='bilinear',
        #                                           align_corners=False)

        # Gaussian Noise: Proxyed by generator
        # noise1 = torch.normal(mean=0,std=torch.ones(result.shape)).cuda()
        # noise2 = torch.normal(mean=0,std=torch.ones(result.shape)).cuda()
        
        result = self.conv1(image) + self.noise1(noise)
        result = self.adain(result, self.style1(w))
        result = self.lrelu(result)
        
        result = self.conv2(result) + self.noise2(noise)
        result = self.adain(result, self.style2(w))
        result = self.lrelu(result)
        
        return result
    
    
class ConvBlock(nn.Module):
    '''
    Used to construct progressive discriminator
    '''
    def __init__(self, in_channel, out_channel, size_kernel1, padding1, 
                 size_kernel2 = None, padding2 = None):
        super().__init__()
        
        size_kernel2 = size_kernel1 if size_kernel2 is None else size_kernel2
        padding2 = padding1 if padding2 is None else padding2
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, size_kernel1, padding=padding1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel, out_channel, size_kernel2, padding=padding2),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, image):
        return self.model(image)
    

# Normalization on every element of input vector
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class LatentMapping(nn.Module):
    '''
    A mapping consists of multiple fully connected layers.
    Used to map the input to an intermediate latent space W.
    '''
    def __init__(self, n_fc, dim_latent):
        super().__init__()
        
        layers = [PixelNorm()]
        for _ in range(n_fc):
            layers.append(nn.Linear(dim_latent, dim_latent))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.mapping(z)  
    