"""
EEFS by zjc
有些参数我也忘了怎么调的
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from stvit2 import STViT
from firwin import *




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return self.filters,F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.dp = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(1, 5),
                               padding=(0, 1),
                               stride=(1,3))
        self.mp = nn.MaxPool2d((1, 4))  # self.mp = nn.MaxPool2d((1,4))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

        # print('out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        #print('conv2 out',out.shape)
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.dp(out)
        return out
    

def masking(spec, ratio=0.1) :
    mask = np.ones(spec.shape)

    t_times = np.random.randint(3)
    f_times = np.random.randint(3)

    for _ in range(t_times) :
        t = np.random.randint((1-ratio)*mask.shape[0])
        mask[t:t+int(mask.shape[0]*ratio), :] = 0

    for _ in range(f_times) :
        f = np.random.randint((1-ratio)*mask.shape[1])
        mask[:, f:f+int(mask.shape[1]*ratio)] = 0
    inv_mask = -1 * (mask - 1)

    return mask, inv_mask

def get_band(x, min_band_size, max_band_size, band_type, mask):
    assert band_type.lower() in ['freq', 'time'], f"band_type must be in ['freq', 'time']"
    if band_type.lower() == 'freq':
        axis = 2
    else:
        axis = 1
    band_size =  random.randint(min_band_size, max_band_size)
    mask_start = random.randint(0, x.size()[axis] - band_size) 
    mask_end = mask_start + band_size
    
    if band_type.lower() == 'freq':
        mask[:, mask_start:mask_end] = 1
    if band_type.lower() == 'time':
        mask[mask_start:mask_end, :] = 1
    return mask

def specmix(x,y, prob = 0.99, min_band_size = 10, max_band_size = 20, max_frequency_bands=4, max_time_bands=3):
    if prob < 0:
        raise ValueError('prob must be a positive value')

    k = random.random()
    if k > 1 - prob:
        batch_size = x.size()[0]
        batch_idx = torch.randperm(batch_size)
        mask = torch.zeros(x.size()[1:3]).unsqueeze(0).unsqueeze(-1).cuda()
        num_frequency_bands = random.randint(1, max_frequency_bands)
        for i in range(1, num_frequency_bands):
            mask = get_band(x, min_band_size, max_band_size, 'freq', mask)
        num_time_bands = random.randint(1, max_time_bands)
        for i in range(1, num_time_bands):
            mask = get_band(x, min_band_size, max_band_size, 'time', mask)
        lam = torch.sum(mask) / (x.size()[1] * x.size()[2])
        x = x * (1 - mask) + x[batch_idx] * mask
        y = y * (1 - lam) + y[batch_idx] * (lam)
        return x,y
    else:
        return x,y

class EEFS(nn.Module):
    def __init__(self, d_args,device):
        super().__init__()
        self.args = d_args
        self.device = device
        self.sample_rate = 16000


        self.effects = [
            ["lowpass", "-1", "300"],  # apply single-pole lowpass filter
            ["speed", "0.8"],  # reduce the speed
            # This only changes sample rate, so it is necessary to
            # add `rate` effect with original sample rate after this.
            ["rate", f"{self.sample_rate}"],
            ["reverb", "-w"],  # Reverbration gives some dramatic feeling
        ]

        self.conv = nn.Conv2d(1,1,kernel_size=(3,10),stride= (2,5))

        filts =[70, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        self.conv_time = SincConv_fast(out_channels=70,kernel_size=128,in_channels=1,sample_rate=self.sample_rate)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)

        self.selu = nn.GELU()
        #只一个encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))
        
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=(1,1)),
        )
        self.drop = nn.Dropout(0.5)
        self.back = STViT(
                    in_chans=64,
                    embed_dim=[96, 192, 384, 512], # 52M, 9.9G, 361 FPS
                    depths=[4, 6, 14, 6],
                    num_heads=[2, 3, 6, 8],
                    n_iter=[1, 1, 1, 1], 
                    stoken_size=[8, 4, 1, 1],
                    projection=1024,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.1,
                    drop_path_rate=0.5,
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0, 0],
                    layerscale=[False, False, True, True],
                    init_values=1e-6,)
        
        self.norm = nn.BatchNorm2d(64)



        self.conv2d = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(9,9),stride=(3,2))

        self.out_layer = nn.Linear(4 * 32, 2)


    def forward(self, x,y = 0,train=False):
        
        if(train):
                #x = self.noise.pickRandomNoiser(x)
            a = 1
            if a > 0.15:
                Fpass = random.randint(3400,4000)
                b = random.uniform(1.05,1.2)
                filter = LowPass(sample_hz=self.sample_rate, cutoff_hz=min(Fpass * b,7999))
                
            else :
                Fpass = random.randint(20,300)
                b = random.uniform(0.5,0.8)
                filter =HighPass(sample_hz=self.sample_rate, cutoff_hz=Fpass * b)
  
            x = x.data.cpu().numpy()
            x = torch.from_numpy(filter(x)).float().to(self.device)
        x = x.unsqueeze(1)
        filters,x = self.conv_time(x)

        
        # version 1
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = self.first_bn(x)
        x = self.selu(x)
        x = self.encoder(x)
        
        x = self.first_bn1(x)
        w = self.selu(x)
        
        if(train):

            w,y = specmix(w.permute(0,2,3,1),y)
            w = w.permute(0,3,1,2)

        output = self.back(w)

        return y, output
    
    