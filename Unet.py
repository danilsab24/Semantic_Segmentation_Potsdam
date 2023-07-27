import torch 
import torch.nn as nn
import torchvision.transform.functional as F

class DoubleCovolution(nn.module):
    def __init__(self, in_ch, out_ch):
        super(DoubleCovolution, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
            nn.BatchNorm2d(out_ch)
        )