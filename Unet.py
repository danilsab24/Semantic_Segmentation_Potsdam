import torch 
import torch.nn as nn
import torchvision.transforms.functional as F

class DoubleCovolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleCovolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
        )
    
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
        self, in_ch=3, out_ch=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of Unet
        for feature in features:
            self.downs.append(DoubleCovolution(in_ch, feature))
            in_ch = feature
        
        # Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleCovolution(feature*2, feature))
        self.bottleneck = DoubleCovolution(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)
    
def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_ch=1, out_ch=1)
    pred = model(x)
    print(pred.shape)
    print(x.shape)
    assert pred.shape == x.shape
        
if __name__ == "__main__":
    test()

        