import torch.nn as nn

class DWS(nn.Module):
    def __init__(self, input_channel, output_channel, downsampling=False):
        super().__init__()
        if downsampling:
            conv_stride = 2
            conv_padding = 0
        else:
            conv_stride=  1
            conv_padding = 1

        self.dwconv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, stride=conv_stride, padding=conv_padding, groups=input_channel)
        self.bn1 = nn.BatchNorm2d(num_features=input_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.pwconv = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pwconv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class MultiMobile(nn.Module):
    def __init__(self, output_dim=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.features = nn.Sequential(
            DWS(input_channel=32, output_channel=64, downsampling=False),
            DWS(input_channel=64, output_channel=128, downsampling=True),
            DWS(input_channel=128, output_channel=128, downsampling=False),
            DWS(input_channel=128, output_channel=128, downsampling=True),
            DWS(input_channel=128, output_channel=256, downsampling=False),
            DWS(input_channel=256, output_channel=256, downsampling=True),
            DWS(input_channel=256, output_channel=512, downsampling=False),
            DWS(input_channel=512, output_channel=512, downsampling=False),
            DWS(input_channel=512, output_channel=512, downsampling=False),
            DWS(input_channel=512, output_channel=512, downsampling=False),
            DWS(input_channel=512, output_channel=512, downsampling=False),
            DWS(input_channel=512, output_channel=512, downsampling=True),
            DWS(input_channel=512, output_channel=1024, downsampling=False)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x