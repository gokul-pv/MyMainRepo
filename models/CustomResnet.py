'''ResNet in PyTorch.
Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>
'''


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        # Prep Layer
        self.conv1 = self.create_conv(3,64, MaxPool=False)      
        # Layer 1 
        self.conv2 = self.create_conv(64, 128, MaxPool=True)
        self.in_channels = 128
        self.res1  = self._make_layer(block, 128, num_blocks[0], stride = 1) 
        # Layer 2 
        self.conv3 = self.create_conv(128, 256, MaxPool=True)
        # Layer 3
        self.conv4 = self.create_conv(256, 512, MaxPool=True)
        self.in_channels = 512
        self.res2  = self._make_layer(block, 512, num_blocks[1], stride = 1)   

        self.pool = nn.MaxPool2d(4,4) 
        self.linear = nn.Linear(512, num_classes,bias=False) 

    def create_conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, MaxPool=False):
        if MaxPool:
            self.conv = nn.Sequential(
                                      nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                      nn.MaxPool2d(2,2),
                                      nn.BatchNorm2d(out_channels), 
                                      nn.ReLU()
                                      )
        else:
            self.conv = nn.Sequential(
                                      nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                      nn.BatchNorm2d(out_channels), 
                                      nn.ReLU()
                                      )
        return self.conv  

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers  = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        
        conv1x = self.conv1(x)

        conv2x = self.conv2(conv1x)
        res1   = self.res1(conv2x)
        res1X  = res1 + conv2x

        conv3x = self.conv3(res1X)
        
        conv4x = self.conv4(conv3x)
        res2   = self.res2(conv4x)
        res2X = res2 + conv4x

        outX = self.pool(res2X)
        outX = outX.view(outX.size(0), -1)
        outX = self.linear(outX)

        return outX

def ResNet_custom():
    return ResNet(BasicBlock, [2, 2])
