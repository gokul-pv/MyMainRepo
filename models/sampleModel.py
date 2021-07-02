'''
Example
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class sample_net(nn.Module):

    def __init__(self):
        super(sample_net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1,padding=1, bias=False),       #Input: 32*32*3  Output: 32 * 32 * 16    RF = 3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p = 0.05),
 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1,padding=1, bias=False),      #Input: 32*32*16  Output: 32 * 32 * 32    RF = 5 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.05),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,padding=1, bias=False),      #Input: 32*32*16  Output: 32 * 32 * 32    RF = 5 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.05)             
        )
        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, bias=False), #Input: 12*12*10  Output: 10 * 10 * 12   RF = 10 
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), stride=1, padding=0, bias=False), #Input: 12*12*10  Output: 10 * 10 * 12   RF = 10 


            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1,padding=1, bias=False),       #Input: 32*32*3  Output: 32 * 32 * 16    RF = 3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p = 0.05),
 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1,padding=1, bias=False),      #Input: 32*32*16  Output: 32 * 32 * 32    RF = 5 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.05)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, bias=False), #Input: 12*12*10  Output: 10 * 10 * 12   RF = 10 
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), stride=1, padding=0, bias=False), #Input: 12*12*10  Output: 10 * 10 * 12   RF = 10 
            

            #DepthWise Seperable Network
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1,groups = 16, bias=False ), # jout = 4, rf = 20+(2)*4 = 28, o/p = 7
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False), #jout = 4, rf = 28+(0)*4 = 28, o/p = 9
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p = 0.05),

            #DepthWise Seperable Network
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1,groups=32, bias=False),#jout = 4, rf = 28+(2)*4 = 36, o/p = 9
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False), #jin = jout = 4, rf = 36+(0)*4 = 36, o/p = 11
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.05)
            )

        self.convblock4 = nn.Sequential(
            #Dialated Network
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1,dilation=2, bias=False), #jout = 8, rf = 40+(4)*8 = 72, o/p = 3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.05),

            #AVG Pool
            nn.AvgPool2d(6), #op = 1
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)        #Op_size = 1, 
            )

    def forward(self, x):
        x = self.convblock1(x)    # i/p= 32 o/p=32 Rf = 6
        x = self.convblock2(x)    # Rf = 18 jout = 2, o/p =14
        x = self.convblock3(x)    # jout = 4, Rf = 36, o/p = 11
        x = self.convblock4(x)    # o/p = 1
     
        x = x.view(-1, 10)
        return x


def exampleNet():
  return sample_net()
