import  torch
from torch import nn
import torch.nn.functional as F

class resblk(nn.Module):

    def __init__(self, ch_in, ch_out,stride = 1):

        super(resblk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride,padding=0),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out = self.extra(x) + out

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # a = self.extra(x)
        # print(out.shape,a.shape)
        out = self.extra(x) + out

        return out

class rest18(nn.Module):
    def __init__(self):
        super(rest18,self).__init__()
        #preprossce
        self.con1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(64)
        )
        #[b,64,w,h]
        #[64-128]
        self.blk1 = resblk(64, 128, stride=2)
        self.blk2 = resblk(128, 256, stride=2)
        self.blk3 = resblk(256, 512, stride=2)
        self.blk4 = resblk(512, 512, stride=1)

        self.connect = nn.Linear(512, 10)

    def forward(self, x):
        #[2,3,32,32]->[2,64,32,32]
        x = F.relu(self.con1(x))
        #[2,64,w,h]->[2,512,w,h]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        #print('x after blk4:',x.shape) [b,512,2,2]=>[b,512,1,1]
        x = F.adaptive_avg_pool2d(x, [1,1])
        #print('x after adaptive_avg_pool2d:',x.shape)
        x = x.view(x.size(0),-1)
        #print('x after view:', x.shape) [b,512]
        x = self.connect(x)
        #print('x after connect:', x.shape)[b,10]
        return x



def main():
    # tmp = torch.randn(2,32,24,24)
    # model = resblk(32,64,stride = 2)
    # out = model(tmp)
    # print(out.shape)

    tmp2 =torch.randn(2,3,32,32)
    model2 = rest18()
    out2 = model2(tmp2)
    print(out2.shape)

if __name__ == '__main__':
    main()

