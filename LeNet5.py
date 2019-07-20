import torch
from torch import nn,optim

class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5, self).__init__()


        self.conv1 = nn.Sequential(
            #[b,3,32,32]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(6, 16,kernel_size=5, stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        # x = torch.randn(2, 3, 32, 32)
        # out = self.conv1(x)
        # print(out.shape)
    def forward(self,x):
        batch = x.size(0)
        #[b,3,32,32] =>[b,16,5,5]
        x = self.conv1(x)
        x = x.view(batch, 16*5*5)
        #[b,16,5,5]=>[b,10]
        logist = self.conv2(x)

        return logist


def main():
    net = Lenet5()
    tmp = torch.randn(2,3,32,32)
    out = net(tmp)
    print(out.shape)
if __name__ == '__main__':
    main()
