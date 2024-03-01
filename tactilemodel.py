import torch
from torch import nn



class TactNet4(nn.Module):
    def __init__(self):
        super(TactNet4, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(8,16,kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Flatten(0,1),
            nn.Linear(128, 22)
        )

    def forward(self, x):
        x = self.model(x).view(1,-1)
        #x = nn.functional.softmax(x, dim=1)
        return x

tactile = TactNet4()
print(tactile)

if __name__ == '__main__':#验证网络搭建的是否正确
    tactile = TactNet4()
    input = torch.ones((1, 28, 50))
    output = tactile(input)
    print(output.shape)