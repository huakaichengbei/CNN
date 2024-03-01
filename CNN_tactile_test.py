import torch
import torchvision
from PIL import Image
from torch import nn

path="./testphoto/finger27.jpg"
image=Image.open(path)
image.show()
print(image)
image = image.convert('L')
transform = torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

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

model = torch.load("tactile_45.pth", map_location=torch.device('cpu'))
#注意：若加载利用gpu训练得来的模型在cpu环境下跑，需加map_location加以说明,例如利用谷歌colab
print(model)

image = torch.reshape(image, (1, 1, 50, 28))

model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))

#可视化
classfication = {"0":"adhesive","1":"allen_key","2":"arm","3":"ball","4":"bottle","5":"box","6":"branch",
                 "7":"cable", "8":"cable_pipe","9":"caliper","10":"can","11":"finger","12":"hand",
                 "13":"highlighter","14":"key","15":"pen","16":"pliers","17":"rock","18":"rubber",
                 "19":"scissors","20":"sticky_tape","21":"tube"}


print("This is a {}".format(classfication[str(output.argmax(1).item())]))#.item()把tensor变数字