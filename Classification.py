#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

num_epochs = 10
batch_size = 4
learning_rate = 0.001


# Tensor -1~1 change
transform = transforms.Compose(
    [transforms.Resize((1000,1000)),
    transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

# Data Load
image_dataset = torchvision.datasets.ImageFolder(root="/home/iai/Desktop/Jeewon/Seminar/20230112_MVtecAD/data/simpleclassification_good_bad/datset", transform = transform)
print(f'Entire classes : {image_dataset.classes}')

train_size = int(0.8 * len(image_dataset))
test_size = len(image_dataset) - train_size

train_dataset, test_dataset = random_split(image_dataset, [train_size, test_size])
# print(train_dataset[0][0].size())
# print(test_dataset[0][0].size())
print(f'Size of entire dataset : {len(image_dataset)}')
print(f'Size of train dataset : {len(train_dataset)}')
print(f'Size of train dataset : {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 2, shuffle = False)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(images.shape)

# conv1 = nn.Conv2d(3,6,4)
# pool = nn.MaxPool2d(3,3)
# conv2 = nn.Conv2d(6,16,4)
# pool = nn.MaxPool2d(3,3)
# conv3 = nn.Conv2d(16,26,4)
# pool = nn.MaxPool2d(3,3)
# conv4 = nn.Conv2d(26,36,4)
# print(images.shape)
# x = conv1(images)
# print(x.shape)
# x = pool(x)
# print(x.shape)
# x = conv2(x)
# print(x.shape)
# x = pool(x)
# print(x.shape)
# x = conv3(x)
# print(x.shape)
# x = pool(x)
# print(x.shape)
# x = conv4(x)
# print(x.shape)
# x = pool(x)
# print(x.shape)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(3,3)
        self.conv2 = nn.Conv2d(6,16,5)
        self.conv3 = nn.Conv3d(16,26,5)
        self.conv4 = nn.Conv2d(26,36,5)
        self.fc1 = nn.Linear(36*10*10, 1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250,120)
        self.fc5 = nn.Linear(120, 84)
        self.fc6 = nn.Linear(84, 2)
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 36*10*10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
            
print("Training done")