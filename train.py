import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import random
from PIL import Image
# from sklearn.model_selection import train_test_split
import torchvision.models as models
import torch.nn as nn

x_list = []
y_list = []
with open('hw1_dataset/classes.txt') as f:
    # list of class names
    classes = [l.strip() for l in f.readlines()]

with open('hw1_dataset/training_labels.txt') as f:
    for l in f.readlines():
        line = l.strip()
        # list of training image names
        x_list.append(line.split(' ')[0])
        y = line.split(' ')[1]
        # list of class number (0~199)
        y_list.append(classes.index(y))

dic = {
    "filename": x_list,
    "label": y_list,
}
df = pd.DataFrame(dic)


class image_dataset(Dataset):
    def __init__(self, dataframe, training=False,
                 rootPath='hw1_dataset/training_images', img_size=224):
        self.rootPath = rootPath
        self.dataframe = dataframe
        self.img_size = img_size
        self.training = training

    def __getitem__(self, index):
        filename = self.dataframe.iloc[index, 0]
        img = Image.open(os.path.join(self.rootPath, filename))
        img = img.convert('RGB')
        img = self.trans_img(img)
        return img, self.dataframe.iloc[index, 1]

    def __len__(self):
        return len(self.dataframe.index)

    def trans_img(self, img):
        if self.training:
            augmentation = [
                transforms.RandomRotation(10, expand=False),
                transforms.RandomResizedCrop((self.img_size, self.img_size)),
                ]
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4, saturation=0.4),
                transforms.RandomApply(augmentation, p=0.8),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                # Normalize for RGB image
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                # Normalize for RGB image
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
                ])

        return transform(img)

batch_size = 12

training_data = image_dataset(df, training=True)
train_dataloader = DataLoader(training_data,
                              batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = models.resnet50(pretrained=True)
# number of neuron that input to last FC
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 200)
model = model.to(device)

init_lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
criterion = nn.CrossEntropyLoss().to(device)


def rand_bbox(size, lam):
    # width
    W = size[2]
    # height
    H = size[3]
    # ratio of cutted image and original image
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # (cx, cy) is the center of bbox
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # cut image and let bbox limited in the border of image
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

epochs = 60
train_accuracy_history = []

for epoch in range(epochs):
    # Adjust learning rate every 20 epochs
    new_lr = init_lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    model.train()
    totalLoss = 0
    count = 0
    correct_count = 0

    for x, label in train_dataloader:
        x = x.to(device)
        label = label.to(device).type(torch.long)
        optimizer.zero_grad()

        # Apply CutMix with p=0.25
        if random.uniform(0, 1) < 0.25:
            # lambda value to mix two samples
            lam = random.uniform(0, 1)
            # mess up the permutation of data in one batch
            rand_index = torch.randperm(x.size()[0]).to(device)
            label_a = label
            label_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            # paste a patch of image xb onto image xa
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :,
                                              bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match area ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                       (x.size()[-1] * x.size()[-2]))
            # compute output
            output = model(x)
            loss = (criterion(output, label_a) * lam +
                    criterion(output, label_b) * (1. - lam))
        else:
            output = model(x)
            loss = criterion(output, label)

        _, predicted = torch.max(output.data, 1)
        count += len(x)
        correct_count += (predicted == label).sum().item()
        totalLoss += loss.item()*len(label)
        loss.backward()
        optimizer.step()

    train_loss = totalLoss / count
    accuracy = correct_count / count
    train_accuracy_history.append(accuracy)

    with open('log.txt', 'a') as f:
        f.write("Epoch {}: Training Loss: {:.4f}, accuracy: {:.4f}%\n"
                .format(epoch+1, train_loss, 100*accuracy))
    print("Epoch {}: Training Loss: {:.4f}, accuracy: {:.4f}%"
          .format(epoch+1, train_loss, 100*accuracy))

    torch.save(model.state_dict(), "model_ep{}_loss{:.4f}.pkl"
               .format(epoch+1, train_loss))

    print("-------")

epoch_history = [*range(1, epochs+1, 1)]
line1, = plt.plot(epoch_history, train_accuracy_history, label='Training')
plt.legend(handles=[line1])
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.savefig('acc.png')
plt.show()
