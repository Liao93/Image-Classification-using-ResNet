import os
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
import torchvision.models as models
import torch.nn as nn

img_size = 224

with open('hw1_dataset/testing_img_order.txt') as f:
    # all the testing images
    test_images = [x.strip() for x in f.readlines()]

with open('hw1_dataset/classes.txt') as f:
    # list of class names
    classes = [l.strip() for l in f.readlines()]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 200)
model = model.to(device)
model.load_state_dict(torch.load("model/model_0.659.pkl"))

submission = []

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

model.eval()
with torch.no_grad():
    # image order is important to your result
    for filename in test_images:
        img = Image.open(os.path.join('hw1_dataset/testing_images', filename))
        img = img.convert('RGB')
        img = transform(img)
        img = img.unsqueeze_(0).to(device)
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        # the predicted category
        classname = classes[predicted.cpu().item()]
        submission.append([filename, classname])

np.savetxt('answer.txt', submission, fmt='%s')
