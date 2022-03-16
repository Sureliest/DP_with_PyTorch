import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils import data, tensorboard
from model import AlexNet
# import os
import json
# import time


BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "test": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
# data_root = os.getcwd()
image_path = r"D:\Project\python\StudyOfVehicle\CNN\Data"
train_dataset = datasets.ImageFolder(root=image_path + "/train", transform=data_transform["train"])
train_num = len(train_dataset)
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())

json_str = json.dumps(cla_dict, indent=4)
with open("class_index.json", "w") as json_file:
    json_file.write(json_str)
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.ImageFolder(root=image_path + "/val", transform=data_transform["test"])
test_num = len(test_dataset)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()

net = AlexNet(num_classes=5, init_weights=True)

net = net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)
# save_path = './AlexNet.pth'

EPOCH = 10
for epoch in range(EPOCH):
    net.train()
    total_loss = 0
    step = 0
    for data in train_loader:
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss
        step += 1
        print("step:{}  loss:{} total_loss:{}".format(step, loss, total_loss))

net.eval()
acc = 0.0
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = test_data
        outputs = net(test_images.to(device))
        pre_y = torch.max(outputs, dim=1)[1]
        acc += (pre_y == test_labels.to(device)).sum().item()
    test_accurate = acc/test_num
    print("test_accuracy:", test_accurate)

torch.save(net.state_dict(), 'Alexnet.pth')

