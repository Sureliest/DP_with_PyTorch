import matplotlib.pyplot as plt
import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import json

data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img = Image.open(r"D:\Project\python\StudyOfVehicle\CNN\AlexNet\rose.png")
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)
class_indict = {}
try:
    json_file = open(r'D:\Project\python\StudyOfVehicle\CNN\AlexNet\class_index.json', 'r')
    class_indict = json.load(json_file)
    # print(class_indict)
except Exception as e:
    print(e)
    exit(-1)

model = AlexNet(num_classes=5)
model_weight = r'D:\Project\python\StudyOfVehicle\CNN\AlexNet\Alexnet.pth'
model.load_state_dict(torch.load(model_weight))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))
    print(output)
    predict = torch.softmax(output, dim=0)  # 一维时使用dim=0
    print(predict)
    pre_cla = torch.argmax(predict).numpy()
    print(class_indict[str(pre_cla)], predict[pre_cla].item())
